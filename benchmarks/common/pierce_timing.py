from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from typing import Any, Dict, Mapping


TIMING_POLICY = "steady_query_v1"


@dataclass(frozen=True)
class PierceTimingSummary:
    query_time_ms: float
    breakdown: Dict[str, float]
    overhead_breakdown: Dict[str, float]
    num_gpus_active: int | None


def normalize_phase_name(name: str) -> str:
    key = str(name).lower().strip()
    key = re.sub(r"_\d+$", "", key)
    key = re.sub(r"_+$", "", key)
    return key


def phase_values_from_json(data: Mapping[str, Any]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for key, phase_data in data.get("phases", {}).items():
        if not isinstance(phase_data, Mapping):
            continue
        normalized = normalize_phase_name(key)
        values[normalized] = values.get(normalized, 0.0) + float(phase_data.get("duration_ms", 0.0) or 0.0)
    return values


def _counter_us(counters: Mapping[str, Any], key: str) -> float:
    value = counters.get(key)
    if value is None:
        value = counters.get(key.lower())
    if isinstance(value, (int, float)):
        return float(value)
    return 0.0


def _counter_ms(counters: Mapping[str, Any], key: str) -> float:
    return _counter_us(counters, key) / 1000.0


def _slab_worker_counters(counters: Mapping[str, Any]) -> Dict[int, Dict[str, float]]:
    slabs: Dict[int, Dict[str, float]] = {}
    pattern = re.compile(r"^profile_slab_(\d+)_worker_(.+)_us$")
    for key, value in counters.items():
        match = pattern.match(str(key).lower())
        if not match or not isinstance(value, (int, float)):
            continue
        slab = int(match.group(1))
        suffix = match.group(2)
        slabs.setdefault(slab, {})[suffix] = float(value) / 1000.0
    return slabs


def _download_ms(worker: Mapping[str, float]) -> float:
    return sum(
        worker.get(key, 0.0)
        for key in (
            "download_pairs",
            "overflow_download",
            "fingerprint_download",
            "hit_histogram_download",
            "profiling_stats_download",
        )
    )


def _compaction_ms(worker: Mapping[str, float]) -> float:
    return sum(
        worker.get(key, 0.0)
        for key in (
            "hash_result_buffer_alloc",
            "hash_compact_table",
            "hash_count_table_pairs",
            "hash_failure_copy",
        )
    )


def _worker_query_ms(worker: Mapping[str, float]) -> float:
    measured = worker.get("measured_hash_query_total", 0.0)
    return measured + _download_ms(worker)


def _legacy_query_breakdown(phase_values: Mapping[str, float]) -> Dict[str, float]:
    breakdown: Dict[str, float] = {}
    for key, value in phase_values.items():
        if value <= 0.0:
            continue
        if key == "selectivity estimation":
            breakdown[key] = breakdown.get(key, 0.0) + value
        elif key.startswith("raytrace_"):
            breakdown[key] = breakdown.get(key, 0.0) + value
        elif key in (
            "compact_hash_table_pairs",
            "compact_hash_table_pairs (containment)",
            "compact_hash_table_pairs (overlap)",
            "download results",
            "gpu deduplication",
        ):
            breakdown[key] = breakdown.get(key, 0.0) + value

    return breakdown


def summarize_pierce_timing(
    data: Mapping[str, Any],
    *,
    include_global_dedup: bool = True,
    require_worker_counters: bool = True,
) -> PierceTimingSummary:
    phase_values = phase_values_from_json(data)
    counters = data.get("counters", {})
    if not isinstance(counters, Mapping):
        counters = {}
    counters = {str(key).lower(): value for key, value in counters.items()}

    selectivity_ms = phase_values.get("selectivity estimation", 0.0)
    active_gpus_raw = counters.get("profile_active_gpu_count")
    num_gpus_active = int(active_gpus_raw) if isinstance(active_gpus_raw, (int, float)) else None
    global_dedup_ms = 0.0
    if include_global_dedup:
        global_dedup_ms = _counter_ms(counters, "Profile_Global_Dedup_Total_Us")
        if global_dedup_ms <= 0.0:
            global_dedup_ms = phase_values.get("global gpu deduplication", 0.0)

    slab_workers = _slab_worker_counters(counters)
    breakdown: Dict[str, float] = {}
    if slab_workers:
        selected_worker = max(slab_workers.values(), key=_worker_query_ms)
        selected_worker_total = _worker_query_ms(selected_worker)
        compaction_and_download_ms = _compaction_ms(selected_worker) + _download_ms(selected_worker)
        raytrace_ms = max(0.0, selected_worker_total - compaction_and_download_ms)

        if selectivity_ms > 0.0:
            breakdown["selectivity estimation"] = selectivity_ms
        if raytrace_ms > 0.0:
            breakdown["measured hash/raytrace query"] = raytrace_ms
        if compaction_and_download_ms > 0.0:
            breakdown["hash compaction/result movement"] = compaction_and_download_ms
        if global_dedup_ms > 0.0:
            breakdown["global gpu deduplication"] = global_dedup_ms
    else:
        if require_worker_counters:
            raise ValueError(
                "steady_query_v1 requires Profile_Slab_*_Worker_Measured_Hash_Query_Total_Us counters"
            )
        breakdown = _legacy_query_breakdown(phase_values)
        if include_global_dedup and global_dedup_ms > 0.0 and "global gpu deduplication" not in breakdown:
            breakdown["global gpu deduplication"] = global_dedup_ms

    query_time_ms = sum(breakdown.values())
    if query_time_ms <= 0.0:
        raise ValueError("Expected Pierce query timing phases or counters were not found")

    query_keys = set(breakdown)
    overhead_breakdown = {
        key: value
        for key, value in phase_values.items()
        if value > 0.0 and key not in query_keys
    }

    return PierceTimingSummary(
        query_time_ms=query_time_ms,
        breakdown=breakdown,
        overhead_breakdown=overhead_breakdown,
        num_gpus_active=num_gpus_active,
    )


def aggregate_breakdowns(items: list[Mapping[str, float]]) -> Dict[str, float]:
    keys = sorted({key for item in items for key in item})
    return {
        key: float(statistics.mean(float(item.get(key, 0.0)) for item in items))
        for key in keys
    }
