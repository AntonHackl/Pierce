from __future__ import annotations

import re
from typing import Dict


TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE = "index_compute_evaluate"
TDBASE_TIMING_MODE_COMPUTE_ONLY = "compute_only"
TDBASE_TIMING_MODES = (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODE_COMPUTE_ONLY,
)


def validate_tdbase_timing_mode(mode: str) -> str:
    if mode not in TDBASE_TIMING_MODES:
        raise ValueError(
            f"Unsupported TDBase timing mode '{mode}'. Expected one of: {', '.join(TDBASE_TIMING_MODES)}"
        )
    return mode


def _parse_metric_ms(
    text: str,
    pattern: str,
    *,
    required: bool = True,
    default_unit: str = "ms",
) -> float:
    match = re.search(pattern, text)
    if not match:
        if required:
            raise RuntimeError(f"Could not parse TDBase timing metric with pattern: {pattern}")
        return 0.0

    value_ms = float(match.group(1))
    unit = match.group(2) if match.lastindex and match.lastindex >= 2 else None
    if unit == "s" or (unit is None and default_unit == "s"):
        value_ms *= 1000.0
    return value_ms


def parse_tdbase_report_metric_ms(text: str, label: str) -> float:
    pattern = rf"{re.escape(label)}:\s*([\d.]+)\s*(s|ms)?"
    return _parse_metric_ms(text, pattern, default_unit="ms")


def parse_tdbase_first_available_metric_ms(text: str, labels: tuple[str, ...]) -> float:
    last_error: RuntimeError | None = None
    for label in labels:
        try:
            return parse_tdbase_report_metric_ms(text, label)
        except RuntimeError as exc:
            last_error = exc
    if last_error is not None:
        raise last_error
    raise RuntimeError("No TDBase timing labels were provided")


def parse_tdbase_compute_ms(text: str) -> float:
    try:
        return parse_tdbase_report_metric_ms(text, "compute")
    except RuntimeError:
        pass

    try:
        return _parse_metric_ms(text, r"computation:\s*([\d.]+)\s*(s|ms)?")
    except RuntimeError:
        pass

    legacy_matches = list(
        re.finditer(r"computation for checking intersection takes ([\d.]+) ms", text)
    )
    if not legacy_matches:
        raise RuntimeError("Could not parse TDBase compute time from run output")
    return sum(float(match.group(1)) for match in legacy_matches)


def parse_tdbase_load_tiles_ms(text: str) -> float:
    load_tiles_ms = _parse_metric_ms(
        text,
        r"load tiles takes\s+([\d.]+)\s+(s|ms)",
        required=False,
    )
    if load_tiles_ms:
        return load_tiles_ms
    return _parse_metric_ms(text, r"init tiles takes\s+([\d.]+)\s+(s|ms)", required=False)


def parse_tdbase_run_metrics(text: str) -> Dict[str, float]:
    metrics = {
        "load_tiles_ms": parse_tdbase_load_tiles_ms(text),
        "total_ms": parse_tdbase_report_metric_ms(text, "total"),
        "index_ms": parse_tdbase_report_metric_ms(text, "index"),
        "decode_ms": parse_tdbase_report_metric_ms(text, "decode"),
        # Newer TDBase builds report this phase as "packing" instead of "prepare".
        "prepare_ms": parse_tdbase_first_available_metric_ms(text, ("prepare", "packing")),
        "compute_ms": parse_tdbase_compute_ms(text),
        # Newer TDBase builds report this phase as "updatelist" instead of "evaluate".
        "evaluate_ms": parse_tdbase_first_available_metric_ms(text, ("evaluate", "updatelist")),
        "other_ms": parse_tdbase_report_metric_ms(text, "other"),
    }
    metrics["query_time_index_compute_evaluate_ms"] = (
        metrics["index_ms"] + metrics["compute_ms"] + metrics["evaluate_ms"]
    )
    metrics["query_time_compute_only_ms"] = metrics["compute_ms"]
    metrics["preprocessing_ms"] = (
        metrics["load_tiles_ms"] + metrics["index_ms"] + metrics["decode_ms"] + metrics["prepare_ms"]
    )
    metrics["loading_ms"] = metrics["preprocessing_ms"] + metrics["evaluate_ms"]
    return metrics


def query_time_for_mode(metrics: Dict[str, float], mode: str) -> float:
    validate_tdbase_timing_mode(mode)
    if mode == TDBASE_TIMING_MODE_COMPUTE_ONLY:
        return metrics["query_time_compute_only_ms"]
    return metrics["query_time_index_compute_evaluate_ms"]
