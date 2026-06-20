#!/usr/bin/env python3
import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import apply_paper_style, hatch_for_index, query_style_for


STACK_COMPONENTS = [
    (
        "Selectivity estimation",
        ["selectivity estimation"],
    ),
    (
        "Edge raytracing",
        [
            "raytrace_hash_mesh1tomesh2",
            "raytrace_hash_mesh2tomesh1",
            "raytrace_overlap_hash_mesh1tomesh2",
            "raytrace_overlap_hash_mesh2tomesh1",
        ],
    ),
    (
        "Containment raytracing",
        [
            "raytrace_containment_hash_mesh1tomesh2",
            "raytrace_containment_hash_mesh2tomesh1",
        ],
    ),
    (
        "Download results",
        ["download results"],
    ),
]

STACK_HATCHES = {
    "Selectivity estimation": "///",
    "Edge raytracing": "",
    "Containment raytracing": "xxx",
    "Download results": "...",
}

QUERY_LOG_SUBDIR = {
    "overlap": "Pierce_direct_estimation",
    "intersection": "Pierce_estimated",
    "containment": "Pierce",
}

TRIM_MAX_RULES = {
    ("nu=800", "intersection"): 1,
}

OVERALL_PERFORMANCE_SPECS = [
    {
        "run_prefix": "query_comparison_nu_v_scalability",
        "row_matcher": lambda r: r.get("nu") == 800 and r.get("dataset_profile") == "large_nu_v",
        "group_name": r"Vessel $\bowtie$ Nuclei$_1$",
        "selector": "nu=800",
    },
    {
        "run_prefix": "query_comparison_nu_scalability_nn",
        "row_matcher": lambda r: r.get("nu") == 400 and r.get("dataset_profile") == "large_nu_nn",
        "group_name": r"Nuclei$_2$ $\bowtie$ Nuclei$_3$",
        "selector": "nu=400",
    },
    {
        "run_prefix": "query_comparison_microns",
        "row_matcher": lambda r: r.get("size_gb") == 4,
        "group_name": r"Neurons$_1$ $\bowtie$ Neurons$_2$",
        "selector": "size_gb=4",
    },
    {
        "run_prefix": "query_comparison_microns",
        "row_matcher": lambda r: r.get("size_gb") == 8,
        "group_name": r"Neurons$_3$ $\bowtie$ Neurons$_4$",
        "selector": "size_gb=8",
    },
]


@dataclass
class PointResult:
    group_name: str
    selector: str
    source_run: Path
    query_to_mean: Dict[str, float]
    query_to_components: Dict[str, Dict[str, float]]


def _valid_mean(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v


def _normalize_breakdown_entry(entry: Any) -> Dict[str, float]:
    if not isinstance(entry, dict):
        return {}

    normalized: Dict[str, float] = {}
    for key, value in entry.items():
        if isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
            continue
        if isinstance(value, dict):
            maybe_mean = value.get("mean")
            if isinstance(maybe_mean, (int, float)):
                normalized[str(key)] = float(maybe_mean)
    return normalized


def _normalize_phase_name(label: str) -> str:
    normalized = label.strip().lower()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized


def _parse_phase_breakdown_from_log(log_path: Path) -> Dict[str, float]:
    pattern = re.compile(r"^\[[^\]]+\]\s+(.*?):\s+[-0-9.]+\s+microseconds\s+\(([-0-9.]+)\s+ms\)")
    breakdown: Dict[str, float] = {}

    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return {}

    for line in lines:
        match = pattern.match(line.strip())
        if not match:
            continue
        label, value_ms = match.groups()
        try:
            breakdown[_normalize_phase_name(label)] = float(value_ms)
        except ValueError:
            continue

    return breakdown


def _recompute_breakdown_without_max_run(
    *,
    cell: Dict[str, Any],
    selector: str,
    query: str,
    source_run: Path,
) -> Optional[tuple[float, Dict[str, float]]]:
    trim_count = TRIM_MAX_RULES.get((selector, query), 0)
    if trim_count <= 0:
        return None

    raw_times = cell.get("raw_times")
    if not isinstance(raw_times, list) or len(raw_times) <= trim_count:
        return None

    numeric_times: List[float] = []
    for value in raw_times:
        try:
            numeric_times.append(float(value))
        except (TypeError, ValueError):
            return None

    drop_indices = set(np.argsort(numeric_times)[-trim_count:].tolist())
    kept_indices = [idx for idx in range(len(numeric_times)) if idx not in drop_indices]
    if not kept_indices:
        return None

    log_dir = source_run / "logs" / selector.replace("=", "_") / QUERY_LOG_SUBDIR[query]
    phase_samples: Dict[str, List[float]] = {}
    for idx in kept_indices:
        log_path = log_dir / f"run_{idx:03d}.log"
        phases = _parse_phase_breakdown_from_log(log_path)
        if not phases:
            return None
        for phase_name, duration_ms in phases.items():
            phase_samples.setdefault(phase_name, []).append(duration_ms)

    if not phase_samples:
        return None

    recomputed_breakdown = {
        phase_name: float(np.mean(samples))
        for phase_name, samples in phase_samples.items()
        if samples
    }
    recomputed_mean = float(np.mean([numeric_times[idx] for idx in kept_indices]))
    return recomputed_mean, recomputed_breakdown


def _extract_components(total_mean: float, breakdown: Dict[str, float]) -> Dict[str, float]:
    components: Dict[str, float] = {}
    explained = 0.0

    for label, aliases in STACK_COMPONENTS:
        value = sum(float(breakdown.get(alias, 0.0)) for alias in aliases)
        if value > 0.0:
            components[label] = value
            explained += value

    residual = max(0.0, total_mean - explained)
    if residual > 1e-9:
        components["Download results"] = components.get("Download results", 0.0) + residual

    return components


def _extract_from_row(row: Dict, *, group_name: str, selector: str, source_run: Path) -> Optional[PointResult]:
    query_to_mean: Dict[str, float] = {}
    query_to_components: Dict[str, Dict[str, float]] = {}
    for q in ["overlap", "intersection", "containment"]:
        cell = row.get(q)
        if not isinstance(cell, dict) or "error" in cell:
            continue
        m = _valid_mean(cell.get("mean"))
        if m is not None:
            adjusted = _recompute_breakdown_without_max_run(
                cell=cell,
                selector=selector,
                query=q,
                source_run=source_run,
            )
            if adjusted is not None:
                m, adjusted_breakdown = adjusted
                breakdown = adjusted_breakdown
            else:
                breakdown = _normalize_breakdown_entry(cell.get("breakdown"))
            query_to_mean[q] = m
            query_to_components[q] = _extract_components(m, breakdown)
    if not query_to_mean:
        return None
    return PointResult(
        group_name=group_name,
        selector=selector,
        source_run=source_run,
        query_to_mean=query_to_mean,
        query_to_components=query_to_components,
    )


def _pick_latest_point(
    runs_root: Path,
    run_prefix: str,
    *,
    row_matcher,
    group_name: str,
    selector: str,
) -> PointResult:
    # Use exact prefix matching to avoid accidental cross-matches such as
    # `query_comparison_nu_v_scalability_*` picking `_nn_*` runs.
    run_dirs = sorted(
        [
            p
            for p in runs_root.iterdir()
            if p.is_dir() and p.name.startswith(f"{run_prefix}_")
        ],
        reverse=True,
    )
    for run_dir in run_dirs:
        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue
        try:
            payload = json.loads(result_file.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload.get("results", [])
        if not isinstance(rows, list):
            continue
        row = next((r for r in rows if row_matcher(r)), None)
        if row is None:
            continue
        point = _extract_from_row(row, group_name=group_name, selector=selector, source_run=run_dir)
        if point is not None:
            return point
    raise RuntimeError(f"No usable run found for {group_name} ({selector})")


def _plot(points: List[PointResult], output_base: Path) -> None:
    apply_paper_style()

    queries = [q for q in ["overlap", "intersection", "containment"] if any(q in p.query_to_mean for p in points)]
    x = np.arange(len(points), dtype=float)
    width = 0.8 / max(1, len(queries))

    fig, ax = plt.subplots(figsize=(12, 6.8))
    active_components = []
    for label, _aliases in STACK_COMPONENTS:
        if any(p.query_to_components.get(q, {}).get(label, 0.0) > 0.0 for p in points for q in queries):
            active_components.append(label)

    for i, q in enumerate(queries):
        vals = np.array([p.query_to_mean.get(q, np.nan) for p in points], dtype=float)
        mask = np.isfinite(vals)
        if not np.any(mask):
            continue

        offs = (i - (len(queries) - 1) / 2.0) * width
        st = query_style_for(q)
        bottoms = np.zeros(len(points), dtype=float)
        bottoms[~mask] = np.nan

        for component_idx, component_name in enumerate(active_components):
            comp_vals = np.array(
                [p.query_to_components.get(q, {}).get(component_name, 0.0) if np.isfinite(p.query_to_mean.get(q, np.nan)) else np.nan for p in points],
                dtype=float,
            )
            comp_mask = mask & np.isfinite(comp_vals) & (comp_vals > 0.0)
            if not np.any(comp_mask):
                continue

            ax.bar(
                x[comp_mask] + offs,
                comp_vals[comp_mask],
                width=width,
                bottom=bottoms[comp_mask],
                color=st["color"],
                alpha=0.92,
                hatch=STACK_HATCHES.get(component_name, hatch_for_index(component_idx)),
                edgecolor="black",
                linewidth=0.6,
            )
            bottoms[comp_mask] += comp_vals[comp_mask]

    ax.set_ylabel("Query time (ms)")
    ax.set_xticks(x)
    # Match overlap overall style: render each dataset pair on two lines.
    two_line_labels = [p.group_name.replace(r" $\bowtie$ ", "\n" + r"$\bowtie$" + " ") for p in points]
    ax.set_xticklabels(two_line_labels)
    ax.grid(False)

    query_handles = [
        Patch(facecolor=query_style_for(q)["color"], edgecolor="black", label=query_style_for(q)["label"])
        for q in queries
    ]
    component_handles = [
        Patch(facecolor="white", edgecolor="black", hatch=STACK_HATCHES.get(name, hatch_for_index(i)), label=name)
        for i, name in enumerate(active_components)
    ]

    if query_handles:
        query_legend = ax.legend(
            handles=query_handles,
            loc="upper left",
            ncol=1,
            title="Query Overview",
        )
        query_legend.get_title().set_fontweight("black")
        query_legend.get_title().set_fontsize(15)
        for text in query_legend.get_texts():
            text.set_fontweight("bold")
        ax.add_artist(query_legend)

    if component_handles:
        breakdown_legend = ax.legend(
            handles=component_handles,
            loc="upper right",
            ncol=1,
            title="Breakdown",
        )
        breakdown_legend.get_title().set_fontweight("black")
        breakdown_legend.get_title().set_fontsize(15)
        for text in breakdown_legend.get_texts():
            text.set_fontweight("bold")

    plt.tight_layout()
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grouped bar chart for mesh query comparison overall performance."
    )
    parser.add_argument("--runs-root", type=Path, default=SCRIPT_DIR / "runs")
    args = parser.parse_args()

    runs_root = args.runs_root

    points = []
    for spec in OVERALL_PERFORMANCE_SPECS:
        try:
            points.append(
                _pick_latest_point(
                    runs_root,
                    spec["run_prefix"],
                    row_matcher=spec["row_matcher"],
                    group_name=spec["group_name"],
                    selector=spec["selector"],
                )
            )
        except RuntimeError as exc:
            print(f"Skipping overall-performance point {spec['selector']}: {exc}")

    if not points:
        raise RuntimeError("No mesh query comparison runs were found.")

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "query_comparison_overall_performance")
    figures_dir = Path(run_layout["figures_dir"])
    output_base = figures_dir / f"predicate_comparison_overall_performance_{run_layout['timestamp']}"

    _plot(points, output_base)

    payload = {
        "metadata": {
            "scenario": "query_comparison_overall_performance",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
        },
        "groups": [
            {
                "group_name": p.group_name,
                "selector": p.selector,
                "source_run": str(p.source_run),
                "source_run_name": p.source_run.name,
                "results_ms": p.query_to_mean,
                "breakdown_ms": p.query_to_components,
            }
            for p in points
        ],
    }
    write_json(Path(run_layout["results_json"]), payload)

    print(f"Saved figure: {output_base}.png")
    print(f"Saved figure: {output_base}.pdf")
    for p in points:
        print(f"{p.group_name}: {p.source_run.name} ({p.selector}), queries={sorted(p.query_to_mean.keys())}")


if __name__ == "__main__":
    main()
