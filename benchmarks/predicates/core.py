from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, Iterable, Sequence

import matplotlib.pyplot as plt
from benchmarks.common.viz_utils import PAPER_FIGSIZE, PAPER_WIDE_FIGSIZE, apply_paper_style, hatch_for_index, make_legend_bold, query_style_for
import numpy as np

from benchmarks.predicates.adapters.containment import PierceContainmentAdapter
from benchmarks.predicates.adapters.intersection import PierceIntersectionAdapter
from benchmarks.overlap.adapters.pierce_adapter import PierceAdapter as PierceOverlapAdapter


QUERY_CHOICES = ["overlap", "intersection", "containment"]

GROUP_COLORS = {
    "Selectivity estimation": "#4C78A8",
    "Edge raytrace Mesh1->Mesh2": "#F58518",
    "Edge raytrace Mesh2->Mesh1": "#E45756",
    "Containment raytrace Mesh1->Mesh2": "#72B7B2",
    "Containment raytrace Mesh2->Mesh1": "#54A24B",
    "Download results": "#B279A2",
    "Hash compaction": "#FF9DA6",
}

GROUPED_BREAKDOWN_COMPONENTS = [
    (
        "Selectivity estimation",
        ["selectivity estimation"],
    ),
    (
        "Edge raytrace Mesh1->Mesh2",
        ["raytrace_hash_mesh1tomesh2", "raytrace_overlap_hash_mesh1tomesh2"],
    ),
    (
        "Edge raytrace Mesh2->Mesh1",
        ["raytrace_hash_mesh2tomesh1", "raytrace_overlap_hash_mesh2tomesh1"],
    ),
    (
        "Containment raytrace Mesh1->Mesh2",
        ["raytrace_containment_hash_mesh1tomesh2"],
    ),
    (
        "Containment raytrace Mesh2->Mesh1",
        ["raytrace_containment_hash_mesh2tomesh1"],
    ),
    (
        "Download results",
        ["download results"],
    ),
    (
        "Hash compaction",
        ["compact_hash_table_pairs", "compact_hash_table_pairs (containment)", "compact_hash_table_pairs (overlap)", "deduplication", "gpu deduplication"],
    ),
]


def add_query_selection_arguments(parser) -> None:
    parser.add_argument(
        "--queries",
        type=str,
        nargs="+",
        choices=QUERY_CHOICES,
        default=None,
        help="Query types to compare. Default compares all three.",
    )
    parser.add_argument(
        "--approaches",
        type=str,
        nargs="+",
        choices=QUERY_CHOICES,
        default=None,
        help="Alias for --queries for compatibility.",
    )


def resolve_queries(queries: Sequence[str] | None, approaches: Sequence[str] | None) -> list[str]:
    selected = list(queries) if queries else (list(approaches) if approaches else list(QUERY_CHOICES))
    ordered_unique = [q for q in QUERY_CHOICES if q in selected]
    if len(ordered_unique) < 2:
        raise ValueError("Select at least two query types with --queries/--approaches.")
    return ordered_unique


def build_pierce_query_adapters(
    *,
    repo_root: Path,
    data_dirs: Dict[str, Path],
    grid_cell_size: int,
    warmup_runs: int,
    overlap_mode: str,
    intersection_mode: str,
    include_overlap_pairs: bool,
    overlap_max_iterations: int = 100,
) -> Dict[str, Any]:
    pierce_dir = repo_root / "pierce"

    overlap = PierceOverlapAdapter(
        str(pierce_dir),
        mode=overlap_mode,
        preprocessed_dir=str(data_dirs["preprocessed"]),
        timings_dir=str(data_dirs["timings"]),
        grid_cell_size=grid_cell_size,
        warmup_runs=warmup_runs,
        overlap_max_iterations=overlap_max_iterations,
    )
    intersection = PierceIntersectionAdapter(
        str(pierce_dir),
        mode=intersection_mode,
        preprocessed_dir=str(data_dirs["preprocessed"]),
        timings_dir=str(data_dirs["timings"]),
        grid_cell_size=grid_cell_size,
        warmup_runs=warmup_runs,
    )
    containment = PierceContainmentAdapter(
        str(pierce_dir),
        preprocessed_dir=str(data_dirs["preprocessed"]),
        timings_dir=str(data_dirs["timings"]),
        grid_cell_size=grid_cell_size,
        warmup_runs=warmup_runs,
        include_overlap_pairs=include_overlap_pairs,
    )
    return {
        "overlap": overlap,
        "intersection": intersection,
        "containment": containment,
    }


def sanitize_case_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def ensure_preprocessed(adapters: Dict[str, Any], mesh_paths: Iterable[Path], log_dir: Path | None = None) -> None:
    # All query adapters read the same .pre output naming scheme.
    overlap_adapter = adapters["overlap"]
    for mesh_path in mesh_paths:
        if not overlap_adapter.check_preprocessed(str(mesh_path)):
            overlap_adapter.preprocess_from_source(
                str(mesh_path),
                str(mesh_path),
                log_dir=str(log_dir) if log_dir is not None else None,
            )


def run_selected_queries(
    *,
    adapters: Dict[str, Any],
    queries: Sequence[str],
    mesh1: Path,
    mesh2: Path,
    runs: int,
    timeout: float,
    overlap_query_direction: str,
    intersection_extra_args: list[str],
    log_dir: Path | None = None,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    log_dir_str = str(log_dir) if log_dir is not None else None

    if "overlap" in queries:
        results["overlap"] = adapters["overlap"].run_overlap(
            str(mesh1),
            str(mesh2),
            runs,
            timeout=timeout,
            log_dir=log_dir_str,
            query_direction=overlap_query_direction,
        )

    if "intersection" in queries:
        results["intersection"] = adapters["intersection"].run_intersection(
            str(mesh1),
            str(mesh2),
            runs,
            timeout=timeout,
            log_dir=log_dir_str,
            extra_args=intersection_extra_args,
        )

    if "containment" in queries:
        results["containment"] = adapters["containment"].run_containment(
            str(mesh1),
            str(mesh2),
            runs,
            timeout=timeout,
            log_dir=log_dir_str,
            extra_args=intersection_extra_args,
        )

    return results


def build_intersection_extra_args(
    *,
    overlap_max_iterations: int,
    hash_load_factor: float,
    enable_profiling_stats: bool,
    track_overflow: bool,
    intersection_query_direction: str,
) -> list[str]:
    args = [
        "--query-direction",
        intersection_query_direction,
        "--overlap-max-iterations",
        str(overlap_max_iterations),
        "--hash-load-factor",
        str(hash_load_factor),
    ]
    if enable_profiling_stats:
        args.append("--enable-profiling-stats")
    if track_overflow:
        args.append("--track-overflow")
    return args


def _normalize_breakdown_entry(entry: Any) -> Dict[str, float]:
    if not isinstance(entry, dict):
        return {}

    normalized: Dict[str, float] = {}
    for key, value in entry.items():
        if isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
            continue
        if isinstance(value, dict):
            # Containment uses {'phase': {'mean': ..., 'min': ...}}.
            maybe_mean = value.get("mean")
            if isinstance(maybe_mean, (int, float)):
                normalized[str(key)] = float(maybe_mean)
    return normalized


def _build_grouped_component_matrix(
    query_breakdowns: Sequence[Dict[str, float]],
) -> tuple[list[str], list[np.ndarray]]:
    group_names: list[str] = []
    group_values: list[np.ndarray] = []

    for group_name, aliases in GROUPED_BREAKDOWN_COMPONENTS:
        vals = np.array(
            [
                sum(float(entry.get(alias, 0.0)) for alias in aliases)
                for entry in query_breakdowns
            ],
            dtype=float,
        )
        if np.any(vals > 0):
            group_names.append(group_name)
            group_values.append(vals)

    return group_names, group_values


def generate_query_comparison_figures(
    *,
    results_rows: Sequence[Dict[str, Any]],
    queries: Sequence[str],
    case_labels: Sequence[str],
    figures_dir: Path,
    title_prefix: str,
    x_axis_label: str,
) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    if not results_rows or not case_labels:
        return

    x = np.arange(len(case_labels))

    # Figure 1: Total query time comparison across all selected query types.
    width = 0.8 / max(1, len(queries))
    apply_paper_style()
    fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
    for idx, query in enumerate(queries):
        q_style = query_style_for(query)
        means = []
        for row in results_rows:
            item = row.get(query, {})
            mean = item.get("mean") if isinstance(item, dict) else None
            means.append(float(mean) if isinstance(mean, (int, float)) else np.nan)
        offsets = x - (0.4 - width / 2.0) + idx * width
        ax.bar(
            offsets,
            means,
            width=width,
            label=q_style["label"],
            color=q_style["color"],
            hatch=q_style["hatch"],
            edgecolor="black",
            linewidth=0.6,
        )

        ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel("Query time (ms)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=20, ha="right")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(False)
    make_legend_bold(ax, fontsize=13)
    fig.tight_layout()
    fig.savefig(figures_dir / "query_time_comparison.png", dpi=180)
    fig.savefig(figures_dir / "query_time_comparison.pdf")
    plt.close(fig)

    # Figure 1b: Query-labeled grouped stacked breakdown comparison (one chart, all datasets).
    # This chart shows one "set of bars" per dataset, where each set compares query types.
    width = 0.8 / max(1, len(queries))
    apply_paper_style()
    fig, ax = plt.subplots(figsize=PAPER_WIDE_FIGSIZE)
    bottoms = np.zeros((len(case_labels), len(queries)))

    active_components_added = set()

    for group_name, aliases in GROUPED_BREAKDOWN_COMPONENTS:
        color = GROUP_COLORS.get(group_name, "#9D9D9D")
        for q_idx, query in enumerate(queries):
            q_style = query_style_for(query)
            vals = []
            for row in results_rows:
                item = row.get(query, {})
                breakdown = _normalize_breakdown_entry(item.get("breakdown") if isinstance(item, dict) else {})
                vals.append(sum(breakdown.get(alias, 0.0) for alias in aliases))

            vals = np.array(vals, dtype=float)
            if np.any(vals > 0):
                # Add to legend only if not already present
                label = ""
                if group_name not in active_components_added:
                    label = group_name
                    active_components_added.add(group_name)

                offsets = x - (0.4 - width / 2.0) + q_idx * width
                ax.bar(
                    offsets,
                    vals,
                    width=width,
                    bottom=bottoms[:, q_idx],
                    label=label,
                    color=color,
                    hatch=q_style["hatch"],
                    edgecolor="black",
                    linewidth=0.5,
                )
                bottoms[:, q_idx] += vals

        ax.set_xlabel(x_axis_label, fontsize=16)
    ax.set_ylabel("Breakdown time (ms)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(case_labels, rotation=20, ha="right")
    ax.tick_params(axis="both", labelsize=13)
    ax.grid(False)
    make_legend_bold(ax, loc="upper right", fontsize=13, title="Phases")

    # Add annotation for query order
    query_order_str = " | ".join([f"{q[0].upper()}: {q}" for q in queries])
    ax.text(0.5, -0.22, f"Queries in each group: {query_order_str}",
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')

    fig.tight_layout()
    fig.savefig(figures_dir / "query_time_breakdown_comparison.png", dpi=180)
    fig.savefig(figures_dir / "query_time_breakdown_comparison.pdf")
    plt.close(fig)

    # Figure 2+: One stacked breakdown chart per query.
    for query in queries:
        query_breakdowns = []
        for row in results_rows:
            item = row.get(query, {})
            normalized = _normalize_breakdown_entry(item.get("breakdown") if isinstance(item, dict) else {})
            query_breakdowns.append(normalized)

        group_names, group_values = _build_grouped_component_matrix(query_breakdowns)
        if not group_names:
            continue

        apply_paper_style()
        fig, ax = plt.subplots(figsize=PAPER_FIGSIZE)
        bottoms = np.zeros(len(case_labels), dtype=float)

        for group_idx, (group_name, vals) in enumerate(zip(group_names, group_values)):
            ax.bar(
                x,
                vals,
                bottom=bottoms,
                label=group_name,
                color=GROUP_COLORS.get(group_name, "#9D9D9D"),
                hatch=hatch_for_index(group_idx),
                edgecolor="black",
                linewidth=0.5,
            )
            bottoms += vals

        ax.set_xlabel(x_axis_label, fontsize=16)
        ax.set_ylabel("Time (ms)", fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(case_labels, rotation=20, ha="right")
        ax.tick_params(axis="both", labelsize=13)
        ax.grid(False)
        make_legend_bold(ax, loc="upper right", fontsize=13)
        fig.tight_layout()
        safe_query = sanitize_case_token(query)
        fig.savefig(figures_dir / f"breakdown_{safe_query}.png", dpi=180)
        fig.savefig(figures_dir / f"breakdown_{safe_query}.pdf")
        plt.close(fig)
