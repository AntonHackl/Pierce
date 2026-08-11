#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    canonical_cube_pair_paths,
    canonical_microns_aggregated_paths,
    canonical_nu_pair_paths,
    copy_to_latest_file,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    ensure_microns_aggregated_meshes,
    ensure_microns_splits,
    ensure_nu_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.predicates.core import (
    build_intersection_extra_args,
    build_pierce_query_adapters,
    ensure_preprocessed,
    run_selected_queries,
    sanitize_case_token,
)


LEGACY_OVERLAP_RAW_DIR = REPO_ROOT / "benchmarks" / "overlap" / "data" / "raw"
DEFAULT_QUERIES = ["overlap", "intersection"]
DEFAULT_DATASETS = [
    "cube_200k_vs_1m",
    "cube_translated_uo50_200k_vs_800k",
    "nuclei_vessel_nu800",
    "microns_8gb",
    "microns_16gb",
]
MICRONS_HASH_TABLE_OVERRIDES = {
    8: {
        "true_pairs": 636,
        "estimated_pairs": 5,
    },
    16: {
        # The 16 GB two-GPU run produced 3,481 slab-local pairs before global
        # deduplication.  Use 16,384 global slots, i.e. 8,192 per slab, so a
        # skewed slab remains well below saturation.
        "true_pairs": 8192,
        "estimated_pairs": 22,
    },
}


def _hash_table_slots_for_pairs(true_pairs: int) -> int:
    return int(true_pairs / 0.5)


def _hash_load_factor_for_target_slots(estimated_pairs: int, target_slots: int) -> float:
    if estimated_pairs <= 0:
        return 0.5
    return estimated_pairs / target_slots


def _error_row(label: str, exc: Exception, *, stage: str, gpu_count: int | None = None) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset": label,
        "status": "error",
        "stage": stage,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }
    if gpu_count is not None:
        row["gpu_count"] = gpu_count
        row["gpu_mode"] = "single" if gpu_count == 1 else "multi"
    return {
        **row,
    }


def _prepare_cube() -> tuple[str, Path, Path, float, dict[str, Any]]:
    dirs = get_shared_data_dirs("cube_scalability")
    mesh1, mesh2 = canonical_cube_pair_paths(
        dirs["raw"],
        num_cubes_a=200_000,
        num_cubes_b=1_000_000,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
        grid_cell_size=5.0,
    )
    ensure_cube_pair_dataset(
        mesh1,
        mesh2,
        num_cubes_a=200_000,
        num_cubes_b=1_000_000,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
    )
    meta = {
        "scenario": "cube_scalability",
        "description": "largest default cube pair",
        "num_cubes_a": 200_000,
        "num_cubes_b": 1_000_000,
        "shared_data_root": str(dirs["root"]),
    }
    return "cube_200k_vs_1m", mesh1, mesh2, 5.0, meta


def _prepare_translated_cube() -> tuple[str, Path, Path, float, dict[str, Any]]:
    dirs = get_shared_data_dirs("cube_translated_overlap")
    mesh1, mesh2 = canonical_cube_pair_paths(
        dirs["raw"],
        num_cubes_a=200_000,
        num_cubes_b=800_000,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
        grid_cell_size=5.0,
        universe_overlap_fraction=0.50,
        translation_axis="x",
    )
    ensure_cube_pair_dataset(
        mesh1,
        mesh2,
        num_cubes_a=200_000,
        num_cubes_b=800_000,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
        universe_overlap_fraction=0.50,
        translation_axis="x",
    )
    meta = {
        "scenario": "cube_translated_overlap",
        "description": "translated uniform cube pair with 50% universe overlap",
        "num_cubes_a": 200_000,
        "num_cubes_b": 800_000,
        "selectivity": 0.001,
        "universe_overlap_fraction": 0.50,
        "translation_axis": "x",
        "shared_data_root": str(dirs["root"]),
    }
    return "cube_translated_uo50_200k_vs_800k", mesh1, mesh2, 5.0, meta


def _prepare_nuclei_vessel() -> tuple[str, Path, Path, float, dict[str, Any]]:
    dirs = get_shared_data_dirs("large_nu_nn_scalability")
    nuclei, vessel = canonical_nu_pair_paths(
        dirs["raw"],
        nu=800,
        nv=750,
        prefix="tdbase_large",
    )
    ensure_nu_pair_dataset(nuclei, vessel, legacy_raw_dirs=[LEGACY_OVERLAP_RAW_DIR])
    meta = {
        "scenario": "large_nu_nn_scalability",
        "description": "largest default vessel-nuclei pair",
        "dataset_profile": "large_nu_v",
        "nu": 800,
        "nv": 750,
        "shared_data_root": str(dirs["root"]),
    }
    return "nuclei_vessel_nu800", vessel, nuclei, 200.0, meta


def _prepare_microns(
    size_gb: int,
    *,
    source_root: Path,
) -> tuple[str, Path, Path, float, dict[str, Any]]:
    dirs = get_shared_data_dirs("microns_overlap")
    splits_dir = dirs["root"] / "splits"
    split_a, split_b = ensure_microns_splits(size_gb, source_root, splits_dir)
    mesh1, mesh2 = canonical_microns_aggregated_paths(dirs["raw"], size_gb)
    ensure_microns_aggregated_meshes(split_a, split_b, mesh1, mesh2)
    meta = {
        "scenario": "microns_overlap",
        "description": f"MICRONS {size_gb} GB split pair",
        "size_gb": size_gb,
        "source_root": str(source_root),
        "shared_data_root": str(dirs["root"]),
    }
    if size_gb in MICRONS_HASH_TABLE_OVERRIDES:
        override = MICRONS_HASH_TABLE_OVERRIDES[size_gb]
        hash_table_slots = _hash_table_slots_for_pairs(override["true_pairs"])
        meta.update({
            "hash_table_override_reason": "observed true pairs at load factor 0.5",
            "hash_table_true_pairs": override["true_pairs"],
            "hash_table_estimated_pairs": override["estimated_pairs"],
            "overlap_hash_table_size": hash_table_slots,
            "intersection_hash_load_factor": _hash_load_factor_for_target_slots(
                override["estimated_pairs"],
                hash_table_slots,
            ),
        })
    return f"microns_{size_gb}gb", mesh1, mesh2, 700.0, meta


def _run_case(
    *,
    dataset_label: str,
    mesh1: Path,
    mesh2: Path,
    grid_cell_size: float,
    dataset_meta: dict[str, Any],
    gpu_count: int,
    run_layout: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    data_dirs = get_shared_data_dirs(dataset_meta["scenario"])
    adapters = build_pierce_query_adapters(
        repo_root=REPO_ROOT,
        data_dirs=data_dirs,
        grid_cell_size=grid_cell_size,
        warmup_runs=args.warmup_runs,
        overlap_mode="direct_estimation",
        intersection_mode=args.intersection_mode,
        include_overlap_pairs=False,
        track_gpu_memory=args.track_gpu_memory,
        overlap_max_iterations=args.overlap_max_iterations,
        num_gpus=gpu_count,
        overlap_hash_table_size=dataset_meta.get("overlap_hash_table_size"),
    )
    intersection_hash_load_factor = dataset_meta.get("intersection_hash_load_factor", args.hash_load_factor)
    intersection_extra_args = build_intersection_extra_args(
        overlap_max_iterations=args.overlap_max_iterations,
        hash_load_factor=intersection_hash_load_factor,
        enable_profiling_stats=args.enable_profiling_stats,
        track_overflow=args.track_overflow,
        track_gpu_memory=args.track_gpu_memory,
        intersection_query_direction=args.intersection_query_direction,
    )

    label = f"{dataset_label}_gpu{gpu_count}"
    log_dir = Path(run_layout["logs_dir"]) / sanitize_case_token(label)
    log_dir.mkdir(parents=True, exist_ok=True)

    ensure_preprocessed(adapters, [mesh1, mesh2], log_dir=log_dir)
    query_results = run_selected_queries(
        adapters=adapters,
        queries=args.queries,
        mesh1=mesh1,
        mesh2=mesh2,
        runs=args.runs,
        timeout=args.timeout,
        overlap_query_direction=args.overlap_query_direction,
        intersection_extra_args=intersection_extra_args,
        log_dir=log_dir,
    )

    return {
        "dataset": dataset_label,
        "status": "ok",
        "gpu_count": gpu_count,
        "gpu_mode": "single" if gpu_count == 1 else "multi",
        "grid_cell_size": grid_cell_size,
        "mesh1": str(mesh1),
        "mesh2": str(mesh2),
        "size_bytes1": mesh1.stat().st_size if mesh1.exists() else 0,
        "size_bytes2": mesh2.stat().st_size if mesh2.exists() else 0,
        "dataset_metadata": dataset_meta,
        "queries": query_results,
    }


def _write_summary_csv(path: Path, rows: list[dict[str, Any]], queries: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "gpu_count",
        "gpu_mode",
        "query",
        "status",
        "mean_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "num_gpus_active",
        "error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if row.get("status") != "ok":
                writer.writerow({
                    "dataset": row.get("dataset"),
                    "status": row.get("status"),
                    "error": row.get("error"),
                })
                continue
            for query in queries:
                result = row["queries"].get(query, {})
                writer.writerow({
                    "dataset": row["dataset"],
                    "gpu_count": row["gpu_count"],
                    "gpu_mode": row["gpu_mode"],
                    "query": query,
                    "status": "error" if "error" in result else "ok",
                    "mean_ms": result.get("mean"),
                    "min_ms": result.get("min"),
                    "max_ms": result.get("max"),
                    "std_ms": result.get("std"),
                    "num_gpus_active": result.get("num_gpus_active"),
                    "error": result.get("error"),
                })


def _write_runtime_plot(
    pdf_path: Path,
    png_path: Path,
    rows: list[dict[str, Any]],
    queries: list[str],
) -> None:
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    if not ok_rows:
        return

    datasets = sorted({row["dataset"] for row in ok_rows})
    speedups: dict[str, dict[str, float]] = {query: {} for query in queries}
    palette = {
        "overlap": "#2F6F73",
        "intersection": "#D18C2D",
    }

    for dataset in datasets:
        for query in queries:
            single_mean = None
            multi_mean = None
            for row in ok_rows:
                if row["dataset"] != dataset:
                    continue
                mean = row["queries"].get(query, {}).get("mean")
                if not isinstance(mean, (int, float)) or mean <= 0:
                    continue
                if row.get("gpu_count") == 1:
                    single_mean = float(mean)
                elif multi_mean is None or row.get("gpu_count", 0) > 1:
                    multi_mean = float(mean)
            if single_mean is not None and multi_mean is not None:
                speedups[query][dataset] = single_mean / multi_mean

    comparable_datasets = [
        dataset for dataset in datasets
        if any(dataset in query_speedups for query_speedups in speedups.values())
    ]
    if not comparable_datasets:
        runtime_bars = [
            (row["dataset"], query, row.get("gpu_count"), float(mean))
            for row in ok_rows
            for query in queries
            if isinstance((mean := row["queries"].get(query, {}).get("mean")), (int, float)) and mean > 0
        ]
        if not runtime_bars:
            return
        fig, ax = plt.subplots(figsize=(max(5.0, len(runtime_bars) * 1.4), 4.5))
        bars = ax.bar(
            range(len(runtime_bars)),
            [mean for _, _, _, mean in runtime_bars],
            color=[palette.get(query, "#5F6C7B") for _, query, _, _ in runtime_bars],
        )
        for bar, (_, _, _, mean) in zip(bars, runtime_bars):
            ax.text(bar.get_x() + bar.get_width() / 2.0, mean, f"{mean:.1f} ms", ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("Mean query runtime (ms)")
        ax.set_xticks(range(len(runtime_bars)))
        ax.set_xticklabels(
            [f"{dataset}\n{query}\n{gpu_count} GPU" for dataset, query, gpu_count, _ in runtime_bars],
        )
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(pdf_path)
        fig.savefig(png_path, dpi=300)
        plt.close(fig)
        return

    x_positions = list(range(len(comparable_datasets)))
    active_queries = [query for query in queries if speedups.get(query)]
    bar_width = min(0.35, 0.8 / max(1, len(active_queries)))
    fig, ax = plt.subplots(figsize=(max(8.0, len(comparable_datasets) * 2.4), 5.0))

    for query_index, query in enumerate(active_queries):
        offset = (query_index - (len(active_queries) - 1) / 2.0) * bar_width
        values = [speedups[query].get(dataset, 0.0) for dataset in comparable_datasets]
        bars = ax.bar(
            [x + offset for x in x_positions],
            values,
            width=bar_width,
            label=query,
            color=palette.get(query, "#5F6C7B"),
        )
        for bar, value in zip(bars, values):
            if value > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    f"{value:.2f}x",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--", alpha=0.7)
    ax.set_ylabel("Multi-GPU speedup over single GPU (x)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(comparable_datasets, rotation=20, ha="right")
    ax.legend(title="Query")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _write_query_runtime_plot(
    pdf_path: Path,
    png_path: Path,
    rows: list[dict[str, Any]],
    queries: list[str],
) -> None:
    """Plot the measured query times for every available dataset/query/GPU variant."""
    palette = {
        "overlap": "#2F6F73",
        "intersection": "#D18C2D",
    }
    dataset_order = list(dict.fromkeys(
        row["dataset"] for row in rows if row.get("status") == "ok"
    ))
    runtimes = {
        (row["dataset"], query, int(row["gpu_count"])): float(mean)
        for row in rows
        if row.get("status") == "ok"
        for query in queries
        if isinstance((mean := row.get("queries", {}).get(query, {}).get("mean")), (int, float)) and mean > 0
    }
    if not runtimes:
        return

    gpu_counts = sorted({gpu_count for _, _, gpu_count in runtimes})
    bar_width = min(0.20, 0.8 / max(1, len(queries) * len(gpu_counts)))
    x_positions = list(range(len(dataset_order)))
    total_bars = len(queries) * len(gpu_counts)
    fig, ax = plt.subplots(figsize=(max(8.0, len(dataset_order) * 2.4), 5.5))
    for query_index, query in enumerate(queries):
        for gpu_index, gpu_count in enumerate(gpu_counts):
            slot = query_index * len(gpu_counts) + gpu_index
            offset = (slot - (total_bars - 1) / 2.0) * bar_width
            values = [runtimes.get((dataset, query, gpu_count)) for dataset in dataset_order]
            for x, value in zip(x_positions, values):
                if value is None:
                    continue
                bar = ax.bar(
                    x + offset,
                    value,
                    width=bar_width,
                    color=palette.get(query, "#5F6C7B"),
                    hatch="//" if gpu_count > 1 else None,
                    edgecolor="#333333",
                    linewidth=0.5,
                )[0]
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    value,
                    f"{value:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    rotation=90,
                )

    ax.set_ylabel("Mean query runtime (ms)")
    ax.set_xlabel("Dataset")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(dataset_order, rotation=20, ha="right")
    ax.legend(
        handles=[
            Patch(facecolor=palette["overlap"], edgecolor="#333333", label="Overlap"),
            Patch(facecolor=palette["intersection"], edgecolor="#333333", label="Intersection"),
            Patch(facecolor="white", edgecolor="#333333", label="1 GPU"),
            Patch(facecolor="white", edgecolor="#333333", hatch="//", label="Multi-GPU"),
        ],
        ncols=2,
    )
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _write_microns_16gb_runtime_plot(
    pdf_path: Path,
    png_path: Path,
    rows: list[dict[str, Any]],
    queries: list[str],
) -> None:
    microns_rows = [
        row for row in rows
        if row.get("status") == "ok" and row.get("dataset") == "microns_16gb"
    ]
    if not microns_rows:
        return

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    palette = {
        "overlap": "#2F6F73",
        "intersection": "#D18C2D",
    }
    for row in microns_rows:
        for query in queries:
            mean = row["queries"].get(query, {}).get("mean")
            if isinstance(mean, (int, float)):
                labels.append(f"{query}\n{row['gpu_count']} GPU")
                values.append(float(mean))
                colors.append(palette.get(query, "#5F6C7B"))

    if not values:
        return

    fig, ax = plt.subplots(figsize=(max(5.0, len(values) * 1.4), 4.5))
    bars = ax.bar(range(len(values)), values, color=colors)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            value,
            f"{value:.1f} ms",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax.set_ylabel("Mean query runtime (ms)")
    ax.set_xlabel("MICRONS 16 GB query")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)


def _write_outputs(
    *,
    run_layout: dict[str, Any],
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    summary_csv = Path(run_layout["run_dir"]) / "summary.csv"
    figure_pdf = Path(run_layout["figures_dir"]) / "runtime_by_dataset_gpu.pdf"
    figure_png = Path(run_layout["figures_dir"]) / "runtime_by_dataset_gpu.png"
    query_runtime_pdf = Path(run_layout["figures_dir"]) / "query_runtime_by_dataset_gpu.pdf"
    query_runtime_png = Path(run_layout["figures_dir"]) / "query_runtime_by_dataset_gpu.png"
    microns_16gb_figure_pdf = Path(run_layout["figures_dir"]) / "microns_16gb_runtime.pdf"
    microns_16gb_figure_png = Path(run_layout["figures_dir"]) / "microns_16gb_runtime.png"
    _write_summary_csv(summary_csv, rows, args.queries)
    _write_runtime_plot(figure_pdf, figure_png, rows, args.queries)
    _write_query_runtime_plot(query_runtime_pdf, query_runtime_png, rows, args.queries)
    _write_microns_16gb_runtime_plot(microns_16gb_figure_pdf, microns_16gb_figure_png, rows, args.queries)

    payload = {
        "metadata": {
            "scenario": "single_vs_multi_gpu",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "multi_gpus": args.multi_gpus,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "datasets": args.datasets,
            "queries": args.queries,
            "source_root": str(args.source_root),
            "summary_csv": str(summary_csv),
            "runtime_figure": str(figure_pdf),
            "runtime_figures": {
                "pdf": str(figure_pdf),
                "png": str(figure_png),
            },
            "query_runtime_figures": {
                "pdf": str(query_runtime_pdf),
                "png": str(query_runtime_png),
            },
            "microns_16gb_runtime_figures": {
                "pdf": str(microns_16gb_figure_pdf),
                "png": str(microns_16gb_figure_png),
            },
        },
        "results": rows,
    }
    results_path = Path(run_layout["results_json"])
    write_json(results_path, payload)
    copy_to_latest_file(results_path, SCRIPT_DIR / "single_vs_multi_gpu_latest.json")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare Pierce single-GPU and multi-GPU runtime on large predicate benchmark datasets."
    )
    parser.add_argument("--multi-gpus", type=int, default=2, help="GPU count for the multi-GPU runs.")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--source-root", type=Path, default=REPO_ROOT / "scripts" / "microns_data")
    parser.add_argument("--datasets", nargs="+", choices=DEFAULT_DATASETS, default=DEFAULT_DATASETS)
    parser.add_argument(
        "--gpu-counts",
        type=int,
        nargs="+",
        default=None,
        help="Run only these GPU-count variants (default: each dataset's standard single/multi-GPU variants).",
    )
    parser.add_argument("--queries", nargs="+", choices=DEFAULT_QUERIES, default=DEFAULT_QUERIES)
    parser.add_argument("--overlap-query-direction", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"], default="both")
    parser.add_argument("--intersection-query-direction", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"], default="both")
    parser.add_argument("--intersection-mode", choices=["estimated", "estimate_only"], default="estimated")
    parser.add_argument("--overlap-max-iterations", type=int, default=100)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--track-overflow", action="store_true")
    parser.add_argument("--enable-profiling-stats", action="store_true")
    parser.add_argument("--track-gpu-memory", action="store_true")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first failed dataset or run.")
    args = parser.parse_args()

    if args.multi_gpus <= 1:
        parser.error("--multi-gpus must be greater than 1")
    if args.gpu_counts is not None and any(gpu_count <= 0 for gpu_count in args.gpu_counts):
        parser.error("--gpu-counts values must be positive")

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "single_vs_multi_gpu")
    rows: list[dict[str, Any]] = []

    dataset_specs = [
        ("cube_200k_vs_1m", lambda: _prepare_cube(), [1, args.multi_gpus]),
        ("cube_translated_uo50_200k_vs_800k", lambda: _prepare_translated_cube(), [1, args.multi_gpus]),
        ("nuclei_vessel_nu800", lambda: _prepare_nuclei_vessel(), [1, args.multi_gpus]),
        ("microns_8gb", lambda: _prepare_microns(8, source_root=args.source_root), [1, args.multi_gpus]),
        ("microns_16gb", lambda: _prepare_microns(16, source_root=args.source_root), [args.multi_gpus]),
    ]
    selected_datasets = set(args.datasets)
    dataset_specs = [
        spec for spec in dataset_specs
        if spec[0] in selected_datasets
    ]
    if args.gpu_counts is not None:
        dataset_specs = [
            (label, prepare, [gpu_count for gpu_count in gpu_counts if gpu_count in args.gpu_counts])
            for label, prepare, gpu_counts in dataset_specs
        ]

    for expected_label, prepare, gpu_counts in dataset_specs:
        try:
            dataset_label, mesh1, mesh2, grid_cell_size, dataset_meta = prepare()
        except Exception as exc:
            if args.fail_fast:
                raise
            rows.append(_error_row(expected_label, exc, stage="prepare"))
            _write_outputs(run_layout=run_layout, rows=rows, args=args)
            continue

        for gpu_count in gpu_counts:
            print(f"\n--- {dataset_label}: {gpu_count} GPU(s) ---")
            try:
                rows.append(
                    _run_case(
                        dataset_label=dataset_label,
                        mesh1=mesh1,
                        mesh2=mesh2,
                        grid_cell_size=grid_cell_size,
                        dataset_meta=dataset_meta,
                        gpu_count=gpu_count,
                        run_layout=run_layout,
                        args=args,
                    )
                )
            except Exception as exc:
                if args.fail_fast:
                    raise
                rows.append(_error_row(dataset_label, exc, stage="run", gpu_count=gpu_count))
            _write_outputs(run_layout=run_layout, rows=rows, args=args)

    _write_outputs(run_layout=run_layout, rows=rows, args=args)
    summary_csv = Path(run_layout["run_dir"]) / "summary.csv"
    figure_pdf = Path(run_layout["figures_dir"]) / "runtime_by_dataset_gpu.pdf"
    figure_png = Path(run_layout["figures_dir"]) / "runtime_by_dataset_gpu.png"
    query_runtime_pdf = Path(run_layout["figures_dir"]) / "query_runtime_by_dataset_gpu.pdf"
    query_runtime_png = Path(run_layout["figures_dir"]) / "query_runtime_by_dataset_gpu.png"
    microns_16gb_figure_pdf = Path(run_layout["figures_dir"]) / "microns_16gb_runtime.pdf"
    microns_16gb_figure_png = Path(run_layout["figures_dir"]) / "microns_16gb_runtime.png"
    results_path = Path(run_layout["results_json"])
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_csv}")
    if figure_pdf.exists():
        print(f"Saved figure PDF: {figure_pdf}")
    if figure_png.exists():
        print(f"Saved figure PNG: {figure_png}")
    if query_runtime_pdf.exists():
        print(f"Saved query-runtime figure PDF: {query_runtime_pdf}")
    if query_runtime_png.exists():
        print(f"Saved query-runtime figure PNG: {query_runtime_png}")
    if microns_16gb_figure_pdf.exists():
        print(f"Saved MICRONS 16 GB figure PDF: {microns_16gb_figure_pdf}")
    if microns_16gb_figure_png.exists():
        print(f"Saved MICRONS 16 GB figure PNG: {microns_16gb_figure_png}")


if __name__ == "__main__":
    main()
