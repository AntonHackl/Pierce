#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

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
    )
    intersection_extra_args = build_intersection_extra_args(
        overlap_max_iterations=args.overlap_max_iterations,
        hash_load_factor=args.hash_load_factor,
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

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    palette = {
        "overlap": "#2F6F73",
        "intersection": "#D18C2D",
    }

    for row in ok_rows:
        for query in queries:
            result = row["queries"].get(query, {})
            mean = result.get("mean")
            if isinstance(mean, (int, float)):
                labels.append(f"{row['dataset']}\n{query}\n{row['gpu_count']} GPU")
                values.append(float(mean))
                colors.append(palette.get(query, "#5F6C7B"))

    if not values:
        return

    width = max(10.0, len(values) * 0.85)
    fig, ax = plt.subplots(figsize=(width, 5.0))
    ax.bar(range(len(values)), values, color=colors)
    ax.set_ylabel("Mean query runtime (ms)")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
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
    _write_summary_csv(summary_csv, rows, args.queries)
    _write_runtime_plot(figure_pdf, figure_png, rows, args.queries)

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
            "queries": args.queries,
            "source_root": str(args.source_root),
            "summary_csv": str(summary_csv),
            "runtime_figure": str(figure_pdf),
            "runtime_figures": {
                "pdf": str(figure_pdf),
                "png": str(figure_png),
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

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "single_vs_multi_gpu")
    rows: list[dict[str, Any]] = []

    dataset_specs = [
        ("cube_200k_vs_1m", lambda: _prepare_cube(), [1, args.multi_gpus]),
        ("nuclei_vessel_nu800", lambda: _prepare_nuclei_vessel(), [1, args.multi_gpus]),
        ("microns_8gb", lambda: _prepare_microns(8, source_root=args.source_root), [1, args.multi_gpus]),
        ("microns_16gb", lambda: _prepare_microns(16, source_root=args.source_root), [args.multi_gpus]),
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
    results_path = Path(run_layout["results_json"])
    print(f"Saved results: {results_path}")
    print(f"Saved summary: {summary_csv}")
    if figure_pdf.exists():
        print(f"Saved figure PDF: {figure_pdf}")
    if figure_png.exists():
        print(f"Saved figure PNG: {figure_png}")


if __name__ == "__main__":
    main()
