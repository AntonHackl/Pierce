#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    CUBE_SCALABILITY_COUNTS,
    canonical_cube_pair_paths,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.predicates.core import (
    add_query_selection_arguments,
    build_intersection_extra_args,
    build_pierce_query_adapters,
    ensure_preprocessed,
    generate_query_comparison_figures,
    resolve_queries,
    sanitize_case_token,
    run_selected_queries,
)


DEFAULT_CUBE_COUNTS = CUBE_SCALABILITY_COUNTS
FIXED_CUBE_COUNT_A = 200000
DEFAULT_GRID_CELL_SIZE = 5.0
DEFAULT_MIN_SIZE = 1.0
DEFAULT_MAX_SIZE = 2.0
DEFAULT_SELECTIVITY = 0.001
DEFAULT_SEED = 42
SHARED_SCENARIO = "cube_scalability"


def main():
    parser = argparse.ArgumentParser(description="Cube scalability benchmark for mesh query comparison")
    parser.add_argument(
        "--cube-counts",
        type=int,
        nargs="+",
        default=DEFAULT_CUBE_COUNTS,
        help="Cube counts for dataset B to benchmark. Defaults to the overlap cube scalability cases.",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--grid-cell-size", type=float, default=DEFAULT_GRID_CELL_SIZE)

    add_query_selection_arguments(parser)

    parser.add_argument("--overlap-mode", type=str, default="direct_estimation", choices=["direct_estimation"])
    parser.add_argument("--intersection-mode", type=str, default="estimated", choices=["estimated", "estimate_only"])
    parser.add_argument("--overlap-query-direction", type=str, default="both", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"])
    parser.add_argument("--intersection-query-direction", type=str, default="both", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"])
    parser.add_argument("--overlap-max-iterations", type=float, default=100.0)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--track-overflow", action="store_true")
    parser.add_argument("--enable-profiling-stats", action="store_true")
    parser.add_argument("--include-overlap-pairs", action="store_true")
    args = parser.parse_args()

    queries = resolve_queries(args.queries, args.approaches)

    shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "query_comparison_cube_scalability")
    logs_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])

    adapters = build_pierce_query_adapters(
        repo_root=REPO_ROOT,
        data_dirs=shared_dirs,
        grid_cell_size=args.grid_cell_size,
        warmup_runs=args.warmup_runs,
        overlap_mode=args.overlap_mode,
        intersection_mode=args.intersection_mode,
        include_overlap_pairs=args.include_overlap_pairs,
        overlap_max_iterations=int(args.overlap_max_iterations),
    )

    intersection_extra_args = build_intersection_extra_args(
        overlap_max_iterations=args.overlap_max_iterations,
        hash_load_factor=args.hash_load_factor,
        track_overflow=args.track_overflow,
        enable_profiling_stats=args.enable_profiling_stats,
        intersection_query_direction=args.intersection_query_direction,
    )

    results = []
    case_labels = []

    for cube_count_b in args.cube_counts:
        mesh1, mesh2 = canonical_cube_pair_paths(
            shared_dirs["raw"],
            num_cubes_a=FIXED_CUBE_COUNT_A,
            num_cubes_b=cube_count_b,
            min_size=DEFAULT_MIN_SIZE,
            max_size=DEFAULT_MAX_SIZE,
            selectivity=DEFAULT_SELECTIVITY,
            seed=DEFAULT_SEED,
            grid_cell_size=args.grid_cell_size,
        )
        ensure_cube_pair_dataset(
            mesh1,
            mesh2,
            num_cubes_a=FIXED_CUBE_COUNT_A,
            num_cubes_b=cube_count_b,
            min_size=DEFAULT_MIN_SIZE,
            max_size=DEFAULT_MAX_SIZE,
            selectivity=DEFAULT_SELECTIVITY,
            seed=DEFAULT_SEED,
        )

        case_label = f"cube_nb_{cube_count_b}"
        case_log_dir = logs_dir / sanitize_case_token(case_label)
        case_log_dir.mkdir(parents=True, exist_ok=True)

        ensure_preprocessed(adapters, [mesh1, mesh2], log_dir=case_log_dir)

        row = {
            "num_cubes_a": FIXED_CUBE_COUNT_A,
            "num_cubes_b": cube_count_b,
            "mesh1": str(mesh1),
            "mesh2": str(mesh2),
            "size_bytes1": mesh1.stat().st_size if mesh1.exists() else 0,
            "size_bytes2": mesh2.stat().st_size if mesh2.exists() else 0,
        }
        row.update(
            run_selected_queries(
                adapters=adapters,
                queries=queries,
                mesh1=mesh1,
                mesh2=mesh2,
                runs=args.runs,
                timeout=args.timeout,
                overlap_query_direction=args.overlap_query_direction,
                intersection_extra_args=intersection_extra_args,
                log_dir=case_log_dir,
            )
        )

        results.append(row)
        case_labels.append(f"200k vs {cube_count_b // 1000}k")
        print(f"cube_count_b={cube_count_b}: done")

    generate_query_comparison_figures(
        results_rows=results,
        queries=queries,
        case_labels=case_labels,
        figures_dir=figures_dir,
        title_prefix="Cube scalability",
        x_axis_label="Dataset case",
    )

    payload = {
        "metadata": {
            "scenario": "cube_scalability_query_comparison",
            "query_type": "predicate_comparison",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "cube_counts": args.cube_counts,
            "fixed_cube_count_a": FIXED_CUBE_COUNT_A,
            "grid_cell_size": args.grid_cell_size,
            "cube_size_range": [DEFAULT_MIN_SIZE, DEFAULT_MAX_SIZE],
            "selectivity": DEFAULT_SELECTIVITY,
            "seed": DEFAULT_SEED,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "queries": queries,
            "query_implementations": {
                "overlap": f"pierce_{args.overlap_mode}",
                "intersection": f"pierce_{args.intersection_mode}",
                "containment": "pierce_containment",
            },
            "intersection_query_direction": args.intersection_query_direction,
            "overlap_query_direction": args.overlap_query_direction,
            "overlap_max_iterations": args.overlap_max_iterations,
            "hash_load_factor": args.hash_load_factor,
            "track_overflow": args.track_overflow,
            "enable_profiling_stats": args.enable_profiling_stats,
            "include_overlap_pairs": args.include_overlap_pairs,
            "shared_data_root": str(shared_dirs["root"]),
        },
        "results": results,
    }

    out = Path(run_layout["results_json"])
    write_json(out, payload)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
