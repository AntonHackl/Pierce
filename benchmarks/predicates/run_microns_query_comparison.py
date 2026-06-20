#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path

# Add project root to sys.path to allow imports from 'benchmarks'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    PIERCE_DIR,
    canonical_microns_aggregated_paths,
    create_benchmark_run_layout,
    ensure_microns_splits,
    ensure_microns_aggregated_meshes,
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

REPO_ROOT = SCRIPT_DIR.parent.parent
PIERCE_DIR = REPO_ROOT / "pierce"
DEFAULT_MICRONS_SIZES = [4, 8]

def main():
    parser = argparse.ArgumentParser(description="MICrONS subset benchmark for mesh query comparison")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_MICRONS_SIZES,
                        help="MICrONS subset sizes in GB to benchmark")
    parser.add_argument("--source-root", type=str, 
                        default=str(REPO_ROOT / "scripts" / "microns_data"),
                        help="Root directory containing MICrONS GLB subset folders")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--grid-cell-size", type=float, default=700.0)
    
    add_query_selection_arguments(parser)

    parser.add_argument("--overlap-mode", type=str, default="direct_estimation", choices=["direct_estimation"])
    parser.add_argument("--intersection-mode", type=str, default="estimated", choices=["estimated", "estimate_only"])
    parser.add_argument("--overlap-query-direction", type=str, default="both", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"])
    parser.add_argument("--intersection-query-direction", type=str, default="both", choices=["both", "mesh1_to_mesh2", "mesh2_to_mesh1"])
    parser.add_argument("--overlap-max-iterations", type=float, default=100.0)
    parser.add_argument("--hash-load-factor", type=float, default=0.5)
    parser.add_argument("--enable-profiling-stats", action="store_true")
    parser.add_argument("--include-overlap-pairs", action="store_true")
    args = parser.parse_args()

    queries = resolve_queries(args.queries, args.approaches)

    # Use the same shared dataset root as overlap overall performance MICrONS runs.
    shared_dirs = get_shared_data_dirs("microns_overlap")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "query_comparison_microns")
    logs_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])

    splits_dir = shared_dirs["root"] / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root)

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
        enable_profiling_stats=args.enable_profiling_stats,
        intersection_query_direction=args.intersection_query_direction,
    )

    results = []
    case_labels = []
    
    for size_gb in args.sizes:
        print(f"\n--- Preparing MICrONS {size_gb}GB dataset ---")
        split_a, split_b = ensure_microns_splits(size_gb, source_root, splits_dir)
        
        agg_a, agg_b = canonical_microns_aggregated_paths(shared_dirs["raw"], size_gb)
        ensure_microns_aggregated_meshes(split_a, split_b, agg_a, agg_b)

        case_label = f"microns_{size_gb}gb"
        case_log_dir = logs_dir / sanitize_case_token(case_label)
        case_log_dir.mkdir(parents=True, exist_ok=True)

        # Preprocessing
        ensure_preprocessed(adapters, [agg_a, agg_b], log_dir=case_log_dir)

        row = {
            "size_gb": size_gb,
            "mesh1": str(agg_a),
            "mesh2": str(agg_b),
            "size_bytes1": agg_a.stat().st_size if agg_a.exists() else 0,
            "size_bytes2": agg_b.stat().st_size if agg_b.exists() else 0,
        }
        row.update(
            run_selected_queries(
                adapters=adapters,
                queries=queries,
                mesh1=agg_a,
                mesh2=agg_b,
                runs=args.runs,
                timeout=args.timeout,
                overlap_query_direction=args.overlap_query_direction,
                intersection_extra_args=intersection_extra_args,
                log_dir=case_log_dir,
            )
        )

        results.append(row)
        case_labels.append(case_label)
        print(f"size_gb={size_gb}: done")

    generate_query_comparison_figures(
        results_rows=results,
        queries=queries,
        case_labels=case_labels,
        figures_dir=figures_dir,
        title_prefix="MICrONS dataset",
        x_axis_label="Dataset case",
    )

    payload = {
        "metadata": {
            "scenario": "microns_query_comparison",
            "query_type": "predicate_comparison",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "sizes": args.sizes,
            "grid_cell_size": args.grid_cell_size,
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
