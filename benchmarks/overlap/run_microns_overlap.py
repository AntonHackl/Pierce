#!/usr/bin/env python3
import argparse
import sys
import subprocess
from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from benchmarks.common.viz_utils import generate_scalability_figure, generate_breakdown_figure
from benchmarks.overlap.adapters.pierce_adapter import PierceAdapter
from benchmarks.overlap.adapters.cgal_adapter import CGALAdapter
from benchmarks.overlap.adapters.touch_adapter import TOUCHAdapter

CGAL_DIR = REPO_ROOT / "baselines/face"


def main():
    parser = argparse.ArgumentParser(description="MICrONS subset benchmark for mesh overlap (Direct Estimation, Face, TOUCH)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 8],
                        help="MICrONS subset sizes in GB to benchmark")
    parser.add_argument("--source-root", type=str, 
                        default=str(REPO_ROOT / "scripts" / "microns_data"),
                        help="Root directory containing MICrONS GLB subset folders")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for CGAL/TOUCH")
    parser.add_argument("--grid-cell-size", type=float, default=700.0)
    parser.add_argument("--overlap-max-iterations", type=int, default=100)
    parser.add_argument("--track-hash-contention", action="store_true", help="Enable overlap hash contention tracking for Pierce direct estimation")
    parser.add_argument("--query-direction", type=str, default="both", choices=["both", "mesh1tomesh2", "mesh2tomesh1"])
    parser.add_argument("--approaches", type=str, nargs="+", default=["direct_estimation", "cgal", "touch"],
                        help="Approaches to run")
    args = parser.parse_args()

    dirs = get_shared_data_dirs("microns_overlap")
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_microns")
    run_log_dir = Path(run_layout["logs_dir"])
    print(f"CGAL/TOUCH threads: {args.threads if args.threads else 'OpenMP default'}")

    splits_dir = dirs["root"] / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    source_root = Path(args.source_root)

    # Initialize Adapters
    adapters = {}
    if "direct_estimation" in args.approaches:
        adapters["direct_estimation"] = PierceAdapter(
            str(PIERCE_DIR), mode="direct_estimation", preprocessed_dir=str(dirs["preprocessed"]),
            timings_dir=str(dirs["timings"]), grid_cell_size=args.grid_cell_size, warmup_runs=args.warmup_runs,
            track_hash_contention=args.track_hash_contention,
        )
    if "cgal" in args.approaches:
        adapters["cgal"] = CGALAdapter(str(CGAL_DIR), preprocessed_dir=str(dirs["preprocessed"]), threads=args.threads, grid_cell_size=args.grid_cell_size)
    if "touch" in args.approaches:
        adapters["touch"] = TOUCHAdapter(str(CGAL_DIR), preprocessed_dir=str(dirs["preprocessed"]), threads=args.threads, grid_cell_size=args.grid_cell_size)

    results = []
    for size_gb in args.sizes:
        print(f"\n--- Preparing MICrONS {size_gb}GB dataset ---")
        split_a, split_b = ensure_microns_splits(size_gb, source_root, splits_dir)
        
        agg_a, agg_b = canonical_microns_aggregated_paths(dirs["raw"], size_gb)
        ensure_microns_aggregated_meshes(split_a, split_b, agg_a, agg_b)

        # Preprocessing (All approaches share the Pierce .pre files)
        if any(a in args.approaches for a in ["direct_estimation", "cgal", "touch"]):
             # Use direct_estimation adapter for preprocessing if it exists, otherwise create a temporary one
             pre_adapter = adapters.get("direct_estimation")
             if not pre_adapter:
                 pre_adapter = PierceAdapter(
                     str(PIERCE_DIR), mode="direct_estimation", preprocessed_dir=str(dirs["preprocessed"]),
                     timings_dir=str(dirs["timings"]), grid_cell_size=args.grid_cell_size,
                 )
             for file_path in (agg_a, agg_b):
                 if not pre_adapter.check_preprocessed(str(file_path)):
                     pre_adapter.preprocess_from_source(str(file_path), str(file_path), log_dir=str(run_log_dir))

        entry = {
            "size_gb": size_gb,
            "size_bytes_a": agg_a.stat().st_size if agg_a.exists() else 0,
            "size_bytes_b": agg_b.stat().st_size if agg_b.exists() else 0,
        }
        result_size = 0

        # Run Benchmarks
        for approach_name in args.approaches:
            if approach_name not in adapters:
                continue
            
            print(f"Running {approach_name}...")
            adapter = adapters[approach_name]
            
            if approach_name == "direct_estimation":
                res = adapter.run_overlap(
                    str(agg_a), str(agg_b), args.runs, timeout=args.timeout, 
                    query_direction=args.query_direction,
                    overlap_max_iterations=args.overlap_max_iterations,
                    log_dir=str(run_log_dir)
                )
            else:
                # Face/TOUCH
                res = adapter.run_overlap(
                    str(agg_a), str(agg_b), args.runs, timeout=args.timeout,
                    log_dir=str(run_log_dir)
                )

            if approach_name == "direct_estimation" and "error" not in res:
                # Result size must be the true number of unique output pairs, not the estimate.
                result_size = int(res.get("num_intersections", 0))
            
            entry[approach_name] = res

        entry["result_size"] = result_size

        results.append(entry)
        print(f"size_gb={size_gb}: done")

    payload = {
        "metadata": {
            "scenario": "microns_overlap",
            "query_type": "overlap",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "sizes": args.sizes,
            "grid_cell_size": args.grid_cell_size,
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "timeout_seconds": args.timeout,
            "threads": args.threads,
            "overlap_max_iterations": args.overlap_max_iterations,
            "track_hash_contention": args.track_hash_contention,
            "query_direction": args.query_direction,
            "approaches": args.approaches,
            "shared_data_root": str(dirs["root"]),
        },
        "results": results,
    }

    out = run_layout["results_json"]
    write_json(out, payload)
    print(f"Saved: {out}")

    # Generate figures automatically
    figures_dir = Path(run_layout["figures_dir"])
    generate_scalability_figure(
        results=results,
        approaches=args.approaches,
        figures_dir=figures_dir,
        timestamp=run_layout["timestamp"],
        scenario_name="microns_overlap",
        x_axis_key="size_gb",
        x_axis_label="MICrONS subset size (GB)",
        y_axis_label="Overlap query time (ms) [log scale]",
        title="MICrONS Overlap Scalability"
    )
    
    if "direct_estimation" in args.approaches:
        generate_breakdown_figure(
            results=results,
            approaches=["direct_estimation"],
            figures_dir=figures_dir,
            timestamp=run_layout["timestamp"],
            scenario_name="microns_overlap",
            x_axis_key="size_gb",
            x_axis_label="Dataset Size",
            y_axis_label="Query time (ms)",
            title="Overlap Runtime Breakdown"
        )

if __name__ == "__main__":
    main()
