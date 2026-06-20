
import os
import sys
import json
import argparse
from pathlib import Path
import subprocess
# Add project root to sys.path to allow imports from 'benchmarks'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.scenario_utils import (
    build_selectivity_sweep,
    canonical_cube_pair_paths,
    count_triangles,
    compute_universe_for_selectivity,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
)

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.pierce_adapter import PierceAdapter
from adapters.tdbase_adapter import TDBaseAdapter
from adapters.cgal_adapter import CGALAdapter
from adapters.touch_adapter import TOUCHAdapter
from visualize_selectivity_test import visualize_selectivity
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODES,
)

# Configuration
SELECTIVITIES = build_selectivity_sweep(scale="linear")
NUM_CUBES = 50000
MIN_SIZE = 1
MAX_SIZE = 4
GRID_CELL_SIZE = 5
TIMEOUT_SECONDS = 120.0  # 2 minutes timeout per run
ESTIMATED_MEM10_FIXED_BYTES = 10 * 1024 * 1024 * 1024  # 10 GiB fixed hash table budget

WORKSPACE_ROOT = Path(__file__).resolve().parent.parent.parent
PIERCE_DIR = WORKSPACE_ROOT / "pierce"
TDBASE_DIR = WORKSPACE_ROOT / "baselines/tdbase_extensions"
CGAL_DIR = WORKSPACE_ROOT / "baselines/face"
DATA_DIR = Path(__file__).parent / "data"
RAW_DIR = DATA_DIR / "raw" / "selectivity_test"
PREPROCESSED_DIR = DATA_DIR / "preprocessed" / "selectivity_test"
TIMINGS_DIR = DATA_DIR / "timings" / "selectivity_test"
RESULTS_DIR = Path(__file__).parent / "results" / "selectivity_test"

GENERATOR_SCRIPT = PIERCE_DIR / "scripts/generate_cubes_by_selectivity.py"
SHARED_SCENARIO = "selectivity"

def main():
    parser = argparse.ArgumentParser(description="Selectivity Benchmark for Mesh Overlap")
    parser.add_argument("--approaches", type=str, nargs="+", 
                        default=["exact", "estimated", "direct_estimation", "estimated_mem10", "tdbase", "cgal", "touch"],
                        choices=["exact", "estimated", "direct_estimation", "estimated_mem10", "tdbase", "cgal", "touch"],
                        help="Approaches to run")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per selectivity")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds per run")
    parser.add_argument(
        "--tdbase-timing-mode",
        type=str,
        default=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
        choices=TDBASE_TIMING_MODES,
        help="TDBase query-time definition. Default uses index+compute+evaluate; use compute_only to revert.",
    )
    args = parser.parse_args()

    run_layout = create_benchmark_run_layout(Path(__file__).parent, "overlap_selectivity")

    shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
    PREPROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TIMINGS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    summary_results = []

    for selectivity in SELECTIVITIES:
        print(f"\n{'='*60}")
        print(f"Processing Selectivity: {selectivity}")
        print(f"{'='*60}")

        # 1. Compute Universe and Grid Resolution
        universe_extent = compute_universe_for_selectivity(selectivity, MIN_SIZE, MAX_SIZE)
        grid_cell_size = int(round(universe_extent / GRID_CELL_SIZE))
        if grid_cell_size < 1: grid_cell_size = 1
        
        print(f"Universe Extent: {universe_extent:.2f}")
        print(f"Grid Resolution: {grid_cell_size} (Cell Size: {universe_extent/grid_cell_size:.2f})")

        # 2. Generate Data
        obj_a, obj_b = canonical_cube_pair_paths(
            shared_dirs["raw"],
            num_cubes_a=NUM_CUBES,
            num_cubes_b=NUM_CUBES,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            selectivity=selectivity,
            seed=42,
            grid_cell_size=grid_cell_size,
        )
        
        # .dt paths for consistent naming in adapter
        dt_a = obj_a.with_suffix('.dt') 
        dt_b = obj_b.with_suffix('.dt')
        ensure_cube_pair_dataset(
            obj_a,
            obj_b,
            num_cubes_a=NUM_CUBES,
            num_cubes_b=NUM_CUBES,
            min_size=MIN_SIZE,
            max_size=MAX_SIZE,
            selectivity=selectivity,
            seed=42,
        )

        # 3. Setup Adapter
        adapter = PierceAdapter(
            str(PIERCE_DIR), 
            mode="exact", 
            preprocessed_dir=str(shared_dirs["preprocessed"]),
            timings_dir=str(shared_dirs["timings"]),
            grid_cell_size=grid_cell_size,
            warmup_runs=2
        )

        # 4. Preprocess
        if any(m in args.approaches for m in ["exact", "estimated", "direct_estimation", "estimated_mem10", "cgal", "touch"]):
            # PierceAdapter.check_preprocessed() checks if .pre exists based on input filename
            # We assume if the file exists it was generated with the correct parameters (grid size).
            # To be safe, we could delete it, but for now we trust the flow or use distinct filenames if parameters changed repeatedly (they don't here).
            print("Ensuring preprocessed files (Pierce)...")
            adapter.preprocess_from_source(str(obj_a), str(dt_a), log_dir=str(run_layout["logs_dir"]))
            adapter.preprocess_from_source(str(obj_b), str(dt_b), log_dir=str(run_layout["logs_dir"]))

        # Setup TDBase & Preprocess
        if "tdbase" in args.approaches:
            tdbase_adapter = TDBaseAdapter(
                str(TDBASE_DIR),
                preprocessed_dir=str(shared_dirs["preprocessed"]),
                query_timing_mode=args.tdbase_timing_mode,
            )
            print("Ensuring preprocessed files (TDBase)...")
            tdbase_adapter.preprocess_from_source(str(obj_a), str(dt_a), log_dir=str(run_layout["logs_dir"]))
            tdbase_adapter.preprocess_from_source(str(obj_b), str(dt_b), log_dir=str(run_layout["logs_dir"]))

        # Setup CGAL & TOUCH
        if "cgal" in args.approaches:
            cgal_adapter = CGALAdapter(
                str(CGAL_DIR),
                preprocessed_dir=str(shared_dirs["preprocessed"]),
                grid_cell_size=grid_cell_size
            )
        if "touch" in args.approaches:
            touch_adapter = TOUCHAdapter(
                str(CGAL_DIR),
                preprocessed_dir=str(shared_dirs["preprocessed"]),
                grid_cell_size=grid_cell_size
            )
        # CGAL and TOUCH use the same preprocessed files as Pierce (.pre), which are already generated.

        # 5. Run Benchmark
        print("Running benchmark...")
        
        # Test requested modes
        modes = args.approaches
        res_per_sel = {
            "selectivity": selectivity, 
            "grid_cell_size": grid_cell_size, 
            "universe": universe_extent,
            "num_cubes": NUM_CUBES,
            "num_obj1": NUM_CUBES,
            "num_obj2": NUM_CUBES,
            "size_bytes1": obj_a.stat().st_size if obj_a.exists() else 0,
            "size_bytes2": obj_b.stat().st_size if obj_b.exists() else 0,
            "num_triangles1": count_triangles(obj_a) if obj_a.exists() else 0,
            "num_triangles2": count_triangles(obj_b) if obj_b.exists() else 0,
            "universe_extents1": [0.0, 0.0, 0.0],
            "universe_extents2": [0.0, 0.0, 0.0],
        }
        
        for mode in modes:
            current_adapter = None
            if mode == "tdbase":
                current_adapter = tdbase_adapter
                # Update name for consistent result key
                current_adapter.name = "tdbase"
            elif mode == "cgal":
                current_adapter = cgal_adapter
                current_adapter.name = "cgal"
            elif mode == "touch":
                current_adapter = touch_adapter
                current_adapter.name = "touch"
            else:
                adapter.hash_table_size = None
                adapter.hash_table_free_mem_fraction = None
                adapter.mode = mode
                # Update executable manually as correct binary depends on mode
                if mode == "exact":
                    adapter.executable = adapter.pierce_dir / "query/build/bin/pierce_overlap_two_pass"
                    adapter.name = "exact" # For result key consistency in json
                elif mode == "estimated":
                    adapter.mode = "direct_estimation"
                    adapter.executable = adapter.pierce_dir / "query/build/bin/pierce_overlap"
                    adapter.name = "estimated"
                elif mode == "direct_estimation":
                    adapter.executable = adapter.pierce_dir / "query/build/bin/pierce_overlap"
                    adapter.name = "direct_estimation"
                elif mode == "estimated_mem10":
                    adapter.mode = "direct_estimation"
                    adapter.executable = adapter.pierce_dir / "query/build/bin/pierce_overlap"
                    adapter.name = "estimated_mem10"
                    hash_table_slots = ESTIMATED_MEM10_FIXED_BYTES // 8  # unsigned long long slots
                    if hash_table_slots < 1024:
                        hash_table_slots = 1024
                    if (hash_table_slots % 2) == 0:
                        hash_table_slots += 1
                    adapter.hash_table_size = int(hash_table_slots)
                current_adapter = adapter
                
            results = current_adapter.run_overlap(
                str(obj_a),
                str(obj_b),
                num_runs=args.runs,
                timeout=args.timeout,
                log_dir=str(run_layout["logs_dir"])
            )
            
            # Use mode name as key in results
            result_key = mode
            
            if "error" in results:
                print(f"[{result_key}] Error: {results['error']}")
                res_per_sel[result_key] = {"error": results['error']}
            else:
                breakdown = results.get("breakdown", {})
                mean_time = results['mean']
                print(f"[{result_key}] Mean Time: {mean_time:.4f} ms")
                
                # Check for estimated mode to include download time in mean?
                # The mean reported by run_overlap covers the total time measured by Python or the tool.
                # In PierceAdapter, it sums up phases.
                # The breakdown should already be correct.
                
                res_per_sel[result_key] = {
                    "mean_ms": mean_time,
                    "std_ms": results['std'],
                    "intersections": results.get("num_intersections", 0),
                    "breakdown": breakdown
                }
                if "num_obj1" in results:
                    res_per_sel["num_obj1"] = int(results.get("num_obj1", res_per_sel["num_obj1"]))
                if "num_obj2" in results:
                    res_per_sel["num_obj2"] = int(results.get("num_obj2", res_per_sel["num_obj2"]))
                memory_stats = {
                    "hash_table_allocated_bytes": results.get("hash_table_allocated_bytes", 0),
                    "result_buffer_allocated_bytes": results.get("result_buffer_allocated_bytes", 0),
                    "result_buffer_used_bytes": results.get("result_buffer_used_bytes", 0),
                }
                if any(v and v > 0 for v in memory_stats.values()):
                    memory_stats["total_allocated_bytes"] = (
                        memory_stats["hash_table_allocated_bytes"] +
                        memory_stats["result_buffer_allocated_bytes"]
                    )
                    res_per_sel[result_key]["memory"] = memory_stats
                if "universe_extents1" in results:
                    res_per_sel["universe_extents1"] = results["universe_extents1"]
                if "universe_extents2" in results:
                    res_per_sel["universe_extents2"] = results["universe_extents2"]

        summary_results.append(res_per_sel)

    # Keep legacy summary output for compatibility.
    summary_path = RESULTS_DIR / "summary.json"
    write_json(summary_path, summary_results)
    print(f"\nSummary saved to {summary_path}")

    # Canonical per-run artifact.
    run_results_path = Path(run_layout["results_json"])
    write_json(
        run_results_path,
        {
            "metadata": {
                "timestamp": run_layout["timestamp"],
                "run_name": run_layout["run_name"],
                "run_dir": str(run_layout["run_dir"]),
                "runs": args.runs,
                "approaches": args.approaches,
                "selectivities": SELECTIVITIES,
                "num_cubes": NUM_CUBES,
            },
            "results": summary_results,
        },
    )
    print(f"Run results saved to {run_results_path}")

    # 6. Generate Visualization
    print("\nGenerating visualization...")
    try:
        visualize_selectivity(str(run_results_path))
    except Exception as e:
        print(f"Failed to generate visualization: {e}")

if __name__ == "__main__":
    main()
