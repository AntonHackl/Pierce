#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess 
import json
# Add project root to sys.path to allow imports from 'benchmarks'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import apply_paper_style, make_legend_bold, PAPER_FIGSIZE, set_log_timing_axis_limits, style_for
from benchmarks.common.scenario_utils import (
    canonical_cube_pair_paths,
    ensure_cube_pair_dataset,
    get_shared_data_dirs,
)

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.pierce_adapter import PierceAdapter
from adapters.cgal_adapter import CGALAdapter
from adapters.touch_adapter import TOUCHAdapter

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PIERCE_DIR = REPO_ROOT / "pierce"
CGAL_DIR = REPO_ROOT / "baselines/face"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0

# Cube Counts for Dataset B (Dataset A is fixed at 200k)
CUBE_COUNTS = [200000, 400000, 600000, 1000000]
FIXED_COUNT = "200k_a"
SHARED_SCENARIO = "cube_scalability"

def run_experiment(runs, grid_cell_size, run_log_dir, threads=None, approaches=None, timeout=120.0):
    if approaches is None:
        approaches = ["exact", "estimated", "cgal", "touch"]
    print("--- Starting Cube Scalability Experiment ---")
    print(f"CGAL/TOUCH threads: {threads if threads else 'OpenMP default'}")
    
    shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
    shared_raw_dir = shared_dirs["raw"]

    print(f"Logging runs to: {run_log_dir}")

    # Initialize Adapters
    print("Initializing adapters...")
    exact_adapter = PierceAdapter(
        str(PIERCE_DIR), 
        mode="exact", 
        preprocessed_dir=str(shared_dirs["preprocessed"]), 
        timings_dir=str(shared_dirs["timings"]),
        grid_cell_size=grid_cell_size,
        warmup_runs=1
    )
    
    estimated_adapter = PierceAdapter(
        str(PIERCE_DIR), 
        mode="estimated", 
        preprocessed_dir=str(shared_dirs["preprocessed"]), 
        timings_dir=str(shared_dirs["timings"]),
        grid_cell_size=grid_cell_size,
        warmup_runs=1
    )

    cgal_adapter = CGALAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        threads=threads,
        grid_cell_size=grid_cell_size
    )
    
    touch_adapter = TOUCHAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(shared_dirs["preprocessed"]),
        threads=threads,
        grid_cell_size=grid_cell_size
    )
    
    results = {
        "counts": [],
        "exact": {"mean": [], "std": [], "breakdown": []},
        "estimated": {"mean": [], "std": [], "breakdown": []},
        "cgal": {"mean": [], "std": []},
        "touch": {"mean": [], "std": []},
        "result_sizes": []
    }

    for count in CUBE_COUNTS:
        f1_path, f2_path = canonical_cube_pair_paths(
            shared_raw_dir,
            num_cubes_a=200000,
            num_cubes_b=count,
            min_size=1.0,
            max_size=2.0,
            selectivity=0.001,
            seed=42,
            grid_cell_size=grid_cell_size,
        )
        ensure_cube_pair_dataset(
            f1_path,
            f2_path,
            num_cubes_a=200000,
            num_cubes_b=count,
            min_size=1.0,
            max_size=2.0,
            selectivity=0.001,
            seed=42,
        )
        
        print(f"\nProcessing: {f1_path.name} vs {f2_path.name}")

        # Check/Run Preprocessing (Exact/Estimated/Face/TOUCH share preprocessed files)
        if any(a in approaches for a in ["exact", "estimated", "cgal", "touch"]):
            print("Checking preprocessing (Pierce)...")
            # Force preprocessing if not exists or ensure it's up to date
            # Note: PierceAdapter.preprocess_from_source checks existence inside
            exact_adapter.preprocess_from_source(str(f1_path), str(f1_path), log_dir=str(run_log_dir))
            exact_adapter.preprocess_from_source(str(f2_path), str(f2_path), log_dir=str(run_log_dir))

        res_exact = {"mean": 0, "std": 0, "breakdown": {}}
        if "exact" in approaches:
            print(f"Running Exact Mode ({runs} runs)...")
            res_exact = exact_adapter.run_overlap(
                str(f1_path), 
                str(f2_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_exact:
                print(f"Error in exact run: {res_exact['error']}")
                # continue # Assuming if exact fails, we skip this point
                res_exact = {"mean": 0, "std": 0, "breakdown": {}}
            
        # Run Estimated Benchmark
        res_est = {"mean": 0, "std": 0, "breakdown": {}}
        if "estimated" in approaches:
            print(f"Running Estimated Mode ({runs} runs)...")
            res_est = estimated_adapter.run_overlap(
                str(f1_path), 
                str(f2_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_est:
                print(f"Error in estimated run: {res_est['error']}")
                # We continue even if estimated fails? Let's say yes for robustness
                res_est = {"mean": 0, "std": 0, "breakdown": {}}

        # Run Face Benchmark
        res_cgal = {"mean": None, "std": None}
        if "cgal" in approaches:
            print(f"Running Face Mode ({runs} runs)...")
            res_cgal = cgal_adapter.run_overlap(
                str(f1_path), 
                str(f2_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_cgal:
                print(f"Error in Face run: {res_cgal['error']}")
                # Allow Face to fail (e.g. timeout)
                res_cgal = {"mean": None, "std": None}

        # Run TOUCH Benchmark
        res_touch = {"mean": None, "std": None}
        if "touch" in approaches:
            print(f"Running TOUCH Mode ({runs} runs)...")
            res_touch = touch_adapter.run_overlap(
                str(f1_path), 
                str(f2_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_touch:
                print(f"Error in TOUCH run: {res_touch['error']}")
                res_touch = {"mean": None, "std": None}

        results["counts"].append(count)
        
        results["exact"]["mean"].append(res_exact["mean"])
        results["exact"]["std"].append(res_exact["std"])
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["estimated"]["mean"].append(res_est["mean"])
        results["estimated"]["std"].append(res_est["std"])
        results["estimated"]["breakdown"].append(res_est.get("breakdown", {}))
        
        results["cgal"]["mean"].append(res_cgal["mean"])
        results["cgal"]["std"].append(res_cgal["std"])
        
        results["touch"]["mean"].append(res_touch["mean"])
        results["touch"]["std"].append(res_touch["std"])

        result_size = 0
        for res in [res_exact, res_est]:
            if "num_intersections" in res:
                result_size = int(res.get("num_intersections", 0))
                break
        results["result_sizes"].append(result_size)
        
        # Add dataset sizes
        if "num_obj1" not in results:
             results["num_obj1"] = []
             results["num_obj2"] = []
             results["size_bytes1"] = []
             results["size_bytes2"] = []
             results["universe_extents1"] = []
             results["universe_extents2"] = []
        
        results["num_obj1"].append(int(res_exact.get("num_obj1", 0)))
        results["num_obj2"].append(int(res_exact.get("num_obj2", 0)))
        results["size_bytes1"].append(f1_path.stat().st_size if f1_path.exists() else 0)
        results["size_bytes2"].append(f2_path.stat().st_size if f2_path.exists() else 0)
        results["universe_extents1"].append(res_exact.get("universe_extents1", [0.0, 0.0, 0.0]))
        results["universe_extents2"].append(res_exact.get("universe_extents2", [0.0, 0.0, 0.0]))
        
        cgal_str = f"{res_cgal['mean']:.2f}ms" if res_cgal['mean'] else "TIMEOUT/ERR"
        touch_str = f"{res_touch['mean']:.2f}ms" if res_touch['mean'] else "TIMEOUT/ERR"
        print(f"Done {count}: Exact={res_exact['mean']:.2f}ms, Est={res_est['mean']:.2f}ms, Face={cgal_str}, TOUCH={touch_str}")

    return results

def plot_results(results, figures_dir):
    print("\nPlotting results...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    counts = results["counts"]
    if not counts:
        print("No results to plot.")
        return

    apply_paper_style()
    fig, (ax_main, ax_breakdown) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)
    all_y_vals = []

    # --- Plot 1: Line Chart (Scaling) ---
    ax_main.errorbar(counts, results["exact"]["mean"], 
                     fmt='-o', label='Pierce (Two Pass)', capsize=5, color=style_for("exact")["color"])
    all_y_vals.extend([v for v in results["exact"]["mean"] if isinstance(v, (int, float)) and v > 0])
    ax_main.errorbar(counts, results["estimated"]["mean"], 
                     fmt='-s', label='Pierce (Selectivity Estimation)', capsize=5, color=style_for("estimated")["color"])
    all_y_vals.extend([v for v in results["estimated"]["mean"] if isinstance(v, (int, float)) and v > 0])
    
    # Filter valid Face points
    cgal_valid_indices = [i for i, m in enumerate(results["cgal"]["mean"]) if m is not None]
    if cgal_valid_indices:
        cgal_counts = [counts[i] for i in cgal_valid_indices]
        cgal_means = [results["cgal"]["mean"][i] for i in cgal_valid_indices]
        cgal_stds = [results["cgal"]["std"][i] for i in cgal_valid_indices]
        all_y_vals.extend(cgal_means)
        ax_main.errorbar(cgal_counts, cgal_means, 
                         fmt='-D', label='Face', capsize=5, color=style_for("cgal")["color"])

    # Filter valid TOUCH points
    touch_valid_indices = [i for i, m in enumerate(results["touch"]["mean"]) if m is not None]
    if touch_valid_indices:
        touch_counts = [counts[i] for i in touch_valid_indices]
        touch_means = [results["touch"]["mean"][i] for i in touch_valid_indices]
        touch_stds = [results["touch"]["std"][i] for i in touch_valid_indices]
        all_y_vals.extend(touch_means)
        ax_main.errorbar(touch_counts, touch_means, 
                         fmt='-^', label='TOUCH', capsize=5, color=style_for("touch")["color"])

    ax_main.set_xlabel('Number of Cubes in Dataset B (A=200k)')
    ax_main.set_ylabel('Execution Time (ms) [Log Scale]')
    ax_main.set_yscale('log')
    set_log_timing_axis_limits(ax_main, all_y_vals)
    make_legend_bold(ax_main)
    ax_main.grid(False)
    ax_main.set_xticks(counts)

    # --- Plot 2: Breakdown Bar Chart (Exact & Estimated ONLY) ---
    # Breakdown visual settings
    phase_mapping = {
        "selectivity estimation_": "Selectivity Est.",
        "execute hash query_": "Hash Query",
        "query_": "Ray Query",
        "gpu deduplication_": "Deduplication",
        "download results_": "Download"
    }
    ordered_phases_raw = [
        "selectivity estimation_",
        "query_",
        "execute hash query_",
        "gpu deduplication_",
        "download results_"
    ]
    colors = {
        "selectivity estimation_": "#ff9999", 
        "query_": "#66b3ff",              
        "execute hash query_": "#3399ff",   
        "gpu deduplication_": "#99ff99",    
        "download results_": "#ffcc99"      
    }
    
    # Collect active phases
    all_active_phases = set(ordered_phases_raw)
    for mode in ["exact", "estimated"]:
        for bd in results[mode]["breakdown"]:
            all_active_phases.update(bd.keys())
    
    active_phases_ordered = [p for p in ordered_phases_raw if p in all_active_phases]
    for p in all_active_phases:
        if p not in active_phases_ordered: active_phases_ordered.append(p)

    legend_handles = []
    legend_labels = []
    for phase in active_phases_ordered:
        label = phase_mapping.get(phase, phase)
        color = colors.get(phase, "#cccccc")
        patch = plt.Rectangle((0, 0), 1, 1, fc=color, ec='white')
        legend_handles.append(patch)
        legend_labels.append(label)

    modes_to_plot = ["exact", "estimated"]
    num_modes = len(modes_to_plot)
    group_width = 0.8
    mode_width = group_width / num_modes
    
    # Use indices (0, 1, 2...) for x-axis of bar chart, label with counts
    x_indices = range(len(counts))

    for i, count_idx in enumerate(x_indices):
        for j, mode in enumerate(modes_to_plot):
            x_pos = i - group_width/2 + (j + 0.5) * mode_width
            
            # Get breakdown for this run
            breakdown = results[mode]["breakdown"][i]
            mean_time = results[mode]["mean"][i] # Fallback if no breakdown
            
            if not breakdown:
                ax_breakdown.bar(x_pos, mean_time, mode_width, color="#cccccc", edgecolor='white', alpha=0.5)
            else:
                bottom = 0
                for phase in active_phases_ordered:
                    val = breakdown.get(phase, 0.0)
                    if val > 0:
                        ax_breakdown.bar(x_pos, val, mode_width, bottom=bottom, 
                                         color=colors.get(phase, None), edgecolor='white')
                        bottom += val

    ax_breakdown.set_xticks(x_indices)
    ax_breakdown.set_xticklabels([f"{c//1000}k" for c in counts])
    ax_breakdown.set_xlabel('Dataset Size (Cubes)')
    ax_breakdown.set_ylabel('Query Time (ms)')
    ax_breakdown.grid(False)    
    # Legend
    make_legend_bold(
        ax_breakdown,
        legend_handles,
        legend_labels,
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
    )

    plt.tight_layout()
    output_path = figures_dir / "mesh_overlap_cube_scalability.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {output_path}")
    
    # Also save PDF
    pdf_path = str(output_path).replace('.png', '.pdf')
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"PDF saved to {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Cube Scalability Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument("--grid-cell-size", type=float, default=5.0, help="Grid cell size for RaySpace")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds per run")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for CGAL/TOUCH")
    parser.add_argument("--approaches", type=str, nargs="+", default=["exact", "estimated", "cgal", "touch"], 
                        choices=["exact", "estimated", "cgal", "touch"], help="Approaches to run")
    args = parser.parse_args()
    
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_cube_scalability")
    run_log_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    results = run_experiment(args.runs, args.grid_cell_size, run_log_dir, threads=args.threads, approaches=args.approaches, timeout=args.timeout)
    
    if results and results["counts"]:
        print("\nResults Summary:")
        print(f"{'Count':<10} {'Exact (ms)':<15} {'Estimated (ms)':<15} {'Face (ms)':<15} {'TOUCH (ms)':<15}")
        for i, n in enumerate(results["counts"]):
            ex = results['exact']['mean'][i]
            est = results['estimated']['mean'][i]
            cg = results['cgal']['mean'][i]
            to = results['touch']['mean'][i]
            cg_str = f"{cg:.2f}" if cg else "N/A"
            to_str = f"{to:.2f}" if to else "N/A"
            print(f"{n:<10} {ex:<15.2f} {est:<15.2f} {cg_str:<15} {to_str:<15}")
                
        plot_results(results, figures_dir)
        
        # Save summary to canonical run results path
        out_json = Path(run_layout["results_json"])
        clean_results = {}
        for k, v in results.items():
            if isinstance(v, dict):
                clean_results[k] = {ki: (vi.tolist() if isinstance(vi, np.ndarray) else vi) for ki, vi in v.items()}
            elif isinstance(v, list):
                clean_results[k] = v
            else:
                clean_results[k] = v
        write_json(
            out_json,
            {
                "metadata": {
                    "timestamp": run_layout["timestamp"],
                    "run_name": run_layout["run_name"],
                    "run_dir": str(run_layout["run_dir"]),
                    "runs": args.runs,
                    "grid_cell_size": args.grid_cell_size,
                    "timeout": args.timeout,
                },
                "results": clean_results,
            },
        )
        print(f"Raw results saved to {out_json}")
    else:
        print("No successful runs.")

if __name__ == "__main__":
    main()
