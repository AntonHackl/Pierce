#!/usr/bin/env python3
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import argparse
import sys
from pathlib import Path
from datetime import datetime
import subprocess 
import json
import re
import shutil
# Add project root to sys.path to allow imports from 'benchmarks'
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import (
    PAPER_FIGSIZE,
    PAPER_WIDE_FIGSIZE,
    PAPER_SIDE_BY_SIDE_FIGSIZE,
    apply_paper_style,
    apply_side_by_side_style,
    make_legend_bold,
    set_log_timing_axis_limits,
    style_for,
)
from benchmarks.common.scenario_utils import (
    canonical_nn_pair_paths,
    canonical_nu_pair_paths,
    count_triangles,
    create_isolated_run_data_dirs,
    ensure_nn_pair_dataset,
    ensure_nu_pair_dataset,
    get_shared_data_dirs,
    stage_input_files,
)

# Add current directory to path to import adapters
sys.path.append(str(Path(__file__).parent))
from adapters.pierce_adapter import PierceAdapter
from adapters.cgal_adapter import CGALAdapter
from adapters.touch_adapter import TOUCHAdapter
from adapters.tdbase_adapter import TDBaseAdapter
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODES,
)

# Configuration
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
PIERCE_DIR = REPO_ROOT / "pierce"
CGAL_DIR = REPO_ROOT / "baselines/face"
TDBASE_DIR = REPO_ROOT / "baselines/tdbase_extensions"
DATA_DIR = SCRIPT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PREPROCESSED_DIR = DATA_DIR / "preprocessed"
TIMINGS_DIR = DATA_DIR / "timings"
FIGURES_DIR = SCRIPT_DIR / "figures"
RUNS_DIR = SCRIPT_DIR / "runs"

TIMEOUT_SECONDS = 120.0

# Nu Counts for Dataset B (Dataset A is fixed at corresponding vessel count)
DEFAULT_NU_COUNTS = [200, 400, 600, 800]
SUPPORTED_NU_COUNTS_BY_PROFILE = {
    "standard": DEFAULT_NU_COUNTS,
    "large_nu_v": [200, 400, 600, 800],
    "large_nu_nn": [200, 400],
}
LEGACY_RAW_DIR = RAW_DIR
SHARED_SCENARIO = "nu_scalability"
LARGE_NU_V_LEGACY_RAW_DIR = SCRIPT_DIR / "data" / "raw"

def resolve_dataset_pair(
    raw_shared_dir: Path,
    nu: int,
    *,
    nv: int = 150,
    prefix: str = "tdbase",
    profile_kind: str = "nu_v",
    legacy_raw_dirs=None,
):
    if profile_kind == "nu_nn":
        n_file1, n_file2 = canonical_nn_pair_paths(raw_shared_dir, nu=nu, nv=nv, prefix=prefix)
        ensure_nn_pair_dataset(
            n_file1,
            n_file2,
            legacy_raw_dirs=legacy_raw_dirs if legacy_raw_dirs is not None else [LEGACY_RAW_DIR],
        )
        return n_file1, n_file2

    n_file, v_file = canonical_nu_pair_paths(raw_shared_dir, nu=nu, nv=nv, prefix=prefix)
    ensure_nu_pair_dataset(
        n_file,
        v_file,
        legacy_raw_dirs=legacy_raw_dirs if legacy_raw_dirs is not None else [LEGACY_RAW_DIR],
    )
    return v_file, n_file


def _filter_supported_nu_counts(nu_counts, dataset_profile):
    supported = SUPPORTED_NU_COUNTS_BY_PROFILE.get(dataset_profile, DEFAULT_NU_COUNTS)
    filtered = [nu for nu in nu_counts if nu in supported]
    dropped = [nu for nu in nu_counts if nu not in supported]
    if dropped:
        print(
            f"Skipping unsupported nu counts for dataset profile '{dataset_profile}': "
            f"{dropped}. Supported counts: {supported}"
        )
    return filtered

def run_experiment(
    runs,
    grid_cell_size,
    nu_counts,
    run_log_dir,
    isolated_data_dirs,
    threads=None,
    tdbase_threads=None,
    tdbase_compute_threads=1,
    approaches=None,
    track_hash_contention=False,
    timeout=120.0,
    dataset_profile="standard",
    tdbase_timing_mode=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
):
    if approaches is None:
        approaches = ["exact", "direct_estimation", "cgal", "touch", "tdbase"]
    
    print(f"--- Starting Nu Scalability Experiment ({nu_counts}) ---")
    print(f"Approaches: {approaches}")
    print(f"CGAL/TOUCH threads: {threads if threads else 'OpenMP default'}")
    print(f"TDBase threads: {tdbase_threads if tdbase_threads else 'TDBase default'} (compute threads: {tdbase_compute_threads})")
    if track_hash_contention:
        print("Direct estimation hash contention tracking: enabled")
    
    # Ensure directories exist
    profile_kind = "nu_v"
    if dataset_profile == "large_nu_v":
        shared_dirs = get_shared_data_dirs("large_nu_nn_scalability")
        dataset_nv = 750
        dataset_prefix = "tdbase_large"
        legacy_raw_dirs = [LARGE_NU_V_LEGACY_RAW_DIR, LEGACY_RAW_DIR]
    elif dataset_profile == "large_nu_nn":
        shared_dirs = get_shared_data_dirs("large_nu_nn_scalability")
        dataset_nv = 750
        dataset_prefix = "tdbase_large"
        profile_kind = "nu_nn"
        legacy_raw_dirs = [LARGE_NU_V_LEGACY_RAW_DIR, LEGACY_RAW_DIR]
    else:
        shared_dirs = get_shared_data_dirs(SHARED_SCENARIO)
        dataset_nv = 150
        dataset_prefix = "tdbase"
        legacy_raw_dirs = [LEGACY_RAW_DIR]

    shared_raw_dir = shared_dirs["raw"]
    isolated_raw_dir = isolated_data_dirs["raw"]
    isolated_preprocessed_dir = isolated_data_dirs["preprocessed"]
    isolated_timings_dir = isolated_data_dirs["timings"]

    print(f"Logging runs to: {run_log_dir}")
    print(f"RaySpace Dir: {PIERCE_DIR}")
    print(f"Isolated data dir: {isolated_data_dirs['root']}")

    # Initialize Adapters
    print("Initializing adapters...")
    exact_adapter = PierceAdapter(
        str(PIERCE_DIR), 
        mode="exact", 
        preprocessed_dir=str(isolated_preprocessed_dir), 
        timings_dir=str(isolated_timings_dir),
        grid_cell_size=grid_cell_size,
        warmup_runs=1
    )
    
    direct_estimation_adapter = PierceAdapter(
        str(PIERCE_DIR), 
        mode="direct_estimation", 
        preprocessed_dir=str(isolated_preprocessed_dir), 
        timings_dir=str(isolated_timings_dir),
        grid_cell_size=grid_cell_size,
        warmup_runs=1,
        track_hash_contention=track_hash_contention,
    )

    cgal_adapter = CGALAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(isolated_preprocessed_dir),
        threads=threads,
        grid_cell_size=grid_cell_size
    )
    
    touch_adapter = TOUCHAdapter(
        str(CGAL_DIR),
        preprocessed_dir=str(isolated_preprocessed_dir),
        threads=threads,
        grid_cell_size=grid_cell_size
    )

    tdbase_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=str(isolated_raw_dir),
        threads=tdbase_threads,
        compute_threads=tdbase_compute_threads,
        query_timing_mode=tdbase_timing_mode,
    )
    
    results = {
        "counts": [],
        "enabled_approaches": approaches,
        "exact": {"mean": [], "std": [], "breakdown": []},
        "direct_estimation": {"mean": [], "std": [], "breakdown": []},
        "cgal": {"mean": [], "std": []},
        "touch": {"mean": [], "std": []},
        "tdbase": {"mean": [], "std": []},
        "result_sizes": []
    }

    for nu in nu_counts:
        f_v_path, f_n_path = resolve_dataset_pair(
            shared_raw_dir,
            nu,
            nv=dataset_nv,
            prefix=dataset_prefix,
            profile_kind=profile_kind,
            legacy_raw_dirs=legacy_raw_dirs,
        )
        
        if not f_v_path or not f_n_path:
            print(f"Error: Datasets for nu={nu} not found in {shared_raw_dir}! Skipping.")
            continue

        staged_v_path, staged_n_path = stage_input_files(
            [f_v_path, f_n_path],
            isolated_raw_dir,
        )
        
        print(f"\nProcessing nu={nu}: {f_v_path.name} vs {f_n_path.name}")

        # Check/Run Preprocessing for Pierce (also used by Face/TOUCH adapters)
        needs_preprocessing = any(a in approaches for a in ["exact", "direct_estimation", "cgal", "touch"])
        if needs_preprocessing:
            print("Checking preprocessing...")
            exact_adapter.preprocess_from_source(str(staged_v_path), str(staged_v_path), log_dir=str(run_log_dir))
            exact_adapter.preprocess_from_source(str(staged_n_path), str(staged_n_path), log_dir=str(run_log_dir))

        # Run Exact Benchmark
        res_exact = {"mean": None, "std": None, "breakdown": {}}
        if "exact" in approaches:
            print(f"Running Exact Mode ({runs} runs)...")
            res_exact = exact_adapter.run_overlap(
                str(staged_v_path), 
                str(staged_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_exact:
                print(f"Error in exact run: {res_exact['error']}")
                res_exact = {"mean": None, "std": None, "breakdown": {}}
            
        # Run Selectivity Estimation Benchmark
        res_direct = {"mean": None, "std": None, "breakdown": {}}
        if "direct_estimation" in approaches:
            print(f"Running Selectivity Estimation Mode ({runs} runs)...")
            res_direct = direct_estimation_adapter.run_overlap(
                str(staged_v_path), 
                str(staged_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_direct:
                print(f"Error in selectivity estimation run: {res_direct['error']}")
                res_direct = {"mean": None, "std": None, "breakdown": {}}

        # Run Face Benchmark
        res_cgal = {"mean": None, "std": None}
        if "cgal" in approaches:
            print(f"Running Face Mode ({runs} runs)...")
            res_cgal = cgal_adapter.run_overlap(
                str(staged_v_path), 
                str(staged_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_cgal:
                print(f"Error in Face run: {res_cgal['error']}")
                res_cgal = {"mean": None, "std": None}

        # Run TOUCH Benchmark
        res_touch = {"mean": None, "std": None}
        if "touch" in approaches:
            print(f"Running TOUCH Mode ({runs} runs)...")
            res_touch = touch_adapter.run_overlap(
                str(staged_v_path), 
                str(staged_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_touch:
                print(f"Error in TOUCH run: {res_touch['error']}")
                res_touch = {"mean": None, "std": None}

        # Run TDBase Benchmark
        res_td = {"mean": None, "std": None}
        if "tdbase" in approaches:
            # Keep TDBase inputs symmetric and fresh for this pair.
            tdbase_adapter.preprocess_from_source(str(staged_v_path), str(staged_v_path), log_dir=str(run_log_dir))
            tdbase_adapter.preprocess_from_source(str(staged_n_path), str(staged_n_path), log_dir=str(run_log_dir))
            print(f"Running TDBase Mode ({runs} runs)...")
            res_td = tdbase_adapter.run_overlap(
                str(staged_v_path), 
                str(staged_n_path), 
                runs,
                log_dir=str(run_log_dir),
                timeout=timeout
            )
            if "error" in res_td:
                print(f"Error in TDBase run: {res_td['error']}")
                res_td = {"mean": None, "std": None}

        results["counts"].append(nu)
        
        results["exact"]["mean"].append(res_exact["mean"])
        results["exact"]["std"].append(res_exact["std"])
        results["exact"]["breakdown"].append(res_exact.get("breakdown", {}))
        
        results["direct_estimation"]["mean"].append(res_direct["mean"])
        results["direct_estimation"]["std"].append(res_direct["std"])
        results["direct_estimation"]["breakdown"].append(res_direct.get("breakdown", {}))
        
        results["cgal"]["mean"].append(res_cgal["mean"])
        results["cgal"]["std"].append(res_cgal["std"])
        
        results["touch"]["mean"].append(res_touch["mean"])
        results["touch"]["std"].append(res_touch["std"])

        results["tdbase"]["mean"].append(res_td["mean"])
        results["tdbase"]["std"].append(res_td["std"])
        
        # Add dataset sizes
        if "num_obj1" not in results:
             results["num_obj1"] = []
             results["num_obj2"] = []
             results["num_intersections"] = []
             results["selectivity"] = []
             results["size_bytes1"] = []
             results["size_bytes2"] = []
             results["num_triangles1"] = []
             results["num_triangles2"] = []
             results["universe_extents1"] = []
             results["universe_extents2"] = []
        
        # Use first available result for counts
        found_counts = False
        for res in [res_exact, res_direct]:
            if "num_obj1" in res and res["num_obj1"] > 0:
                results["num_obj1"].append(int(res["num_obj1"]))
                results["num_obj2"].append(int(res["num_obj2"]))
                num_intersections = int(res.get("num_intersections", 0))
                results["num_intersections"].append(num_intersections)
                results["result_sizes"].append(num_intersections)
                cross_product_size = int(res["num_obj1"]) * int(res["num_obj2"])
                results["selectivity"].append(
                    (num_intersections / cross_product_size) if cross_product_size > 0 else 0.0
                )
                results["universe_extents1"].append(res.get("universe_extents1", [0.0, 0.0, 0.0]))
                results["universe_extents2"].append(res.get("universe_extents2", [0.0, 0.0, 0.0]))
                found_counts = True
                break
        
        if not found_counts:
            results["num_obj1"].append(0)
            results["num_obj2"].append(0)
            results["num_intersections"].append(0)
            results["selectivity"].append(0.0)
            results["universe_extents1"].append([0.0, 0.0, 0.0])
            results["universe_extents2"].append([0.0, 0.0, 0.0])
            results["result_sizes"].append(0)

        results["size_bytes1"].append(f_v_path.stat().st_size if f_v_path.exists() else 0)
        results["size_bytes2"].append(f_n_path.stat().st_size if f_n_path.exists() else 0)
        results["num_triangles1"].append(count_triangles(f_v_path) if f_v_path.exists() else 0)
        results["num_triangles2"].append(count_triangles(f_n_path) if f_n_path.exists() else 0)
        
        exact_str = f"{res_exact['mean']:.2f}ms" if res_exact['mean'] is not None else "N/A"
        direct_str = f"{res_direct['mean']:.2f}ms" if res_direct['mean'] is not None else "N/A"
        cgal_str = f"{res_cgal['mean']:.2f}ms" if res_cgal['mean'] is not None else "N/A"
        touch_str = f"{res_touch['mean']:.2f}ms" if res_touch['mean'] is not None else "N/A"
        td_str = f"{res_td['mean']:.2f}ms" if res_td['mean'] is not None else "N/A"
        print(f"Done nu={nu}: Exact={exact_str}, Selectivity Estimation={direct_str}, Face={cgal_str}, TOUCH={touch_str}, TDBase={td_str}")

    return results

def plot_results(results, figures_dir):
    print("\nPlotting results...")
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    counts = results["counts"]
    if not counts:
        print("No results to plot.")
        return

    # --- Plot 1: Line Chart (Scaling) ---
    def generate_scaling_plot(ax, results, counts):
        enabled = set(results.get("enabled_approaches", ["exact", "direct_estimation", "cgal", "touch", "tdbase"]))
        all_y_vals = []
        
        x_vals = []
        for i, nu in enumerate(counts):
            obj2 = results.get("num_obj2", [])[i] if "num_obj2" in results else 0
            if obj2 == 0:
                obj2 = nu * 729
            x_vals.append(obj2)

        x_sel_labels = []
        for i, val in enumerate(x_vals):
            if "num_intersections" in results and i < len(results["num_intersections"]) and results.get("num_obj1", [])[i] > 0 and results.get("num_obj2", [])[i] > 0:
                sel = results["num_intersections"][i] / (results["num_obj1"][i] * results["num_obj2"][i])
                x_sel_labels.append(f"{sel*100:.1e}%")
            else:
                x_sel_labels.append("")

        def get_clean_style(approach):
            st = style_for(approach).copy()
            st["label"] = st["label"].replace(" (Estimation Only)", "")
            st["linestyle"] = "-"
            if st.get("marker") is None:
                markers = {"exact": "o", "direct_estimation": "s", "cgal": "D", "touch": "^", "tdbase": "v"}
                st["marker"] = markers.get(approach, "o")
            return st

        if "exact" in enabled:
            exact_valid_indices = [i for i, m in enumerate(results["exact"]["mean"]) if m is not None]
            if exact_valid_indices:
                exact_x = [x_vals[i] for i in exact_valid_indices]
                exact_means = [results["exact"]["mean"][i] for i in exact_valid_indices]
                exact_stds = [results["exact"]["std"][i] for i in exact_valid_indices]
                all_y_vals.extend(exact_means)
                st = get_clean_style("exact")
                ax.errorbar(
                    exact_x,
                    exact_means,
                    yerr=exact_stds,
                    linestyle=st.get("linestyle", "-"),
                    marker=st.get("marker"),
                    capsize=5,
                    color=st["color"],
                    label=None
                )
                ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        if "direct_estimation" in enabled:
            direct_valid_indices = [i for i, m in enumerate(results["direct_estimation"]["mean"]) if m is not None]
            if direct_valid_indices:
                direct_x = [x_vals[i] for i in direct_valid_indices]
                direct_means = [results["direct_estimation"]["mean"][i] for i in direct_valid_indices]
                direct_stds = [results["direct_estimation"]["std"][i] for i in direct_valid_indices]
                all_y_vals.extend(direct_means)
                st = get_clean_style("direct_estimation")
                ax.errorbar(
                    direct_x,
                    direct_means,
                    yerr=direct_stds,
                    linestyle=st.get("linestyle", "-"),
                    marker=st.get("marker"),
                    capsize=5,
                    color=st["color"],
                    label=None
                )
                ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        # Filter valid Face points
        cgal_valid_indices = [i for i, m in enumerate(results["cgal"]["mean"]) if m is not None] if "cgal" in enabled else []
        if cgal_valid_indices:
            cgal_x = [x_vals[i] for i in cgal_valid_indices]
            cgal_means = [results["cgal"]["mean"][i] for i in cgal_valid_indices]
            cgal_stds = [results["cgal"]["std"][i] for i in cgal_valid_indices]
            all_y_vals.extend(cgal_means)
            st = get_clean_style("cgal")
            ax.errorbar(
                cgal_x,
                cgal_means,
                yerr=cgal_stds,
                linestyle=st.get("linestyle", "-"),
                marker=st.get("marker"),
                capsize=5,
                color=st["color"],
                label=None
            )
            ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        # Filter valid TOUCH points
        touch_valid_indices = [i for i, m in enumerate(results["touch"]["mean"]) if m is not None] if "touch" in enabled else []
        if touch_valid_indices:
            touch_x = [x_vals[i] for i in touch_valid_indices]
            touch_means = [results["touch"]["mean"][i] for i in touch_valid_indices]
            touch_stds = [results["touch"]["std"][i] for i in touch_valid_indices]
            all_y_vals.extend(touch_means)
            st = get_clean_style("touch")
            ax.errorbar(
                touch_x,
                touch_means,
                yerr=touch_stds,
                linestyle=st.get("linestyle", "-"),
                marker=st.get("marker"),
                capsize=5,
                color=st["color"],
                label=None
            )
            ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        # Filter valid TDBase points
        td_valid_indices = [i for i, m in enumerate(results["tdbase"]["mean"]) if m is not None] if "tdbase" in enabled else []
        if td_valid_indices:
            td_x = [x_vals[i] for i in td_valid_indices]
            td_means = [results["tdbase"]["mean"][i] for i in td_valid_indices]
            td_stds = [results["tdbase"]["std"][i] for i in td_valid_indices]
            all_y_vals.extend(td_means)
            st = get_clean_style("tdbase")
            ax.errorbar(
                td_x,
                td_means,
                yerr=td_stds,
                linestyle=st.get("linestyle", "-"),
                marker=st.get("marker"),
                capsize=5,
                color=st["color"],
                label=None
            )
            ax.plot([], [], linestyle=st.get("linestyle", "-"), marker=st.get("marker"), label=st["label"], color=st["color"])

        ax.set_xlabel('Number of Nuclei', fontsize=16)
        ax.set_ylabel('Query Time (ms)', fontsize=16)
        ax.set_yscale('log')
        ax.tick_params(axis='both', labelsize=13)
        
        # Format X-axis with 'k' for thousands
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f'{x/1e3:g}k' if x >= 1e3 else f'{x:g}'))

        # Ensure the x-axis starts at 100k when all data points are >= 100k
        try:
            min_x = min(x_vals)
            max_x = max(x_vals)
            left_lim = 100_000 if min_x >= 100_000 else min_x
            # add small padding to the right so markers aren't clipped
            right_lim = max_x * 1.05 if max_x > left_lim else left_lim * 1.05
            ax.set_xlim(left=left_lim, right=right_lim)
            # Use 100k major ticks for clearer labels (will include 100k)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(100000))
        except Exception:
            # fallback to default autoscaling
            pass

        # Secondary axis for selectivity removed as requested

        set_log_timing_axis_limits(ax, all_y_vals)
        make_legend_bold(ax, loc='center right', fontsize=12)
        ax.grid(False)

    # --- Plot 2: Breakdown Bar Chart ---
    def generate_breakdown_plot(ax, results, counts):
        enabled = set(results.get("enabled_approaches", ["exact", "direct_estimation", "cgal", "touch", "tdbase"]))
        
        x_vals = []
        for i, nu in enumerate(counts):
            obj2 = results.get("num_obj2", [])[i] if "num_obj2" in results else 0
            if obj2 == 0:
                obj2 = nu * 729
            x_vals.append(obj2)

        x_labels = []
        x_selectivities = []
        for i, val in enumerate(x_vals):
            x_labels.append(f"{val/1e3:g}k")
            if "num_intersections" in results and i < len(results["num_intersections"]) and results.get("num_obj1", [])[i] > 0 and results.get("num_obj2", [])[i] > 0:
                sel = results["num_intersections"][i] / (results["num_obj1"][i] * results["num_obj2"][i])
                x_selectivities.append(sel)
            else:
                x_selectivities.append(None)

        def normalize_phase_key(phase: str) -> str:
            key = re.sub(r"_\d+$", "", phase.lower())
            key = re.sub(r"_+$", "", key)
            return key

        phase_labels = {
            "selectivity estimation": "Selectivity Est.",
            "query": "Ray Query",
            "execute hash query": "Hash Query",
            "gpu deduplication": "Deduplication",
            "download results": "Download",
            "raytrace_mesh1tomesh2_pass1": "Raytrace M1→M2 (Pass 1)",
            "raytrace_mesh2tomesh1_pass1": "Raytrace M2→M1 (Pass 1)",
            "raytrace_mesh2tomesh1_pass2": "Raytrace M2→M1 (Pass 2)",
            "raytrace_mesh1tomesh2_pass2": "Raytrace M1→M2 (Pass 2)",
            "raytrace_hash_mesh1tomesh2": "Hash Raytrace M1→M2",
            "raytrace_hash_mesh2tomesh1": "Hash Raytrace M2→M1",
        }
        phase_order = [
            "selectivity estimation",
            "raytrace_mesh1tomesh2_pass1",
            "raytrace_mesh2tomesh1_pass1",
            "raytrace_mesh1tomesh2_pass2",
            "raytrace_mesh2tomesh1_pass2",
            "raytrace_hash_mesh1tomesh2",
            "raytrace_hash_mesh2tomesh1",
            "query",
            "execute hash query",
            "gpu deduplication",
            "download results",
        ]
        phase_colors = {
            "selectivity estimation": "#ff9896",
            "raytrace_mesh1tomesh2_pass1": "#1f77b4",
            "raytrace_mesh2tomesh1_pass1": "#17becf",
            "raytrace_mesh2tomesh1_pass2": "#2ca02c",
            "raytrace_mesh1tomesh2_pass2": "#9467bd",
            "raytrace_hash_mesh1tomesh2": "#bcbd22",
            "raytrace_hash_mesh2tomesh1": "#8c564b",
            "query": "#aec7e8",
            "execute hash query": "#7f7f7f",
            "gpu deduplication": "#98df8a",
            "download results": "#ffbb78",
        }

        modes_to_plot = [
            mode for mode in ["exact", "direct_estimation"]
            if mode in enabled
            if any((m is not None and m > 0) for m in results[mode]["mean"])
        ]
        if not modes_to_plot:
            return

        normalized_breakdowns = {mode: [] for mode in modes_to_plot}
        all_active_phases = set()
        for mode in modes_to_plot:
            for bd in results[mode]["breakdown"]:
                merged = {}
                for key, value in bd.items():
                    nk = normalize_phase_key(key)
                    merged[nk] = merged.get(nk, 0.0) + value
                normalized_breakdowns[mode].append(merged)
                all_active_phases.update(merged.keys())

        active_phases_ordered = [p for p in phase_order if p in all_active_phases]
        for p in sorted(all_active_phases):
            if p not in active_phases_ordered:
                active_phases_ordered.append(p)

        legend_handles = []
        legend_labels = []
        for phase in active_phases_ordered:
            label = phase_labels.get(phase, phase.replace("_", " ").title())
            color = phase_colors.get(phase, "#cccccc")
            patch = plt.Rectangle((0, 0), 1, 1, fc=color, ec='white')
            legend_handles.append(patch)
            legend_labels.append(label)

        num_modes = len(modes_to_plot)
        group_width = 0.8
        mode_width = group_width / num_modes if num_modes > 0 else group_width
        
        x_indices = np.arange(len(counts))

        for i, count_idx in enumerate(x_indices):
            for j, mode in enumerate(modes_to_plot):
                x_pos = i - group_width/2 + (j + 0.5) * mode_width
                
                # Get breakdown for this run
                breakdown = normalized_breakdowns[mode][i]
                mean_time = results[mode]["mean"][i]
                
                if not breakdown or mean_time == 0:
                    pass
                else:
                    bottom = 0
                    for phase in active_phases_ordered:
                        val = breakdown.get(phase, 0.0)
                        if val > 0:
                            rects = ax.bar(x_pos, val, mode_width, bottom=bottom, 
                                             color=phase_colors.get(phase, "#cccccc"), edgecolor='white')
                            bottom += val
                            
                            # Add selectivity text once per group (on top of the last bar of the last approach)
                            if j == len(modes_to_plot) - 1 and phase == active_phases_ordered[-1]:
                                sel = x_selectivities[i]
                                if sel is not None:
                                    ax.text(i, bottom + 5, f"sel={sel*100:.1e}%", ha='center', va='bottom', fontsize=12, fontweight='bold', rotation=90)

        ax.set_xticks(x_indices)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel('Number of Nuclei', fontsize=16)
        ax.set_ylabel('Query Time (ms)', fontsize=16)
        ax.tick_params(axis='both', labelsize=13)
        ax.grid(False)
        for j, mode in enumerate(modes_to_plot):
            x_annot = 0 - group_width/2 + (j + 0.5) * mode_width
            ax.text(
                x_annot,
                -0.08,
                "Exact" if mode == "exact" else "Direct",
                ha='center',
                va='top',
                transform=ax.get_xaxis_transform(),
                fontsize=14,
                fontweight='bold',
                color="#444444",
            )
        
        # Legend
        if legend_handles:
            make_legend_bold(
                ax,
                legend_handles,
                legend_labels,
                bbox_to_anchor=(1.02, 1),
                loc='upper left',
                fontsize=13,
                ncol=1,
                frameon=True,
            )

    # 1. Generate Combined Figure
    apply_side_by_side_style()
    fig, (ax_main, ax_breakdown) = plt.subplots(1, 2, figsize=PAPER_FIGSIZE)
    generate_scaling_plot(ax_main, results, counts)
    generate_breakdown_plot(ax_breakdown, results, counts)
    plt.tight_layout()
    combined_path = figures_dir / "mesh_overlap_nu_scalability.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(combined_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Combined visualization saved to {combined_path}")
    plt.close(fig)

    # 2. Generate Separate Scaling Figure
    apply_side_by_side_style()
    fig_scaling, ax_scaling = plt.subplots(figsize=(6.2, 4.1))
    generate_scaling_plot(ax_scaling, results, counts)
    plt.tight_layout()
    scaling_path = figures_dir / "mesh_overlap_nu_scalability_scaling.png"
    plt.savefig(scaling_path, dpi=300)
    plt.savefig(str(scaling_path).replace('.png', '.pdf'))
    print(f"Scaling visualization saved to {scaling_path}")
    plt.close(fig_scaling)

    # 3. Generate Separate Breakdown Figure
    apply_paper_style()
    fig_breakdown, ax_breakdown_sep = plt.subplots(figsize=PAPER_WIDE_FIGSIZE)
    generate_breakdown_plot(ax_breakdown_sep, results, counts)
    plt.tight_layout()
    breakdown_path = figures_dir / "mesh_overlap_nu_scalability_breakdown.png"
    plt.savefig(breakdown_path, dpi=300, bbox_inches='tight')
    plt.savefig(str(breakdown_path).replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Breakdown visualization saved to {breakdown_path}")
    plt.close(fig_breakdown)

def main():
    parser = argparse.ArgumentParser(description="Mesh Overlap Nu Scalability Experiment")
    parser.add_argument("--runs", type=int, default=5, help="Number of runs per method")
    parser.add_argument("--grid-cell-size", type=float, default=200.0, help="Grid resolution for RaySpace")
    parser.add_argument("--nu", type=int, nargs='+', help="Nu counts to test (e.g. 200 400 600 800)")
    parser.add_argument("--approaches", type=str, nargs='+', choices=["exact", "direct_estimation", "cgal", "touch", "tdbase"], help="Approaches to run")
    parser.add_argument("--track-hash-contention", action="store_true", help="Enable overlap hash contention tracking for Pierce direct estimation")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads for CGAL/TOUCH")
    parser.add_argument("--tdbase-threads", type=int, default=None, help="Number of TDBase join threads")
    parser.add_argument("--tdbase-compute-threads", type=int, default=1, help="Number of TDBase compute threads per tile")
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout in seconds per run")
    parser.add_argument("--revisualize", type=str, help="Path to results.json to re-generate plots from")
    parser.add_argument(
        "--tdbase-timing-mode",
        type=str,
        default=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
        choices=TDBASE_TIMING_MODES,
        help="TDBase query-time definition. Default uses index+compute+evaluate; use compute_only to revert.",
    )
    parser.add_argument(
        "--dataset-profile",
        type=str,
        choices=["standard", "large_nu_v", "large_nu_nn"],
        default="standard",
        help="Dataset source profile: standard nu_scalability, large_nu_v, or large_nu_nn generated data",
    )
    args = parser.parse_args()
    
    if args.revisualize:
        print(f"Re-visualizing results from {args.revisualize}...")
        with open(args.revisualize, 'r') as f:
            data = json.load(f)
        results = data["results"]
        figures_dir = Path(args.revisualize).parent / "figures"
        plot_results(results, figures_dir)
        return

    default_nu_counts = SUPPORTED_NU_COUNTS_BY_PROFILE.get(args.dataset_profile, DEFAULT_NU_COUNTS)
    nu_counts = list(args.nu) if args.nu else list(default_nu_counts)
    nu_counts = _filter_supported_nu_counts(nu_counts, args.dataset_profile)
    if not nu_counts:
        raise SystemExit(
            f"No TDBase-supported nu counts remain for dataset profile '{args.dataset_profile}'."
        )
    
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_nu_scalability")
    run_dir = Path(run_layout["run_dir"])
    run_log_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    isolated_data_dirs = create_isolated_run_data_dirs(run_dir)
    try:
        results = run_experiment(
            args.runs,
            args.grid_cell_size,
            nu_counts,
            run_log_dir,
            isolated_data_dirs,
            threads=args.threads,
            tdbase_threads=args.tdbase_threads,
            tdbase_compute_threads=args.tdbase_compute_threads,
            approaches=args.approaches,
            track_hash_contention=args.track_hash_contention,
            timeout=args.timeout,
            dataset_profile=args.dataset_profile,
            tdbase_timing_mode=args.tdbase_timing_mode,
        )
        
        if results and results["counts"]:
            print("\nResults Summary:")
            header = f"{'Nu':<10} {'Exact (ms)':<15} {'Selectivity Estimation (ms)':<15} {'Face (ms)':<15} {'TOUCH (ms)':<15} {'TDBase (ms)':<15}"
            print(header)
            print("-" * len(header))
            for i, n in enumerate(results["counts"]):
                ex = results['exact']['mean'][i]
                direct = results['direct_estimation']['mean'][i]
                cg = results['cgal']['mean'][i]
                to = results['touch']['mean'][i]
                td = results['tdbase']['mean'][i]
                
                ex_str = f"{ex:.2f}" if ex is not None else "N/A"
                direct_str = f"{direct:.2f}" if direct is not None else "N/A"
                cg_str = f"{cg:.2f}" if cg is not None else "N/A"
                to_str = f"{to:.2f}" if to is not None else "N/A"
                td_str = f"{td:.2f}" if td is not None else "N/A"
                
                print(f"{n:<10} {ex_str:<15} {direct_str:<15} {cg_str:<15} {to_str:<15} {td_str:<15}")
                    
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
                        "nu_counts": nu_counts,
                        "threads": args.threads,
                        "tdbase_threads": args.tdbase_threads,
                        "tdbase_compute_threads": args.tdbase_compute_threads,
                        "timeout": args.timeout,
                        "dataset_profile": args.dataset_profile,
                        "track_hash_contention": args.track_hash_contention,
                        "tdbase_timing_mode": args.tdbase_timing_mode,
                        "isolated_data_root": str(isolated_data_dirs["root"]),
                    },
                    "results": clean_results,
                },
            )
            print(f"Raw results saved to {out_json}")
        else:
            print("No successful runs.")
    finally:
        if isolated_data_dirs["root"].exists():
            print(f"Cleaning isolated preprocessing data: {isolated_data_dirs['root']}")
            shutil.rmtree(isolated_data_dirs["root"], ignore_errors=True)

if __name__ == "__main__":
    main()
