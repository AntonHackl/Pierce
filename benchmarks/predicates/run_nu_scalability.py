#!/usr/bin/env python3
import argparse
from pathlib import Path
import shutil
import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    canonical_nn_pair_paths,
    canonical_nu_pair_paths,
    create_isolated_run_data_dirs,
    create_benchmark_run_layout,
    ensure_nn_pair_dataset,
    ensure_nu_pair_dataset,
    get_shared_data_dirs,
    stage_input_files,
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


LEGACY_OVERLAP_RAW_DIR = REPO_ROOT / "benchmarks" / "overlap" / "data" / "raw"
FORCED_NU_COUNTS_BY_PROFILE = {
    "large_nu_v": [800],
    "large_nu_nn": [400],
}
OVERLAP_SHARED_SCENARIO = "large_nu_nn_scalability"
OVERLAP_NU_NV = 750
OVERLAP_NU_PREFIX = "tdbase_large"


def _forced_nu_counts_for_profile(dataset_profile: str) -> list[int]:
    return FORCED_NU_COUNTS_BY_PROFILE.get(
        dataset_profile,
        FORCED_NU_COUNTS_BY_PROFILE["large_nu_v"],
    ).copy()


def main():
    parser = argparse.ArgumentParser(description="Nu scalability benchmark for mesh query comparison")
    parser.add_argument("--nu", type=int, nargs="+", default=None)
    parser.add_argument(
        "--dataset-profile",
        type=str,
        default="large_nu_v",
        choices=["large_nu_v", "large_nu_nn"],
        help="Use Vessel-Nuclei (large_nu_v) or Nuclei-Nuclei (large_nu_nn) datasets.",
    )
    parser.add_argument("--grid-cell-size", type=float, default=200.0)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=120.0)

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
    forced_nu_counts = _forced_nu_counts_for_profile(args.dataset_profile)
    if args.nu != forced_nu_counts:
        if args.nu is not None:
            print(f"requested nu={args.nu}; forcing nu={forced_nu_counts} for TDBase-supported consistency")
        args.nu = forced_nu_counts

    queries = resolve_queries(args.queries, args.approaches)

    # Use the same shared dataset root as overlap overall performance (large_nu_v profile).
    shared_dirs = get_shared_data_dirs(OVERLAP_SHARED_SCENARIO)
    run_layout_name = (
        "query_comparison_nu_v_scalability"
        if args.dataset_profile == "large_nu_v"
        else "query_comparison_nu_scalability_nn"
    )
    run_layout = create_benchmark_run_layout(SCRIPT_DIR, run_layout_name)
    run_dir = Path(run_layout["run_dir"])
    logs_dir = Path(run_layout["logs_dir"])
    figures_dir = Path(run_layout["figures_dir"])
    isolated_data_dirs = create_isolated_run_data_dirs(run_dir)

    adapters = build_pierce_query_adapters(
        repo_root=REPO_ROOT,
        data_dirs=isolated_data_dirs,
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
    try:
        for nu in args.nu:
            if args.dataset_profile == "large_nu_nn":
                n1_path, n2_path = canonical_nn_pair_paths(
                    shared_dirs["raw"],
                    nu=nu,
                    nv=OVERLAP_NU_NV,
                    prefix=OVERLAP_NU_PREFIX,
                )
                ensure_nn_pair_dataset(n1_path, n2_path, legacy_raw_dirs=[LEGACY_OVERLAP_RAW_DIR])
                mesh1_path, mesh2_path = n1_path, n2_path
            else:
                n_path, v_path = canonical_nu_pair_paths(
                    shared_dirs["raw"],
                    nu=nu,
                    nv=OVERLAP_NU_NV,
                    prefix=OVERLAP_NU_PREFIX,
                )
                ensure_nu_pair_dataset(n_path, v_path, legacy_raw_dirs=[LEGACY_OVERLAP_RAW_DIR])
                mesh1_path, mesh2_path = v_path, n_path
            case_label = f"nu_{nu}"
            case_log_dir = logs_dir / sanitize_case_token(case_label)

            staged_mesh1_path, staged_mesh2_path = stage_input_files(
                [mesh1_path, mesh2_path],
                isolated_data_dirs["raw"],
            )
            ensure_preprocessed(adapters, [staged_mesh1_path, staged_mesh2_path], log_dir=case_log_dir)

            row = {
                "nu": nu,
                "dataset_profile": args.dataset_profile,
                "mesh1": str(mesh1_path),
                "mesh2": str(mesh2_path),
                "size_bytes1": mesh1_path.stat().st_size if mesh1_path.exists() else 0,
                "size_bytes2": mesh2_path.stat().st_size if mesh2_path.exists() else 0,
            }
            row.update(
                run_selected_queries(
                    adapters=adapters,
                    queries=queries,
                    mesh1=staged_mesh1_path,
                    mesh2=staged_mesh2_path,
                    runs=args.runs,
                    timeout=args.timeout,
                    overlap_query_direction=args.overlap_query_direction,
                    intersection_extra_args=intersection_extra_args,
                    log_dir=case_log_dir,
                )
            )

            results.append(row)
            case_labels.append(case_label)
            print(f"nu={nu}: done")

        generate_query_comparison_figures(
            results_rows=results,
            queries=queries,
            case_labels=case_labels,
            figures_dir=figures_dir,
            title_prefix="NU scalability",
            x_axis_label="Dataset case",
        )

        payload = {
            "metadata": {
                "scenario": "nu_scalability",
                "query_type": "predicate_comparison",
                "dataset_profile": args.dataset_profile,
                "timestamp": run_layout["timestamp"],
                "run_name": run_layout["run_name"],
                "run_dir": str(run_layout["run_dir"]),
                "nu": args.nu,
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
                "track_overflow": args.track_overflow,
                "enable_profiling_stats": args.enable_profiling_stats,
                "include_overlap_pairs": args.include_overlap_pairs,
                "shared_data_root": str(shared_dirs["root"]),
                "isolated_data_root": str(isolated_data_dirs["root"]),
            },
            "results": results,
        }

        out = Path(run_layout["results_json"])
        write_json(out, payload)
        print(f"Saved: {out}")
    finally:
        if isolated_data_dirs["root"].exists():
            print(f"Cleaning isolated preprocessing data: {isolated_data_dirs['root']}")
            shutil.rmtree(isolated_data_dirs["root"], ignore_errors=True)


if __name__ == "__main__":
    main()
