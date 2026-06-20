#!/usr/bin/env python3
from __future__ import annotations

import shutil
import argparse
import subprocess
import os
import sys
import json
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
EXPORT_DIR = SCRIPT_DIR / "latest_figures"


@dataclass(frozen=True)
class FigureSpec:
    output_name: str
    runs_dir: Path
    run_prefix: str
    source_pattern: str
    revisualize_script: Path | None = None
    required_dataset_profile: str | None = None


FIGURES = [
    FigureSpec(
        output_name="mesh_complexity_scalability.pdf",
        runs_dir=SCRIPT_DIR / "overlap" / "runs",
        run_prefix="overlap_mesh_complexity_",
        source_pattern="figures/mesh_complexity_scalability.pdf",
        revisualize_script=SCRIPT_DIR / "overlap" / "run_mesh_complexity_benchmark.py",
    ),
    FigureSpec(
        output_name="mesh_overlap_nu_scalability_scaling.pdf",
        runs_dir=SCRIPT_DIR / "overlap" / "runs",
        run_prefix="overlap_nu_scalability_",
        source_pattern="figures/mesh_overlap_nu_scalability_scaling.pdf",
        revisualize_script=SCRIPT_DIR / "overlap" / "run_nu_scalability.py",
        required_dataset_profile="large_nu_v",
    ),
    FigureSpec(
        output_name="mesh_overlap_overall_performance.pdf",
        runs_dir=SCRIPT_DIR / "overlap" / "runs",
        run_prefix="overlap_overall_performance_",
        source_pattern="figures/mesh_overlap_overall_performance_*.pdf",
        revisualize_script=SCRIPT_DIR / "overlap" / "plot_overall_performance.py",
    ),
    FigureSpec(
        output_name="mesh_query_comparison_overall_performance.pdf",
        runs_dir=SCRIPT_DIR / "predicates" / "runs",
        run_prefix="query_comparison_overall_performance_",
        source_pattern="figures/mesh_query_comparison_overall_performance_*.pdf",
        revisualize_script=SCRIPT_DIR / "predicates" / "plot_overall_performance.py",
    ),
    FigureSpec(
        output_name="overlap_selectivity_scaling.pdf",
        runs_dir=SCRIPT_DIR / "overlap" / "runs",
        run_prefix="overlap_selectivity_",
        source_pattern="figures/selectivity_scaling.pdf",
        revisualize_script=SCRIPT_DIR / "overlap" / "visualize_selectivity_test.py",
    ),
]


def newest_matching_files(spec: FigureSpec) -> tuple[Path, Path | None, Path | None]:
    candidate_runs = sorted(
        (
            path
            for path in spec.runs_dir.iterdir()
            if path.is_dir() and path.name.startswith(spec.run_prefix)
        ),
        reverse=True,
    )

    for run_dir in candidate_runs:
        results_json = run_dir / "results.json"
        if spec.required_dataset_profile is not None:
            if not results_json.exists():
                continue
            try:
                payload = json.loads(results_json.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            dataset_profile = (payload.get("metadata") or {}).get("dataset_profile")
            if dataset_profile != spec.required_dataset_profile:
                continue

        # For overall performance, the pattern matches a file in figures/
        matches = sorted(run_dir.glob(spec.source_pattern))
        if matches:
            pdf_path = matches[-1]
            # Try to find matching png
            png_path = pdf_path.with_suffix(".png")
            if not png_path.exists():
                # Try glob if filename differs by more than suffix (e.g. if source_pattern had a wildcard)
                png_matches = sorted(run_dir.glob(spec.source_pattern.replace(".pdf", ".png")))
                if png_matches:
                    png_path = png_matches[-1]
                else:
                    png_path = None
            
            actual_results = results_json if results_json.exists() else None
            return pdf_path, png_path, actual_results

    raise FileNotFoundError(
        f"No PDF matching {spec.source_pattern!r} found in runs with prefix "
        f"{spec.run_prefix!r} under {spec.runs_dir}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch latest benchmark figures.")
    parser.add_argument("--revisualize", action="store_true", help="Re-generate plots from results.json before fetching.")
    args = parser.parse_args()

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Set up environment for subprocesses
    env = os.environ.copy()
    repo_root = SCRIPT_DIR.parent
    if "PYTHONPATH" in env:
        env["PYTHONPATH"] = f"{repo_root}:{env['PYTHONPATH']}"
    else:
        env["PYTHONPATH"] = str(repo_root)

    if args.revisualize:
        print("Re-visualizing latest results...")
        seen_scripts = set()
        for spec in FIGURES:
            if spec.revisualize_script and spec.revisualize_script.exists():
                try:
                    pdf, png, results = newest_matching_files(spec)
                    
                    # Special case for plot_overall_performance which takes no args
                    if "plot_overall_performance.py" in spec.revisualize_script.name:
                        if spec.revisualize_script not in seen_scripts:
                            print(f"Running {spec.revisualize_script.name}...")
                            subprocess.run([sys.executable, str(spec.revisualize_script)], env=env, check=True)
                            seen_scripts.add(spec.revisualize_script)
                        continue

                    if not results:
                        print(f"Warning: No results.json found for {spec.output_name}, skipping revisualization.")
                        continue
                    
                    cmd = [sys.executable, str(spec.revisualize_script)]
                    if "visualize_selectivity_test.py" in spec.revisualize_script.name:
                        cmd.append(str(results))
                    else:
                        cmd.extend(["--revisualize", str(results)])
                    
                    print(f"Running: {' '.join(cmd)}")
                    subprocess.run(cmd, env=env, check=True)
                except Exception as e:
                    print(f"Error revisualizing {spec.output_name}: {e}")

    copied_count = 0
    for spec in FIGURES:
        try:
            source_pdf, source_png, _ = newest_matching_files(spec)
            
            # Copy PDF
            target_pdf = EXPORT_DIR / spec.output_name
            shutil.copy2(source_pdf, target_pdf)
            print(f"{target_pdf.name} <- {source_pdf}")
            copied_count += 1
            
            # Copy PNG if exists
            if source_png and source_png.exists():
                target_png = EXPORT_DIR / f"_{spec.output_name.replace('.pdf', '.png')}"
                shutil.copy2(source_png, target_png)
                print(f"{target_png.name} <- {source_png}")
                copied_count += 1
        except FileNotFoundError as e:
            print(f"Skipping {spec.output_name}: {e}")
    # Also copy the latest dataset_table.tex from the dataset benchmark run
    try:
        dataset_runs_dir = SCRIPT_DIR / "datasets" / "runs"
        if dataset_runs_dir.exists():
            candidate_runs = sorted(
                (
                    path
                    for path in dataset_runs_dir.iterdir()
                    if path.is_dir() and path.name.startswith("dataset_table_benchmark_")
                ),
                reverse=True,
            )

            for run_dir in candidate_runs:
                table_path = run_dir / "dataset_table.tex"
                if table_path.exists():
                    target_table = EXPORT_DIR / "dataset_table.tex"
                    shutil.copy2(table_path, target_table)
                    print(f"{target_table.name} <- {table_path}")
                    copied_count += 1
                    break
    except Exception as e:
        print(f"Skipping dataset_table.tex: {e}")

    print(f"Copied {copied_count} files to {EXPORT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
