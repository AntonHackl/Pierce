#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import (
    PIERCE_DIR,
    canonical_cube_pair_paths,
    canonical_microns_aggregated_paths,
    canonical_nn_pair_paths,
    canonical_nu_pair_paths,
    canonical_sphere_pair_paths,
    create_benchmark_run_layout,
    ensure_cube_pair_dataset,
    ensure_sphere_pair_dataset,
    get_shared_data_dirs,
    write_json,
)
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    TDBASE_TIMING_MODES,
)
from benchmarks.overlap.adapters.pierce_adapter import PierceAdapter
from benchmarks.overlap.adapters.tdbase_adapter import TDBaseAdapter
from benchmarks.overlap.adapters.base import run_command_streaming

TDBASE_DIR = REPO_ROOT / "baselines" / "tdbase_extensions"
SLURM_LOGS_DIR = REPO_ROOT / ".slurm" / "logs"

_TDBASE_GENERATION_FILENAME_RE = re.compile(
    r"^(?P<prefix>.+)_(?P<kind>n|nn|nu)_nv(?P<nv>\d+)_nu(?P<nu>\d+)_(?P<dataset_kind>n|v)_nv\d+_nu\d+_vs\d+_r\d+\.dt$"
)
_TDBASE_LARGE_HEADER_RE = re.compile(
    r"Generating LARGE (?P<kind>n|nn|nu) dataset with prefix=(?P<prefix>[^,]+), nv=(?P<nv>\d+), nu=(?P<nu>\d+)\.\.\."
)
_TDBASE_PREPROCESS_TIMING_RE = re.compile(
    r"preprocess_timing_ms .* preprocessing_only=([0-9.]+)"
)
_TDBASE_PREPROCESS_TIMING_OUTPUT_RE = re.compile(
    r"preprocess_timing_ms dataset_kind=(?P<dataset_kind>n|v) output=(?P<output>\S+) .* preprocessing_only=(?P<preprocess_ms>[0-9.]+)"
)
_TDBASE_TILE_LOAD_TIMING_RE = re.compile(
    r"loaded\s+\d+\s+polyhedra\s+in\s+tile\s+.*?\s+takes\s+([0-9.]+)\s+(s|ms)"
)


@dataclass(frozen=True)
class DatasetRow:
    dataset_id: str
    source_path: Path
    grid_cell_size: float


TDBASE_ELIGIBLE_DATASETS = {"Nuclei_1", "Vessel", "Nuclei_2", "Nuclei_3", "Spheres_1", "Spheres_2"}
TDBASE_REQUIRED_LOADING_DATASETS = {"Nuclei_1", "Vessel", "Nuclei_2", "Nuclei_3"}
SPHERE_TEMPLATE_DIR = REPO_ROOT / "benchmarks" / "overlap" / "data" / "single_obj_files"
SPHERE_DEFAULT_STAGE = 5
SPHERE_DEFAULT_NUM_OBJECTS = 500
SPHERE_DEFAULT_SELECTIVITY = 0.0005
SPHERE_DEFAULT_GRID_CELL_SIZE = 5.0
DEFAULT_TISSUE_NU_V = 800
DEFAULT_TISSUE_NU_NN = 400
DEFAULT_TDBASE_LOADING_SCENARIO = "large_nu_nn_scalability"


def _ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _human_bytes(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(num_bytes)
    for unit in units:
        if value < 1024.0 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _grid_token(grid_cell_size: float) -> str:
    return str(grid_cell_size).replace(".", "_")


def _run_preprocess(
    *,
    dataset: DatasetRow,
    preprocessed_dir: Path,
    timings_dir: Path,
    logs_dir: Path,
) -> Dict[str, float | int | str]:
    _log(f"[preprocess:{dataset.dataset_id}] START source={dataset.source_path} grid={dataset.grid_cell_size}")
    preprocess_exec = PIERCE_DIR / "preprocess" / "build" / "bin" / "pierce_preprocess"
    if not preprocess_exec.exists():
        raise FileNotFoundError(f"Preprocess executable not found: {preprocess_exec}")

    mode = "dt" if dataset.source_path.suffix == ".dt" else "mesh"
    pre_path = preprocessed_dir / f"{dataset.source_path.stem}_g{_grid_token(dataset.grid_cell_size)}.pre"
    timing_path = timings_dir / f"{dataset.source_path.stem}_g{_grid_token(dataset.grid_cell_size)}_timing.json"
    log_path = logs_dir / f"preprocess_{dataset.dataset_id}.log"

    cmd = [
        str(preprocess_exec),
        "--mode", mode,
        "--dataset", str(dataset.source_path),
        "--output-geometry", str(pre_path),
        "--output-timing", str(timing_path),
        "--generate-grid",
        "--grid-cell-size", str(dataset.grid_cell_size),
    ]
    _log(f"[preprocess:{dataset.dataset_id}] command: {' '.join(cmd)}")

    t0 = time.time()
    try:
        stdout_text, stderr_text = run_command_streaming(
            cmd,
            timeout=None,
            log_path=str(log_path),
            prefix=f"[preprocess:{dataset.dataset_id}]",
        )
    except subprocess.CalledProcessError as exc:
        elapsed = time.time() - t0
        combined_err = (exc.output or "") + (exc.stderr or "")
        _log(
            f"[preprocess:{dataset.dataset_id}] failed return_code={exc.returncode} "
            f"elapsed={elapsed:.2f}s log={log_path}"
        )
        raise RuntimeError(f"Preprocess failed for {dataset.dataset_id}: {combined_err}") from exc

    elapsed = time.time() - t0
    combined = (stdout_text or "") + (stderr_text or "")
    _log(
        f"[preprocess:{dataset.dataset_id}] finished return_code=0 "
        f"elapsed={elapsed:.2f}s log={log_path}"
    )

    objects = None
    triangles = None

    m_obj = re.search(r"Loaded tile with\s+(\d+)\s+objects\.", combined)
    if m_obj:
        objects = int(m_obj.group(1))
    else:
        m_obj2 = re.search(r"Successfully loaded\s+(\d+)\s+object\(s\)", combined)
        if m_obj2:
            objects = int(m_obj2.group(1))

    m_tri = re.search(r"Total triangles:\s*(\d+)", combined)
    if m_tri:
        triangles = int(m_tri.group(1))

    if objects is None:
        raise RuntimeError(f"Could not parse object count for {dataset.dataset_id}")
    if triangles is None:
        raise RuntimeError(f"Could not parse triangle count for {dataset.dataset_id}")

    if not timing_path.exists():
        raise RuntimeError(f"Timing json missing for {dataset.dataset_id}: {timing_path}")

    timing_payload = json.loads(timing_path.read_text(encoding="utf-8"))
    preprocess_ms = float(timing_payload.get("total", {}).get("duration_ms", 0.0))
    _log(
        f"[preprocess:{dataset.dataset_id}] parsed objects={objects} triangles={triangles} "
        f"preprocess_ms={preprocess_ms:.2f}"
    )

    return {
        "objects": objects,
        "triangles": triangles,
        "preprocess_ms": preprocess_ms,
        "preprocessed_path": str(pre_path),
        "timing_path": str(timing_path),
        "log_path": str(log_path),
    }


def _run_join_and_parse_loading(
    *,
    case_name: str,
    mesh1: Path,
    mesh2: Path,
    grid_cell_size: float,
    preprocessed_dir: Path,
    timings_dir: Path,
    logs_dir: Path,
) -> Tuple[float, float, str]:
    _log(f"[join:{case_name}] START mesh1={mesh1.name} mesh2={mesh2.name} grid={grid_cell_size}")
    case_log_dir = logs_dir / case_name
    case_log_dir.mkdir(parents=True, exist_ok=True)

    attempted_errors: List[str] = []
    for mode in ["direct_estimation", "exact", "estimated"]:
        _log(f"[join:{case_name}] trying mode={mode}")
        adapter = PierceAdapter(
            str(PIERCE_DIR),
            mode=mode,
            preprocessed_dir=str(preprocessed_dir),
            timings_dir=str(timings_dir),
            grid_cell_size=grid_cell_size,
            warmup_runs=1,
        )

        t0 = time.time()
        result = adapter.run_overlap(
            str(mesh1),
            str(mesh2),
            num_runs=1,
            timeout=1200.0,
            log_dir=str(case_log_dir),
            query_direction="both",
        )
        elapsed = time.time() - t0
        _log(f"[join:{case_name}] mode={mode} finished elapsed={elapsed:.2f}s")
        if "error" in result:
            attempted_errors.append(f"{mode}: {result['error']}")
            _log(f"[join:{case_name}] mode={mode} failed: {result['error']}")
            continue

        run_log = case_log_dir / f"Pierce_{mode}" / "run_000.log"
        if not run_log.exists():
            attempted_errors.append(f"{mode}: missing log {run_log}")
            _log(f"[join:{case_name}] mode={mode} failed: missing log {run_log}")
            continue

        text = run_log.read_text(encoding="utf-8", errors="replace")
        m1 = re.search(r"Upload Mesh1:\s+\d+\s+microseconds\s+\(([0-9.]+)\s+ms\)", text)
        m2 = re.search(r"Upload Mesh2:\s+\d+\s+microseconds\s+\(([0-9.]+)\s+ms\)", text)
        if not m1 or not m2:
            attempted_errors.append(f"{mode}: could not parse upload timings from {run_log}")
            _log(f"[join:{case_name}] mode={mode} failed: upload timing parse failed")
            continue

        _log(
            f"[join:{case_name}] SUCCESS mode={mode} upload_mesh1_ms={float(m1.group(1)):.2f} "
            f"upload_mesh2_ms={float(m2.group(1)):.2f} log={run_log}"
        )
        return float(m1.group(1)), float(m2.group(1)), str(run_log)

    raise RuntimeError(f"Join failed for {case_name}. Attempts: {' | '.join(attempted_errors)}")


def _tdbase_preprocessed_path(preprocessed_dir: Path, source_path: Path) -> Path:
    return preprocessed_dir / source_path.with_suffix(".dt").name


def _parse_tdbase_generation_spec(source_path: Path) -> Dict[str, object] | None:
    match = _TDBASE_GENERATION_FILENAME_RE.match(source_path.name)
    if not match:
        return None
    kind = match.group("kind")
    if kind == "nu":
        kind = "n"
    return {
        "prefix": match.group("prefix"),
        "kind": kind,
        "dataset_kind": match.group("dataset_kind"),
        "nv": int(match.group("nv")),
        "nu": int(match.group("nu")),
    }


def _find_tdbase_generation_timing(
    source_path: Path,
    *,
    log_glob: str = "generate_large_nu_nn_*.out",
) -> Dict[str, object] | None:
    spec = _parse_tdbase_generation_spec(source_path)
    if spec is None:
        return None

    candidates = sorted(SLURM_LOGS_DIR.glob(log_glob), reverse=True)
    for out_path in candidates:
        err_path = out_path.with_suffix(".err")
        if not err_path.exists():
            continue

        out_lines = out_path.read_text(encoding="utf-8", errors="replace").splitlines()
        err_text = err_path.read_text(encoding="utf-8", errors="replace")

        output_matches = list(_TDBASE_PREPROCESS_TIMING_OUTPUT_RE.finditer(err_text))
        if output_matches:
            for match in output_matches:
                output_name = Path(match.group("output")).name
                if output_name != source_path.name:
                    continue
                return {
                    "preprocess_ms": float(match.group("preprocess_ms")),
                    "preprocess_source": "simulator_preprocessing_only_from_slurm_log",
                    "log_path": str(err_path),
                }

        stage_specs: List[Dict[str, object]] = []
        for line in out_lines:
            match = _TDBASE_LARGE_HEADER_RE.search(line)
            if not match:
                continue
            kind = match.group("kind")
            if kind == "nu":
                kind = "n"
            stage_specs.append(
                {
                    "prefix": match.group("prefix"),
                    "kind": kind,
                    "dataset_kind": spec["dataset_kind"],
                    "nv": int(match.group("nv")),
                    "nu": int(match.group("nu")),
                }
            )

        if not stage_specs:
            continue

        preprocess_matches = [float(m.group(1)) for m in _TDBASE_PREPROCESS_TIMING_RE.finditer(err_text)]
        if preprocess_matches:
            for stage_spec, preprocess_ms in zip(stage_specs, preprocess_matches):
                if stage_spec == spec:
                    return {
                        "preprocess_ms": preprocess_ms,
                        "preprocess_source": "simulator_preprocessing_only_from_slurm_log",
                        "log_path": str(err_path),
                    }

    return None


def _run_tdbase_preprocess(
    *,
    dataset: DatasetRow,
    adapter: TDBaseAdapter,
    logs_dir: Path,
    generation_log_glob: str = "generate_large_nu_nn_*.out",
) -> Dict[str, float | str | None]:
    _log(f"[tdbase-preprocess:{dataset.dataset_id}] START source={dataset.source_path}")
    timing_info = (
        _find_tdbase_generation_timing(dataset.source_path, log_glob=generation_log_glob)
        if dataset.source_path.suffix == ".dt"
        else None
    )
    t0 = time.time()
    adapter.preprocess_from_source(str(dataset.source_path), str(dataset.source_path), log_dir=str(logs_dir))
    copy_elapsed_ms = (time.time() - t0) * 1000.0
    dt_path = _tdbase_preprocessed_path(adapter.preprocessed_dir, dataset.source_path)
    if not dt_path.exists():
        raise RuntimeError(f"TDBase preprocessed dataset missing for {dataset.dataset_id}: {dt_path}")

    preprocess_ms: float | None = copy_elapsed_ms
    preprocess_source: str | None = "obj_to_dt_wall_clock"
    preprocess_log_path = None
    if timing_info is not None:
        preprocess_ms = float(timing_info["preprocess_ms"])
        preprocess_source = str(timing_info["preprocess_source"])
        preprocess_log_path = str(timing_info["log_path"])
    elif dataset.source_path.suffix == ".dt":
        preprocess_ms = None
        preprocess_source = None

    preprocess_label = f"{preprocess_ms:.2f}" if preprocess_ms is not None else "missing"
    _log(
        f"[tdbase-preprocess:{dataset.dataset_id}] finished preprocess_ms={preprocess_label} "
        f"copy_elapsed_ms={copy_elapsed_ms:.2f} source={preprocess_source} output={dt_path}"
    )
    return {
        "preprocess_ms": preprocess_ms,
        "copy_elapsed_ms": copy_elapsed_ms,
        "preprocess_source": preprocess_source,
        "preprocess_log_path": preprocess_log_path,
        "preprocessed_path": str(dt_path),
    }


def _run_tdbase_and_parse_loading(
    *,
    case_name: str,
    mesh1: Path,
    mesh2: Path,
    dataset_id_1: str,
    dataset_id_2: str,
    adapter: TDBaseAdapter,
    logs_dir: Path,
) -> Tuple[Dict[str, float | str], Dict[str, float | str]]:
    _log(
        f"[tdbase-load:{case_name}] START mesh1={mesh1.name} ({dataset_id_1}) "
        f"mesh2={mesh2.name} ({dataset_id_2})"
    )
    case_log_dir = logs_dir / case_name
    with tempfile.TemporaryDirectory(prefix="rs_tdb_", dir=tempfile.gettempdir()) as short_dir_str:
        short_dir = Path(short_dir_str)
        staged_mesh1 = short_dir / "a.dt"
        staged_mesh2 = short_dir / "b.dt"
        shutil.copyfile(mesh1, staged_mesh1)
        shutil.copyfile(mesh2, staged_mesh2)
        _log(
            f"[tdbase-load:{case_name}] staged inputs to short paths "
            f"mesh1={staged_mesh1} mesh2={staged_mesh2}"
        )
        result = adapter.run_overlap(
            str(staged_mesh1),
            str(staged_mesh2),
            num_runs=1,
            timeout=1200.0,
            log_dir=str(case_log_dir),
        )
        if "error" in result:
            raise RuntimeError(f"TDBase run failed for {case_name}: {result['error']}")

        if not result.get("run_metrics"):
            raise RuntimeError(f"TDBase metrics missing for {case_name}")

        metrics = result["run_metrics"][0]
        attempt = str(metrics.get("attempt", "primary"))
        log_suffix = "" if attempt == "primary" else f"_{attempt}"
        run_log = case_log_dir / adapter.name / f"run_000{log_suffix}.log"
        if not run_log.exists():
            raise RuntimeError(f"TDBase run log missing for {case_name}: {run_log}")

        text = run_log.read_text(encoding="utf-8", errors="replace")
        command_line = next((line for line in text.splitlines() if line and not line.startswith("COMMAND:")), "")
        if str(staged_mesh1) not in command_line or str(staged_mesh2) not in command_line:
            raise RuntimeError(
                "TDBase loading run did not use the expected staged input datasets. "
                f"Expected mesh1={staged_mesh1} mesh2={staged_mesh2}; command={command_line}"
            )
        load_matches = list(_TDBASE_TILE_LOAD_TIMING_RE.finditer(text))
        if len(load_matches) < 2:
            raise RuntimeError(
                f"Could not parse two per-tile TDBase load timings for {case_name} from {run_log}"
            )

    def _match_to_ms(match: re.Match[str]) -> float:
        value_ms = float(match.group(1))
        if match.group(2) == "s":
            value_ms *= 1000.0
        return value_ms

    load1_ms = _match_to_ms(load_matches[0])
    load2_ms = _match_to_ms(load_matches[1])
    total_ms = float(metrics["total_ms"])
    load_tiles_ms = float(metrics["load_tiles_ms"])
    index_ms = float(metrics["index_ms"])
    decode_ms = float(metrics["decode_ms"])
    prepare_ms = float(metrics["prepare_ms"])
    compute_ms = float(metrics["compute_ms"])
    evaluate_ms = float(metrics["evaluate_ms"])
    other_ms = float(metrics["other_ms"])
    total_loading_ms = float(metrics["loading_ms"])
    _log(
        f"[tdbase-load:{case_name}] finished total_ms={total_ms:.2f} "
        f"{dataset_id_1}_load_ms={load1_ms:.2f} {dataset_id_2}_load_ms={load2_ms:.2f} "
        f"load_tiles_ms={load_tiles_ms:.2f} index_ms={index_ms:.2f} "
        f"decode_ms={decode_ms:.2f} prepare_ms={prepare_ms:.2f} "
        f"evaluate_ms={evaluate_ms:.2f} compute_ms={compute_ms:.2f} other_ms={other_ms:.2f} "
        f"total_loading_ms={total_loading_ms:.2f} attempt={attempt} log={run_log}"
    )
    shared_stats: Dict[str, float | str] = {
        "total_ms": total_ms,
        "load_tiles_ms": load_tiles_ms,
        "index_ms": index_ms,
        "decode_ms": decode_ms,
        "prepare_ms": prepare_ms,
        "evaluate_ms": evaluate_ms,
        "compute_ms": compute_ms,
        "other_ms": other_ms,
        "total_loading_ms": total_loading_ms,
        "run_log": str(run_log),
        "mean_query_preprocessing_ms": float(result.get("mean_preprocessing", 0.0)),
        "attempt": attempt,
        "case_name": case_name,
    }
    return (
        {
            **shared_stats,
            "loading_ms": load1_ms,
            "paired_dataset_id": dataset_id_2,
        },
        {
            **shared_stats,
            "loading_ms": load2_ms,
            "paired_dataset_id": dataset_id_1,
        },
    )


def _build_latex_table(rows: List[Dict[str, object]]) -> str:
    rows_by_id = {str(r["dataset_id"]): r for r in rows}

    def _fmt_int(value: object) -> str:
        return f"{int(value):,}".replace(",", "{,}")

    def _fmt_triangles(value: object) -> str:
        v = int(value)
        if v >= 1_000_000:
            return f"{(v / 1_000_000.0):.1f}M"
        return _fmt_int(v)

    def _fmt_size(size_bytes: object) -> str:
        b = int(size_bytes)
        if b >= 1024**3:
            s = f"{(b / (1024**3)):.2f}".rstrip("0").rstrip(".")
            return f"{s} GB"
        if b >= 1024**2:
            s = f"{(b / (1024**2)):.0f}"
            return f"{s} MB"
        if b >= 1024:
            s = f"{(b / 1024):.0f}"
            return f"{s} KB"
        return f"{b} B"

    def _fmt_pair(left: object, right: object, *, left_scale: float = 1.0, right_scale: float = 1.0) -> str:
        left_s = "--" if left is None else f"{(float(left) / left_scale):.2f}"
        right_s = "--" if right is None else f"{(float(right) / right_scale):.2f}"
        return f"{left_s} / {right_s}"

    def _row(dataset_id: str, label: str, description: str) -> str:
        row = rows_by_id[dataset_id]
        return (
            f"\\quad {label} & {description} & "
            f"{_fmt_int(row['objects'])} & {_fmt_triangles(row['triangles'])} & {_fmt_size(row['dataset_size_bytes'])} & "
            f"{_fmt_pair(row.get('preprocess_pierce_ms'), row.get('preprocess_tdbase_ms'), left_scale=1000.0, right_scale=1000.0)} & "
            f"{_fmt_pair(row.get('loading_pierce_ms'), row.get('loading_tdbase_ms'))} \\\\"
        )

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\scriptsize",
        r"\setlength{\tabcolsep}{4pt}",
        r"\begin{tabular}{lp{5cm}rrrrr}",
        r"\toprule",
        r"Dataset & Description & \#Objects & \#Triangles & Size & Preproc.\ (s) [Pierce/TDBase] & Loading (ms) [Pierce/TDBase] \\",
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{\textsc{Tissue} }} \\",
        _row("Nuclei_1", r"Nuclei$_1$", r"Cell nuclei mesh (TDBase generator, \url{https://github.com/tengdj/tdbase})"),
        _row("Vessel", r"Vessel", r"Blood vessel mesh (TDBase generator, \url{https://github.com/tengdj/tdbase})"),
        _row("Nuclei_2", r"Nuclei$_2$", r"Cell nuclei mesh (TDBase generator, \url{https://github.com/tengdj/tdbase})"),
        _row("Nuclei_3", r"Nuclei$_3$", r"Independent nuclei regeneration (TDBase generator, \url{https://github.com/tengdj/tdbase})"),
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{\textsc{MICrONS}}} \\",
        _row("Neurons_1", r"Neurons$_1$", r"Neuron meshes (regional subset, \url{https://www.microns-explorer.org/})"),
        _row("Neurons_2", r"Neurons$_2$", r"Neuron meshes (regional subset, \url{https://www.microns-explorer.org/})"),
        _row("Neurons_3", r"Neurons$_3$", r"Neuron meshes (regional subset, \url{https://www.microns-explorer.org/})"),
        _row("Neurons_4", r"Neurons$_4$", r"Neuron meshes (regional subset, \url{https://www.microns-explorer.org/})"),
        r"\midrule",
        r"\multicolumn{7}{l}{\textit{\textsc{Synthetic}}} \\",
        _row("Cubes_1", r"Cubes$_1$", r"Uniformly sampled axis-aligned cubes (generate\_cubes\_by\_selectivity.py)"),
        _row("Cubes_2", r"Cubes$_2$", r"Uniformly sampled axis-aligned cubes (generate\_cubes\_by\_selectivity.py)"),
        _row("Spheres_1", r"Spheres$_1$", r"Tessellated spheres, mesh complexity benchmark script"),
        _row("Spheres_2", r"Spheres$_2$", r"Tessellated spheres, mesh complexity benchmark script"),
        r"\bottomrule",
        r"\end{tabular}",
        r"\caption{Datasets used in our evaluation, with preprocessing and loading times. A dash indicates that TDBase does not support the dataset.}",
        r"\label{tab:dataset_benchmark}",
        r"\end{table*}",
    ]
    return "\n".join(lines)


def _resolve_microns_pair_paths(size_gb: int) -> Tuple[Path, Path]:
    scenario_candidates = [
        "microns_overlap",
        "microns_query_comparison",
        "microns_intersection_estimated",
    ]
    tried: List[Tuple[str, Path, Path]] = []
    for scenario in scenario_candidates:
        dirs = get_shared_data_dirs(scenario)
        a_path, b_path = canonical_microns_aggregated_paths(dirs["raw"], size_gb)
        tried.append((scenario, a_path, b_path))
        if a_path.exists() and b_path.exists():
            _log(
                f"resolved MICrONS {size_gb}GB from scenario={scenario}: "
                f"{a_path.name}, {b_path.name}"
            )
            return a_path, b_path

    tried_str = "; ".join(
        f"{scenario}: {a_path} | {b_path}"
        for scenario, a_path, b_path in tried
    )
    raise FileNotFoundError(
        f"MICrONS {size_gb}GB split files not found in any known shared-data scenario. Tried: {tried_str}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Dataset benchmark table generator for overall overlap benchmark datasets")
    parser.add_argument(
        "--nu-v",
        type=int,
        default=DEFAULT_TISSUE_NU_V,
        help="NU count used for the current large vessel-nuclei overall-performance point",
    )
    parser.add_argument(
        "--nu-nn",
        type=int,
        default=DEFAULT_TISSUE_NU_NN,
        help="NU count used for the current large nuclei-nuclei overall-performance point",
    )
    parser.add_argument("--microns-size-gb", type=int, default=4, help="MICrONS size used in overall benchmark point")
    parser.add_argument("--large-microns-size-gb", type=int, default=8, help="MICrONS large size used for Neurons 3 and 4")
    parser.add_argument("--cube-count-b", type=int, default=1000000, help="Cubes count for dataset B used in overall benchmark point")
    parser.add_argument(
        "--tdbase-large-scenario",
        type=str,
        default="large_nu_nn_scalability",
        help="Shared data scenario used for the large nuclei/vessel TDBase datasets",
    )
    parser.add_argument(
        "--tdbase-loading-scenario",
        type=str,
        default=DEFAULT_TDBASE_LOADING_SCENARIO,
        help=(
            "Shared data scenario used specifically for TDBase loading-time extraction. "
            "Defaults to the same shared dataset root used by the overlap overall-performance benchmark."
        ),
    )
    parser.add_argument(
        "--tdbase-generation-log-glob",
        type=str,
        default="generate_large_nu_nn_*.out",
        help="Glob under benchmarks/slurm_logs used to recover simulator preprocessing timings",
    )
    parser.add_argument(
        "--tdbase-timing-mode",
        type=str,
        default=TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
        choices=TDBASE_TIMING_MODES,
        help="TDBase query-time definition for overlap runs. Loading-column extraction is unaffected.",
    )
    parser.add_argument(
        "--require-tdbase-loading",
        action="store_true",
        help="Fail the benchmark if TDBase loading extraction is missing for any tissue dataset.",
    )
    parser.add_argument(
        "--tdbase-loading-threads",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "0")) or None,
        help=(
            "Number of TDBase join threads for loading extraction. "
            "Defaults to SLURM_CPUS_PER_TASK when available."
        ),
    )
    args = parser.parse_args()
    _log(
        f"dataset benchmark start nu_v={args.nu_v} nu_nn={args.nu_nn} microns_size_gb={args.microns_size_gb} "
        f"cube_count_b={args.cube_count_b} tdbase_large_scenario={args.tdbase_large_scenario} "
        f"tdbase_loading_scenario={args.tdbase_loading_scenario}"
    )

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "dataset_table_benchmark")
    logs_dir = Path(run_layout["logs_dir"])
    _log(f"run_dir={run_layout['run_dir']}")
    _log(f"logs_dir={logs_dir}")

    dataset_dirs = get_shared_data_dirs("dataset_table_benchmark")
    _log(f"shared dataset dirs: {dataset_dirs}")

    nu_dirs = get_shared_data_dirs(args.tdbase_large_scenario)
    tdbase_loading_dirs = get_shared_data_dirs(args.tdbase_loading_scenario)
    cube_dirs = get_shared_data_dirs("cube_scalability")

    nuclei_800_path, vessel_800_path = canonical_nu_pair_paths(
        nu_dirs["raw"], nu=args.nu_v, nv=750, prefix="tdbase_large"
    )
    nuclei_400_1_path, nuclei_400_2_path = canonical_nn_pair_paths(
        nu_dirs["raw"], nu=args.nu_nn, nv=750, prefix="tdbase_large"
    )
    loading_nuclei_800_path, loading_vessel_800_path = canonical_nu_pair_paths(
        tdbase_loading_dirs["raw"], nu=args.nu_v, nv=750, prefix="tdbase_large"
    )
    loading_nuclei_400_1_path, loading_nuclei_400_2_path = canonical_nn_pair_paths(
        tdbase_loading_dirs["raw"], nu=args.nu_nn, nv=750, prefix="tdbase_large"
    )
    neurons_1, neurons_2 = _resolve_microns_pair_paths(args.microns_size_gb)
    neurons_3, neurons_4 = _resolve_microns_pair_paths(args.large_microns_size_gb)
    cubes_1, cubes_2 = canonical_cube_pair_paths(
        cube_dirs["raw"],
        num_cubes_a=200000,
        num_cubes_b=args.cube_count_b,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
        grid_cell_size=5.0,
    )
    spheres_dirs = get_shared_data_dirs("mesh_complexity")
    sphere_template_name = f"Sphere_Stage_{SPHERE_DEFAULT_STAGE}.obj"
    sphere_template_path = SPHERE_TEMPLATE_DIR / sphere_template_name
    if not sphere_template_path.exists():
        raise FileNotFoundError(f"Sphere template not found: {sphere_template_path}")
    spheres_1, spheres_2 = canonical_sphere_pair_paths(
        spheres_dirs["raw"],
        template_name=sphere_template_name,
        num_objects=SPHERE_DEFAULT_NUM_OBJECTS,
        min_size=1.0,
        max_size=5.0,
        selectivity=SPHERE_DEFAULT_SELECTIVITY,
        seed=42,
        grid_cell_size=SPHERE_DEFAULT_GRID_CELL_SIZE,
    )

    ensure_cube_pair_dataset(
        cubes_1,
        cubes_2,
        num_cubes_a=200000,
        num_cubes_b=args.cube_count_b,
        min_size=1.0,
        max_size=2.0,
        selectivity=0.001,
        seed=42,
    )
    _log(f"cube files ensured: {cubes_1} | {cubes_2}")
    ensure_sphere_pair_dataset(
        spheres_1,
        spheres_2,
        template_obj=sphere_template_path,
        num_objects=SPHERE_DEFAULT_NUM_OBJECTS,
        min_size=1.0,
        max_size=5.0,
        selectivity=SPHERE_DEFAULT_SELECTIVITY,
        seed=42,
    )
    _log(f"sphere files ensured (largest stage): {spheres_1} | {spheres_2}")

    datasets: List[DatasetRow] = [
        DatasetRow("Nuclei_1", nuclei_800_path, 200.0),
        DatasetRow("Vessel", vessel_800_path, 200.0),
        DatasetRow("Nuclei_2", nuclei_400_1_path, 200.0),
        DatasetRow("Nuclei_3", nuclei_400_2_path, 200.0),
        DatasetRow("Neurons_1", neurons_1, 700.0),
        DatasetRow("Neurons_2", neurons_2, 700.0),
        DatasetRow("Neurons_3", neurons_3, 700.0),
        DatasetRow("Neurons_4", neurons_4, 700.0),
        DatasetRow("Cubes_1", cubes_1, 5.0),
        DatasetRow("Cubes_2", cubes_2, 5.0),
        DatasetRow("Spheres_1", spheres_1, 5.0),
        DatasetRow("Spheres_2", spheres_2, 5.0),
    ]

    for ds in datasets:
        if not ds.source_path.exists():
            raise FileNotFoundError(f"Dataset file not found for {ds.dataset_id}: {ds.source_path}")
        _log(f"dataset present: {ds.dataset_id} -> {ds.source_path}")

    tdbase_loading_sources = {
        "Vessel": loading_vessel_800_path,
        "Nuclei_1": loading_nuclei_800_path,
        "Nuclei_2": loading_nuclei_400_1_path,
        "Nuclei_3": loading_nuclei_400_2_path,
    }
    for dataset_id, source_path in tdbase_loading_sources.items():
        if not source_path.exists():
            raise FileNotFoundError(
                f"TDBase loading dataset not found for {dataset_id} in scenario "
                f"{args.tdbase_loading_scenario}: {source_path}"
            )
        _log(
            f"tdbase loading dataset: {dataset_id} -> {source_path} "
            f"(scenario={args.tdbase_loading_scenario})"
        )
    preprocess_stats: Dict[str, Dict[str, float | int | str]] = {}
    _log("=== Stage: preprocessing all datasets ===")
    for ds in datasets:
        preprocess_stats[ds.dataset_id] = _run_preprocess(
            dataset=ds,
            preprocessed_dir=dataset_dirs["preprocessed"],
            timings_dir=dataset_dirs["timings"],
            logs_dir=logs_dir,
        )
    _log("=== Stage complete: preprocessing ===")

    tdbase_preprocess_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=str(dataset_dirs["preprocessed"]),
        query_timing_mode=args.tdbase_timing_mode,
    )
    # Use the exact shared raw .dt paths selected above. This matches the
    # successful historical overlap log command line for the large tissue point.
    tdbase_loading_adapter = TDBaseAdapter(
        str(TDBASE_DIR),
        preprocessed_dir=None,
        threads=args.tdbase_loading_threads,
        query_timing_mode=args.tdbase_timing_mode,
    )
    tdbase_preprocess_stats: Dict[str, Dict[str, float | str]] = {}
    tdbase_loading_stats: Dict[str, Dict[str, float | str]] = {}
    tdbase_preprocess_errors: Dict[str, str] = {}
    tdbase_loading_errors: Dict[str, str] = {}

    _log("=== Stage: TDBase preprocessing for eligible datasets ===")
    for ds in datasets:
        if ds.dataset_id not in TDBASE_ELIGIBLE_DATASETS:
            continue
        try:
            tdbase_preprocess_stats[ds.dataset_id] = _run_tdbase_preprocess(
                dataset=ds,
                adapter=tdbase_preprocess_adapter,
                logs_dir=logs_dir,
                generation_log_glob=args.tdbase_generation_log_glob,
            )
        except Exception as exc:
            err = str(exc)
            tdbase_preprocess_errors[ds.dataset_id] = err
            _log(f"[tdbase-preprocess:{ds.dataset_id}] FAILED: {err}")
    _log("=== Stage complete: TDBase preprocessing ===")

    _log("=== Stage: TDBase loading extraction from real join pairs ===")
    for case_name, mesh1, dataset_id_1, mesh2, dataset_id_2 in [
        (
            "tdbase_vessel_nuclei_1",
            loading_vessel_800_path,
            "Vessel",
            loading_nuclei_800_path,
            "Nuclei_1",
        ),
        (
            "tdbase_nuclei_2_nuclei_3",
            loading_nuclei_400_1_path,
            "Nuclei_2",
            loading_nuclei_400_2_path,
            "Nuclei_3",
        ),
    ]:
        try:
            stats1, stats2 = _run_tdbase_and_parse_loading(
                case_name=case_name,
                mesh1=mesh1,
                mesh2=mesh2,
                dataset_id_1=dataset_id_1,
                dataset_id_2=dataset_id_2,
                adapter=tdbase_loading_adapter,
                logs_dir=logs_dir,
            )
            tdbase_loading_stats.setdefault(dataset_id_1, stats1)
            tdbase_loading_stats.setdefault(dataset_id_2, stats2)
            tdbase_loading_errors.pop(dataset_id_1, None)
            tdbase_loading_errors.pop(dataset_id_2, None)
        except Exception as exc:
            err = str(exc)
            tdbase_loading_errors.setdefault(dataset_id_1, err)
            tdbase_loading_errors.setdefault(dataset_id_2, err)
            _log(f"[tdbase-load:{case_name}] FAILED: {err}")

    sphere_1_tdbase_path = (
        Path(str(tdbase_preprocess_stats["Spheres_1"]["preprocessed_path"]))
        if "Spheres_1" in tdbase_preprocess_stats
        else None
    )
    sphere_2_tdbase_path = (
        Path(str(tdbase_preprocess_stats["Spheres_2"]["preprocessed_path"]))
        if "Spheres_2" in tdbase_preprocess_stats
        else None
    )
    if sphere_1_tdbase_path is not None and sphere_2_tdbase_path is not None:
        try:
            stats1, stats2 = _run_tdbase_and_parse_loading(
                case_name="tdbase_spheres_1_spheres_2",
                mesh1=sphere_1_tdbase_path,
                mesh2=sphere_2_tdbase_path,
                dataset_id_1="Spheres_1",
                dataset_id_2="Spheres_2",
                adapter=tdbase_loading_adapter,
                logs_dir=logs_dir,
            )
            tdbase_loading_stats.setdefault("Spheres_1", stats1)
            tdbase_loading_stats.setdefault("Spheres_2", stats2)
            tdbase_loading_errors.pop("Spheres_1", None)
            tdbase_loading_errors.pop("Spheres_2", None)
        except Exception as exc:
            err = str(exc)
            tdbase_loading_errors.setdefault("Spheres_1", err)
            tdbase_loading_errors.setdefault("Spheres_2", err)
            _log(f"[tdbase-load:tdbase_spheres_1_spheres_2] FAILED: {err}")
    else:
        missing = []
        if sphere_1_tdbase_path is None:
            missing.append("Spheres_1")
        if sphere_2_tdbase_path is None:
            missing.append("Spheres_2")
        err = (
            "TDBase sphere loading skipped because preprocessing output is missing for: "
            + ", ".join(missing)
        )
        for dataset_id in missing:
            tdbase_loading_errors.setdefault(dataset_id, err)
        _log(f"[tdbase-load:tdbase_spheres_1_spheres_2] SKIPPED: {err}")
    _log("=== Stage complete: TDBase loading extraction ===")
    required_tdbase_loading_errors = {
        dataset_id: err
        for dataset_id, err in tdbase_loading_errors.items()
        if dataset_id in TDBASE_REQUIRED_LOADING_DATASETS
    }
    if args.require_tdbase_loading and required_tdbase_loading_errors:
        raise RuntimeError(
            "TDBase loading extraction failed for required tissue datasets: "
            + " | ".join(
                f"{dataset_id}: {err.strip()}"
                for dataset_id, err in required_tdbase_loading_errors.items()
            )
        )

    loading_ms: Dict[str, float] = {}
    join_logs: Dict[str, str] = {}

    _log("=== Stage: join + loading extraction (nuclei_vessel) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="nuclei_vessel",
        mesh1=vessel_800_path,
        mesh2=nuclei_800_path,
        grid_cell_size=200.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Vessel"] = l1
    loading_ms["Nuclei_1"] = l2
    join_logs["nuclei_vessel"] = log

    _log("=== Stage: join + loading extraction (nuclei_nuclei) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="nuclei_nuclei",
        mesh1=nuclei_400_1_path,
        mesh2=nuclei_400_2_path,
        grid_cell_size=200.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Nuclei_2"] = l1
    loading_ms["Nuclei_3"] = l2
    join_logs["nuclei_nuclei"] = log

    _log("=== Stage: join + loading extraction (neurons_neurons) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="neurons_neurons",
        mesh1=neurons_1,
        mesh2=neurons_2,
        grid_cell_size=700.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Neurons_1"] = l1
    loading_ms["Neurons_2"] = l2
    join_logs["neurons_neurons"] = log

    _log("=== Stage: join + loading extraction (neurons_neurons_large) ===")
    l3, l4, log_large = _run_join_and_parse_loading(
        case_name="neurons_neurons_large",
        mesh1=neurons_3,
        mesh2=neurons_4,
        grid_cell_size=700.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Neurons_3"] = l3
    loading_ms["Neurons_4"] = l4
    join_logs["neurons_neurons_large"] = log_large

    _log("=== Stage: join + loading extraction (cubes_cubes) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="cubes_cubes",
        mesh1=cubes_1,
        mesh2=cubes_2,
        grid_cell_size=5.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Cubes_1"] = l1
    loading_ms["Cubes_2"] = l2
    join_logs["cubes_cubes"] = log
    _log("=== Stage: join + loading extraction (spheres_spheres) ===")
    l1, l2, log = _run_join_and_parse_loading(
        case_name="spheres_spheres",
        mesh1=spheres_1,
        mesh2=spheres_2,
        grid_cell_size=5.0,
        preprocessed_dir=dataset_dirs["preprocessed"],
        timings_dir=dataset_dirs["timings"],
        logs_dir=logs_dir,
    )
    loading_ms["Spheres_1"] = l1
    loading_ms["Spheres_2"] = l2
    join_logs["spheres_spheres"] = log
    _log("=== Stage complete: join + loading extraction ===")

    rows: List[Dict[str, object]] = []
    for ds in datasets:
        stats = preprocess_stats[ds.dataset_id]
        rows.append(
            {
                "dataset_id": ds.dataset_id,
                "source_path": str(ds.source_path),
                "dataset_size_bytes": ds.source_path.stat().st_size,
                "dataset_size_human": _human_bytes(ds.source_path.stat().st_size),
                "objects": int(stats["objects"]),
                "triangles": int(stats["triangles"]),
                "preprocess_pierce_ms": float(stats["preprocess_ms"]),
                "preprocess_tdbase_ms": (
                    float(tdbase_preprocess_stats[ds.dataset_id]["preprocess_ms"])
                    if ds.dataset_id in tdbase_preprocess_stats
                    and tdbase_preprocess_stats[ds.dataset_id]["preprocess_ms"] is not None
                    else None
                ),
                "loading_pierce_ms": float(loading_ms.get(ds.dataset_id, 0.0)),
                "loading_tdbase_ms": (
                    float(tdbase_loading_stats[ds.dataset_id]["loading_ms"])
                    if ds.dataset_id in tdbase_loading_stats
                    else None
                ),
                "preprocess_log": stats["log_path"],
                "preprocess_timing_json": stats["timing_path"],
                "preprocess_tdbase_path": (
                    tdbase_preprocess_stats[ds.dataset_id]["preprocessed_path"]
                    if ds.dataset_id in tdbase_preprocess_stats
                    else None
                ),
                "preprocess_tdbase_source": (
                    tdbase_preprocess_stats[ds.dataset_id]["preprocess_source"]
                    if ds.dataset_id in tdbase_preprocess_stats
                    else None
                ),
                "preprocess_tdbase_log": (
                    tdbase_preprocess_stats[ds.dataset_id]["preprocess_log_path"]
                    if ds.dataset_id in tdbase_preprocess_stats
                    else None
                ),
                "preprocess_tdbase_copy_elapsed_ms": (
                    float(tdbase_preprocess_stats[ds.dataset_id]["copy_elapsed_ms"])
                    if ds.dataset_id in tdbase_preprocess_stats
                    else None
                ),
                "tdbase_run_log": (
                    tdbase_loading_stats[ds.dataset_id]["run_log"]
                    if ds.dataset_id in tdbase_loading_stats
                    else None
                ),
                "tdbase_loading_source_path": (
                    str(tdbase_loading_sources[ds.dataset_id])
                    if ds.dataset_id in tdbase_loading_sources
                    else None
                ),
                "preprocess_tdbase_error": tdbase_preprocess_errors.get(ds.dataset_id),
                "loading_tdbase_error": tdbase_loading_errors.get(ds.dataset_id),
            }
        )

    latex_table = _build_latex_table(rows)
    latex_path = Path(run_layout["run_dir"]) / "dataset_table.tex"
    latex_path.write_text(latex_table + "\n", encoding="utf-8")
    _log(f"latex table written: {latex_path}")

    payload = {
        "metadata": {
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
            "run_dir": str(run_layout["run_dir"]),
            "nu_v": args.nu_v,
            "nu_nn": args.nu_nn,
            "microns_size_gb": args.microns_size_gb,
            "cube_count_b": args.cube_count_b,
            "tdbase_timing_mode": args.tdbase_timing_mode,
            "tdbase_large_scenario": args.tdbase_large_scenario,
            "tdbase_loading_scenario": args.tdbase_loading_scenario,
            "tdbase_loading_threads": args.tdbase_loading_threads,
            "grid_sizes": {"nu": 200.0, "microns": 700.0, "cubes": 5.0},
            "join_logs": join_logs,
            "latex_table_path": str(latex_path),
            "tdbase_preprocess_errors": tdbase_preprocess_errors,
            "tdbase_loading_errors": tdbase_loading_errors,
        },
        "rows": rows,
        "latex_table": latex_table,
    }

    write_json(Path(run_layout["results_json"]), payload)
    _log(f"results json written: {run_layout['results_json']}")

    print(f"Saved results: {run_layout['results_json']}")
    print(f"Saved LaTeX table: {latex_path}")


if __name__ == "__main__":
    main()
