import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARKS_DIR = SCRIPT_DIR.parent
REPO_ROOT = BENCHMARKS_DIR.parent
PIERCE_DIR = REPO_ROOT / "pierce"
SHARED_DATA_ROOT = REPO_ROOT / "data"
GENERATE_CUBES_SCRIPT = PIERCE_DIR / "scripts" / "generate_cubes_by_selectivity.py"
GENERATE_SPHERES_BIN = PIERCE_DIR / "scripts" / "cpp_generator" / "generate_spheres"
CUBE_SCALABILITY_COUNTS = [200000, 400000, 600000, 1000000]


def build_selectivity_sweep(
    min_selectivity: float = 0.0001,
    max_selectivity: float = 0.01,
    num_points: int = 10,
    scale: str = "log",
) -> list[float]:
    if min_selectivity <= 0 or max_selectivity <= 0:
        raise ValueError("Selectivity bounds must be positive")
    if min_selectivity > max_selectivity:
        raise ValueError("min_selectivity must be <= max_selectivity")
    if num_points < 2:
        raise ValueError("num_points must be at least 2")

    if scale == "log":
        log_min = math.log10(min_selectivity)
        log_max = math.log10(max_selectivity)
        step = (log_max - log_min) / (num_points - 1)
        return [float(f"{10 ** (log_min + step * i):.8g}") for i in range(num_points)]
    elif scale == "linear":
        step = (max_selectivity - min_selectivity) / (num_points - 1)
        return [float(f"{(min_selectivity + step * i):.8g}") for i in range(num_points)]
    else:
        raise ValueError(f"Unknown scale: {scale}")


def timestamp_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sanitize_float_token(value: float) -> str:
    return str(value).replace(".", "_")


def canonical_cube_pair_paths(
    raw_dir: Path,
    *,
    num_cubes_a: int,
    num_cubes_b: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
    grid_cell_size: int | None = None,
) -> Tuple[Path, Path]:
    min_tok = sanitize_float_token(min_size)
    max_tok = sanitize_float_token(max_size)
    sel_tok = sanitize_float_token(selectivity)
    stem = (
        f"cubes_na{num_cubes_a}_nb{num_cubes_b}_"
        f"min{min_tok}_max{max_tok}_sel{sel_tok}_seed{seed}"
    )
    if grid_cell_size is not None:
        stem += f"_g{grid_cell_size}"
    return raw_dir / f"{stem}_a.obj", raw_dir / f"{stem}_b.obj"


def canonical_sphere_pair_paths(
    raw_dir: Path,
    *,
    template_name: str,
    num_objects: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
    grid_cell_size: int | None = None,
) -> Tuple[Path, Path]:
    min_tok = sanitize_float_token(min_size)
    max_tok = sanitize_float_token(max_size)
    sel_tok = sanitize_float_token(selectivity)
    template_token = template_name.replace(".obj", "").replace(" ", "_")
    stem = (
        f"spheres_tpl{template_token}_n{num_objects}_"
        f"min{min_tok}_max{max_tok}_sel{sel_tok}_seed{seed}"
    )
    if grid_cell_size is not None:
        stem += f"_g{grid_cell_size}"
    return raw_dir / f"{stem}_a.obj", raw_dir / f"{stem}_b.obj"


def canonical_nu_pair_paths(
    raw_dir: Path,
    *,
    nu: int,
    nv: int = 150,
    vs: int = 100,
    radius: int = 30,
    prefix: str = "tdbase",
) -> Tuple[Path, Path]:
    stem = f"{prefix}_n_nv{nv}_nu{nu}"
    n_file = raw_dir / f"{stem}_n_nv{nv}_nu{nu}_vs{vs}_r{radius}.dt"
    v_file = raw_dir / f"{stem}_v_nv{nv}_nu{nu}_vs{vs}_r{radius}.dt"
    return n_file, v_file


def canonical_nn_pair_paths(
    raw_dir: Path,
    *,
    nu: int,
    nv: int = 150,
    vs: int = 100,
    radius: int = 30,
    prefix: str = "tdbase",
) -> Tuple[Path, Path]:
    stem1 = f"{prefix}_n_nv{nv}_nu{nu}"
    stem2 = f"{prefix}_nn_nv{nv}_nu{nu}"
    n_file1 = raw_dir / f"{stem1}_n_nv{nv}_nu{nu}_vs{vs}_r{radius}.dt"
    n_file2 = raw_dir / f"{stem2}_n_nv{nv}_nu{nu}_vs{vs}_r{radius}.dt"
    return n_file1, n_file2


def compute_universe_for_selectivity(target_selectivity: float, min_size: float, max_size: float) -> float:
    if target_selectivity <= 0:
        raise ValueError("Target selectivity must be positive")
    avg_size = (min_size + max_size) / 2.0
    return (2.0 * avg_size) / (target_selectivity ** (1.0 / 3.0))


def get_shared_data_dirs(scenario_name: str) -> Dict[str, Path]:
    scenario_root = SHARED_DATA_ROOT / scenario_name
    raw_dir = scenario_root / "raw"
    preprocessed_dir = scenario_root / "preprocessed"
    timings_dir = scenario_root / "timings"
    for d in (raw_dir, preprocessed_dir, timings_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "root": scenario_root,
        "raw": raw_dir,
        "preprocessed": preprocessed_dir,
        "timings": timings_dir,
    }


def create_isolated_run_data_dirs(run_dir: Path) -> Dict[str, Path]:
    isolated_root = run_dir / "isolated_data"
    raw_dir = isolated_root / "raw"
    preprocessed_dir = isolated_root / "preprocessed"
    timings_dir = isolated_root / "timings"
    for d in (raw_dir, preprocessed_dir, timings_dir):
        d.mkdir(parents=True, exist_ok=True)
    return {
        "root": isolated_root,
        "raw": raw_dir,
        "preprocessed": preprocessed_dir,
        "timings": timings_dir,
    }


def stage_input_file(source_path: Path, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    staged_path = raw_dir / source_path.name
    if staged_path.exists():
        return staged_path
    shutil.copyfile(source_path, staged_path)
    return staged_path


def stage_input_files(source_paths: Iterable[Path], raw_dir: Path) -> list[Path]:
    return [stage_input_file(Path(source_path), raw_dir) for source_path in source_paths]


def run_cmd(cmd, desc: str):
    print(f"\n>>> {desc}")
    print("    " + " ".join(str(c) for c in cmd))
    return subprocess.run([str(c) for c in cmd], check=True)


def ensure_cube_pair_dataset(
    output_a: Path,
    output_b: Path,
    *,
    num_cubes_a: int,
    num_cubes_b: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
    python_executable: str = sys.executable,
) -> Tuple[Path, Path]:
    if output_a.exists() and output_b.exists():
        return output_a, output_b

    cmd = [
        python_executable,
        str(GENERATE_CUBES_SCRIPT),
        "--num-cubes-a", str(num_cubes_a),
        "--num-cubes-b", str(num_cubes_b),
        "--min-size", str(min_size),
        "--max-size", str(max_size),
        "--selectivity", str(selectivity),
        "--output-a", str(output_a),
        "--output-b", str(output_b),
        "--seed", str(seed),
    ]
    run_cmd(cmd, f"Generating cubes (nA={num_cubes_a}, nB={num_cubes_b}, sel={selectivity})")
    return output_a, output_b


def ensure_sphere_pair_dataset(
    output_a: Path,
    output_b: Path,
    *,
    template_obj: Path,
    num_objects: int,
    min_size: float,
    max_size: float,
    selectivity: float,
    seed: int,
) -> Tuple[Path, Path]:
    if output_a.exists() and output_b.exists():
        return output_a, output_b

    cmd = [
        str(GENERATE_SPHERES_BIN),
        "--template-obj", str(template_obj),
        "--num-objs-a", str(num_objects),
        "--num-objs-b", str(num_objects),
        "--min-size", str(min_size),
        "--max-size", str(max_size),
        "--selectivity", str(selectivity),
        "-oa", str(output_a),
        "-ob", str(output_b),
        "--seed", str(seed),
    ]
    run_cmd(cmd, f"Generating spheres from {template_obj.name} (n={num_objects}, sel={selectivity})")
    return output_a, output_b


def ensure_nu_pair_dataset(
    output_n: Path,
    output_v: Path,
    *,
    legacy_raw_dirs: list[Path] | None = None,
) -> Tuple[Path, Path]:
    if output_n.exists() and output_v.exists():
        return output_n, output_v

    search_dirs = legacy_raw_dirs or []
    for base_dir in search_dirs:
        cand_n = base_dir / output_n.name
        cand_v = base_dir / output_v.name
        if cand_n.exists() and cand_v.exists():
            output_n.parent.mkdir(parents=True, exist_ok=True)
            output_v.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cand_n, output_n)
            shutil.copyfile(cand_v, output_v)
            return output_n, output_v

    raise FileNotFoundError(
        "Could not resolve nu dataset pair. Missing files: "
        f"{output_n} and/or {output_v}. "
        "Provide the canonical files under benchmarks/data_shared/nu_scalability/raw "
        "or pass legacy_raw_dirs containing exact filenames."
    )


def ensure_nn_pair_dataset(
    output_n1: Path,
    output_n2: Path,
    *,
    legacy_raw_dirs: list[Path] | None = None,
) -> Tuple[Path, Path]:
    if output_n1.exists() and output_n2.exists():
        return output_n1, output_n2

    search_dirs = legacy_raw_dirs or []
    for base_dir in search_dirs:
        cand_n1 = base_dir / output_n1.name
        cand_n2 = base_dir / output_n2.name
        if cand_n1.exists() and cand_n2.exists():
            output_n1.parent.mkdir(parents=True, exist_ok=True)
            output_n2.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(cand_n1, output_n1)
            shutil.copyfile(cand_n2, output_n2)
            return output_n1, output_n2

    raise FileNotFoundError(
        "Could not resolve nn dataset pair. Missing files: "
        f"{output_n1} and/or {output_n2}. "
        "Ensure you have generated the second nuclei dataset using generate_data.sh."
    )


def count_vertices(obj_path: Path) -> int:
    if obj_path.suffix.lower() != ".obj":
        return 0
    count = 0
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                count += 1
    return count


def count_triangles(obj_path: Path) -> int:
    if obj_path.suffix.lower() != ".obj":
        return 0
    count = 0
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("f "):
                count += 1
    return count


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def create_benchmark_run_layout(
    benchmark_dir: Path,
    benchmark_name: str,
    *,
    timestamp: str | None = None,
) -> Dict[str, Path | str]:
    ts = timestamp or timestamp_tag()
    run_name = f"{benchmark_name}_{ts}"
    runs_dir = benchmark_dir / "runs"
    run_dir = runs_dir / run_name
    logs_dir = run_dir / "logs"
    figures_dir = run_dir / "figures"
    results_json = run_dir / "results.json"

    logs_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {
        "timestamp": ts,
        "run_name": run_name,
        "runs_dir": runs_dir,
        "run_dir": run_dir,
        "logs_dir": logs_dir,
        "figures_dir": figures_dir,
        "results_json": results_json,
    }


def write_latest_json_alias(latest_path: Path, payload) -> None:
    write_json(latest_path, payload)


def copy_to_latest_file(source_path: Path, latest_path: Path) -> None:
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_path, latest_path)


def canonical_microns_aggregated_paths(raw_dir: Path, size_gb: int) -> Tuple[Path, Path]:
    return raw_dir / f"microns_{size_gb}gb_split_a_aggregated.obj", raw_dir / f"microns_{size_gb}gb_split_b_aggregated.obj"


def ensure_microns_splits(
    size_gb: int,
    source_root: Path,
    splits_dir: Path,
) -> Tuple[Path, Path]:
    split_a = splits_dir / f"microns_{size_gb}gb_split_a.txt"
    split_b = splits_dir / f"microns_{size_gb}gb_split_b.txt"
    meta_path = splits_dir / f"microns_{size_gb}gb_meta.json"

    source_dir = source_root / f"microns_region_{size_gb}gb_glb"
    
    if split_a.exists() and split_b.exists() and meta_path.exists():
        return split_a, split_b

    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    files = sorted([f for f in source_dir.glob("*.glb")])
    if not files:
        raise ValueError(f"No GLB files found in {source_dir}")

    a_files = [f for i, f in enumerate(files) if i % 2 == 0]
    b_files = [f for i, f in enumerate(files) if i % 2 == 1]

    splits_dir.mkdir(parents=True, exist_ok=True)
    with open(split_a, 'w') as f:
        f.write('\n'.join(str(p.resolve()) for p in a_files))
    with open(split_b, 'w') as f:
        f.write('\n'.join(str(p.resolve()) for p in b_files))

    meta = {
        "source_dir": str(source_dir.resolve()),
        "total_files": len(files),
        "split_counts": {"A": len(a_files), "B": len(b_files)},
        "rule": "alternating_50_50",
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    return split_a, split_b


def _hash_file_content(path: Path) -> str:
    import hashlib
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()


def ensure_microns_aggregated_meshes(
    manifest_a: Path,
    manifest_b: Path,
    output_a: Path,
    output_b: Path,
) -> Tuple[Path, Path]:
    def process_split(manifest: Path, output: Path):
        meta_path = output.with_suffix('.meta.json')
        current_hash = _hash_file_content(manifest)
        if output.exists() and meta_path.exists():
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            if meta.get("manifest_hash") == current_hash:
                return

        print(f"Materializing aggregate: {output.name}")
        import trimesh
        with open(manifest, 'r') as f:
            paths = [Path(line.strip()) for line in f if line.strip()]

        meshes = []
        for p in paths:
            try:
                m = trimesh.load(p, force='mesh')
                if isinstance(m, trimesh.Scene):
                    for geom in m.geometry.values():
                        meshes.append(geom)
                elif isinstance(m, trimesh.Trimesh):
                    meshes.append(m)
            except Exception as e:
                print(f"Warning: Failed to load {p}: {e}")
        
        if not meshes:
            raise ValueError(f"No valid meshes found for manifest {manifest}")

        combined = trimesh.Scene(meshes)
        output.parent.mkdir(parents=True, exist_ok=True)
        # Export as OBJ
        combined.export(str(output))

        meta = {
            "manifest_hash": current_hash,
            "input_file_count": len(paths),
            "output_file_size_bytes": output.stat().st_size,
        }
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    process_split(manifest_a, output_a)
    process_split(manifest_b, output_b)
    return output_a, output_b
