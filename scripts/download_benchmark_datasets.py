#!/usr/bin/env python3
"""Install the benchmark inputs published in the HPI-Pierce Hugging Face repos.

The benchmark drivers use the paths below directly.  This script only installs
raw inputs; preprocessing remains intentionally owned by the benchmark drivers.
Use ``--archive-existing`` once to make a clean-room dataset installation.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
# Keep the default on the checkout's filesystem without embedding a site-specific
# path. Clusters can set PIERCE_HF_CACHE once in their shell/module setup.
DEFAULT_CACHE_DIR = Path(os.environ.get("PIERCE_HF_CACHE", REPO_ROOT / ".hf-cache"))
NUCLEI_REPO = "HPI-Pierce/synthetic-nuclei-vessel"
SYNTHETIC_REPO = "HPI-Pierce/synthetic-cubes-spheres"
MICRONS_REPO = "HPI-Pierce/microns"

# Pin the revisions so a clean setup remains repeatable if a dataset changes.
REVISIONS = {
    NUCLEI_REPO: "89d5bcfd3fe6e358c94bf409f12fc029ea4e9ffb",
    SYNTHETIC_REPO: "d15a3c0fe97ccc9fa23e2a4a231ca192fd859a87",
    MICRONS_REPO: "1bb5d278b09327a20e7684ebd7abb23b418af3a4",
}


@dataclass(frozen=True)
class DatasetFile:
    repository: str
    remote_path: str
    destination: str
    size_bytes: int


def _files(repository: str, directory: str, values: list[tuple[str, int]]) -> list[DatasetFile]:
    return [DatasetFile(repository, name, f"{directory}/{Path(name).name}", size) for name, size in values]


NUCLEI_FILES = _files(NUCLEI_REPO, "data/large_nu_nn_scalability/raw", [
    ("tdbase_large_n_nv750_nu200_n_nv750_nu200_vs100_r30.dt", 975256201),
    ("tdbase_large_n_nv750_nu200_v_nv750_nu200_vs100_r30.dt", 495019432),
    ("tdbase_large_n_nv750_nu400_n_nv750_nu400_vs100_r30.dt", 1950512401),
    ("tdbase_large_n_nv750_nu400_v_nv750_nu400_vs100_r30.dt", 495019432),
    ("tdbase_large_n_nv750_nu600_n_nv750_nu600_vs100_r30.dt", 2925768601),
    ("tdbase_large_n_nv750_nu600_v_nv750_nu600_vs100_r30.dt", 495019432),
    ("tdbase_large_n_nv750_nu800_n_nv750_nu800_vs100_r30.dt", 3901024801),
    ("tdbase_large_n_nv750_nu800_v_nv750_nu800_vs100_r30.dt", 495019432),
    ("tdbase_large_nn_nv750_nu200_n_nv750_nu200_vs100_r30.dt", 975256201),
    ("tdbase_large_nn_nv750_nu400_n_nv750_nu400_vs100_r30.dt", 1950512401),
])

CUBE_FILES = _files(SYNTHETIC_REPO, "data/cube_scalability/raw", [
    ("cube_scalability/cubes_na200000_nb1000000_min1_0_max2_0_sel0_001_seed42_g5.0_a.obj", 109948995),
    ("cube_scalability/cubes_na200000_nb1000000_min1_0_max2_0_sel0_001_seed42_g5.0_b.obj", 570187288),
    ("cube_scalability/cubes_na200000_nb200000_min1_0_max2_0_sel0_001_seed42_g5.0_a.obj", 109951583),
    ("cube_scalability/cubes_na200000_nb200000_min1_0_max2_0_sel0_001_seed42_g5.0_b.obj", 109951031),
    ("cube_scalability/cubes_na200000_nb400000_min1_0_max2_0_sel0_001_seed42_g5.0_a.obj", 109948995),
    ("cube_scalability/cubes_na200000_nb400000_min1_0_max2_0_sel0_001_seed42_g5.0_b.obj", 225007915),
    ("cube_scalability/cubes_na200000_nb600000_min1_0_max2_0_sel0_001_seed42_g5.0_a.obj", 109948995),
    ("cube_scalability/cubes_na200000_nb600000_min1_0_max2_0_sel0_001_seed42_g5.0_b.obj", 340067115),
])

SPHERE_FILES = _files(SYNTHETIC_REPO, "data/mesh_complexity/raw", [
    ("mesh_complexity/spheres_tplSphere_Stage_1_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_a.obj", 834651),
    ("mesh_complexity/spheres_tplSphere_Stage_1_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_b.obj", 834125),
    ("mesh_complexity/spheres_tplSphere_Stage_2_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_a.obj", 117209461),
    ("mesh_complexity/spheres_tplSphere_Stage_2_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_b.obj", 117150128),
    ("mesh_complexity/spheres_tplSphere_Stage_3_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_a.obj", 444755576),
    ("mesh_complexity/spheres_tplSphere_Stage_3_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_b.obj", 444538510),
    ("mesh_complexity/spheres_tplSphere_Stage_4_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_a.obj", 987660072),
    ("mesh_complexity/spheres_tplSphere_Stage_4_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_b.obj", 987187229),
    ("mesh_complexity/spheres_tplSphere_Stage_5_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_a.obj", 1777562305),
    ("mesh_complexity/spheres_tplSphere_Stage_5_n500_min1_0_max5_0_sel0_0005_seed42_g5.0_b.obj", 1776737863),
])

# Keep the data table explicit: filenames are part of the benchmark interface.
SELECTIVITY_FILES = _files(SYNTHETIC_REPO, "data/selectivity/raw", [
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0001_seed42_g22_a.obj", 26771278), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0001_seed42_g22_b.obj", 26770894),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0012_seed42_g9_a.obj", 26548102), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0012_seed42_g9_b.obj", 26550874),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0023_seed42_g8_a.obj", 26490366), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0023_seed42_g8_b.obj", 26492806),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0034_seed42_g7_a.obj", 26449526), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0034_seed42_g7_b.obj", 26450882),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0045_seed42_g6_a.obj", 26417250), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0045_seed42_g6_b.obj", 26417498),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0056_seed42_g6_a.obj", 26389406), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0056_seed42_g6_b.obj", 26389366),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0067_seed42_g5_a.obj", 26365082), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0067_seed42_g5_b.obj", 26364294),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0078_seed42_g5_a.obj", 26343218), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0078_seed42_g5_b.obj", 26341982),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0089_seed42_g5_a.obj", 26323106), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_0089_seed42_g5_b.obj", 26322042),
    ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_01_seed42_g5_a.obj", 26304838), ("selectivity/cubes_na50000_nb50000_min1_max4_sel0_01_seed42_g5_b.obj", 26303898),
])

MICRONS_FILES = _files(MICRONS_REPO, "data/microns_overlap/raw", [
    ("microns_large_split_a_aggregated.meta.json", 156),
    ("microns_large_split_a_aggregated.obj", 11619986130),
    ("microns_large_split_b_aggregated.meta.json", 156),
    ("microns_large_split_b_aggregated.obj", 12058069959),
    ("microns_small_split_a_aggregated.meta.json", 155),
    ("microns_small_split_a_aggregated.obj", 6260427985),
    ("microns_small_split_b_aggregated.meta.json", 155),
    ("microns_small_split_b_aggregated.obj", 5666383288),
])

MANIFEST = NUCLEI_FILES + CUBE_FILES + SPHERE_FILES + SELECTIVITY_FILES + MICRONS_FILES


def parse_revisions(values: list[str]) -> dict[str, str]:
    revisions = REVISIONS.copy()
    for value in values:
        repository, separator, revision = value.partition("=")
        if not separator or repository not in revisions or not revision:
            choices = ", ".join(REVISIONS)
            raise ValueError(f"--revision must be REPOSITORY=REVISION; repositories: {choices}")
        revisions[repository] = revision
    return revisions


def is_valid(entry: DatasetFile) -> bool:
    path = REPO_ROOT / entry.destination
    return path.is_file() and path.stat().st_size == entry.size_bytes


def verify() -> bool:
    invalid = [entry for entry in MANIFEST if not is_valid(entry)]
    if not invalid:
        print(f"Verified {len(MANIFEST)} benchmark inputs.")
        return True
    print(f"Verification failed: {len(invalid)} of {len(MANIFEST)} files are missing or have an unexpected size.")
    for entry in invalid:
        path = REPO_ROOT / entry.destination
        actual = path.stat().st_size if path.is_file() else "missing"
        print(f"  {entry.destination}: expected {entry.size_bytes}, got {actual}")
    return False


def archive_existing() -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    sources = [REPO_ROOT / "data", REPO_ROOT / "scripts" / "microns_data"]
    moves = [(source, source.with_name(f"{source.name}.hf-backup-{timestamp}")) for source in sources if source.exists()]
    collisions = [destination for _, destination in moves if destination.exists()]
    if collisions:
        raise RuntimeError("Refusing to overwrite archive path(s): " + ", ".join(map(str, collisions)))
    for source, destination in moves:
        source.rename(destination)
        print(f"Archived {source.relative_to(REPO_ROOT)} -> {destination.relative_to(REPO_ROOT)}")
    if moves:
        print("Restore by removing the new destination only after saving it, then renaming these backups back.")


def download(revisions: dict[str, str], cache_dir: Path) -> bool:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("huggingface_hub is required. Install the benchmark environment or run: pip install huggingface_hub", file=sys.stderr)
        return False

    pending = [entry for entry in MANIFEST if not is_valid(entry)]
    print(f"{len(MANIFEST) - len(pending)} inputs already valid; downloading {len(pending)} input(s).")
    for index, entry in enumerate(pending, 1):
        destination = REPO_ROOT / entry.destination
        print(f"[{index}/{len(pending)}] {entry.repository}:{entry.remote_path} -> {entry.destination}")
        cached_path = Path(hf_hub_download(
            repo_id=entry.repository,
            repo_type="dataset",
            filename=entry.remote_path,
            revision=revisions[entry.repository],
            cache_dir=str(cache_dir),
        ))
        if cached_path.stat().st_size != entry.size_bytes:
            raise RuntimeError(
                f"Downloaded size mismatch for {entry.remote_path}: "
                f"expected {entry.size_bytes}, got {cached_path.stat().st_size}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.hf-download-{os.getpid()}")
        try:
            shutil.copyfile(cached_path, temporary)
            if temporary.stat().st_size != entry.size_bytes:
                raise RuntimeError(f"Copied size mismatch for {destination}")
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    return verify()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-existing", action="store_true", help="rename data/ and scripts/microns_data/ before installing")
    parser.add_argument("--dry-run", action="store_true", help="print the pinned manifest without changing files")
    parser.add_argument("--verify", action="store_true", help="check local files only; do not download")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Hugging Face cache directory (default: {DEFAULT_CACHE_DIR})",
    )
    parser.add_argument("--revision", action="append", default=[], metavar="REPOSITORY=REVISION", help="override one pinned dataset revision")
    args = parser.parse_args()
    if args.dry_run and (args.archive_existing or args.verify):
        parser.error("--dry-run cannot be combined with --archive-existing or --verify")
    if args.verify and args.archive_existing:
        parser.error("--verify cannot be combined with --archive-existing")

    try:
        revisions = parse_revisions(args.revision)
    except ValueError as error:
        parser.error(str(error))

    total = sum(entry.size_bytes for entry in MANIFEST)
    if args.dry_run:
        print(f"{len(MANIFEST)} files, {total / 1_000_000_000:.2f} GB total")
        print(f"Hugging Face cache: {args.cache_dir}")
        if args.archive_existing:
            print("Would archive data/ and scripts/microns_data/")
        for entry in MANIFEST:
            print(f"{entry.repository}@{revisions[entry.repository]} {entry.remote_path} -> {entry.destination}")
        return 0
    if args.verify:
        return 0 if verify() else 1
    if args.archive_existing:
        archive_existing()
    return 0 if download(revisions, args.cache_dir) else 1


if __name__ == "__main__":
    raise SystemExit(main())
