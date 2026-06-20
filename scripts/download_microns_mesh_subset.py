#!/usr/bin/env python3
"""Download a subset of MICrONS meshes up to a target local size.

Default behavior downloads static meshes at highest static detail (lod=0)
from seg_m1300 and stores each mesh as an .npz file.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume


DEFAULT_SEG_PATH = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
DEFAULT_ID_SOURCE = "gs://microns-static-links/skel/swc/proofread"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segmentation-path",
        default=DEFAULT_SEG_PATH,
        help="CloudVolume segmentation path.",
    )
    parser.add_argument(
        "--id-source",
        default=DEFAULT_ID_SOURCE,
        help="Bucket/folder used to enumerate candidate root IDs from .swc filenames.",
    )
    parser.add_argument(
        "--ids-file",
        default=None,
        help="Optional text file with one root_id per line.",
    )
    
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--out-dir",
        default=str(script_dir / "microns_data" / "microns_mesh_subset_lod0"),
        help="Output directory for downloaded meshes.",
    )
    parser.add_argument(
        "--target-gb",
        type=float,
        default=2.0,
        help="Target output size in GiB.",
    )
    parser.add_argument(
        "--max-gb",
        type=float,
        default=2.2,
        help="Hard stop output size in GiB to avoid overshooting too much.",
    )
    parser.add_argument(
        "--lod",
        type=int,
        default=0,
        help="Mesh level of detail. Highest static detail is 0.",
    )
    parser.add_argument(
        "--max-meshes",
        type=int,
        default=200,
        help="Safety cap on number of meshes to process.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle candidate IDs before download.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when --shuffle is enabled.",
    )
    return parser.parse_args()


def dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.3f} GiB"


def load_ids_from_file(ids_file: Path) -> list[int]:
    ids: list[int] = []
    for line in ids_file.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        ids.append(int(stripped))
    return ids


def enumerate_ids_from_swc_bucket(id_source: str, limit: int) -> list[int]:
    cf = CloudFiles(id_source, use_https=True)
    ids: list[int] = []
    for rel_path in cf.list():
        if not rel_path.endswith(".swc"):
            continue
        stem = Path(rel_path).stem
        try:
            ids.append(int(stem))
        except ValueError:
            continue
        if len(ids) >= limit:
            break
    return ids


def iter_candidate_ids(args: argparse.Namespace) -> Iterable[int]:
    if args.ids_file:
        return load_ids_from_file(Path(args.ids_file))
    # Pull more candidates than max_meshes to account for download failures.
    candidate_limit = max(args.max_meshes * 5, 500)
    return enumerate_ids_from_swc_bucket(args.id_source, candidate_limit)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gb * (1024 ** 3))
    max_bytes = int(args.max_gb * (1024 ** 3))
    if max_bytes < target_bytes:
        raise ValueError("--max-gb must be >= --target-gb")

    cv = CloudVolume(args.segmentation_path, progress=False, use_https=True)
    candidate_ids = list(iter_candidate_ids(args))
    if args.shuffle:
        random.Random(args.seed).shuffle(candidate_ids)

    initial_size = dir_size_bytes(out_dir)
    print(f"Output directory: {out_dir}")
    print(f"Initial size: {format_gib(initial_size)}")
    print(f"Target size:  {format_gib(target_bytes)}")
    print(f"Hard cap:     {format_gib(max_bytes)}")
    print(f"Candidates loaded: {len(candidate_ids)}")

    downloaded: list[dict[str, int]] = []
    failures: list[dict[str, str]] = []

    for idx, root_id in enumerate(candidate_ids, start=1):
        if len(downloaded) >= args.max_meshes:
            print(f"Reached --max-meshes={args.max_meshes}. Stopping.")
            break

        current_size = dir_size_bytes(out_dir)
        if current_size >= target_bytes:
            print("Reached target size. Stopping.")
            break
        if current_size >= max_bytes:
            print("Reached hard cap size. Stopping.")
            break

        mesh_file = out_dir / f"{root_id}.npz"
        if mesh_file.exists():
            continue

        print(f"[{idx}/{len(candidate_ids)}] downloading root_id={root_id} lod={args.lod} ...")
        try:
            mesh_dict = cv.mesh.get(root_id, lod=args.lod)
            if root_id not in mesh_dict:
                failures.append({"root_id": str(root_id), "error": "mesh_missing"})
                print(f"  missing mesh for {root_id}")
                continue

            mesh = mesh_dict[root_id]
            np.savez(mesh_file, vertices=mesh.vertices, faces=mesh.faces)

            after_size = dir_size_bytes(out_dir)
            added = after_size - current_size
            downloaded.append(
                {
                    "root_id": int(root_id),
                    "added_bytes": int(added),
                    "total_bytes": int(after_size),
                }
            )
            print(
                f"  saved {mesh_file.name}: +{format_gib(added)} "
                f"(total {format_gib(after_size)})"
            )
        except Exception as exc:  # noqa: BLE001
            failures.append({"root_id": str(root_id), "error": str(exc)})
            print(f"  failed: {exc}")

    final_size = dir_size_bytes(out_dir)
    summary = {
        "segmentation_path": args.segmentation_path,
        "id_source": args.id_source,
        "lod": args.lod,
        "target_gb": args.target_gb,
        "max_gb": args.max_gb,
        "initial_bytes": initial_size,
        "final_bytes": final_size,
        "downloaded_count": len(downloaded),
        "failure_count": len(failures),
        "downloaded": downloaded,
        "failures": failures,
    }
    summary_file = out_dir / "download_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print("\nDone.")
    print(f"Final size: {format_gib(final_size)}")
    print(f"Downloaded meshes: {len(downloaded)}")
    print(f"Failures: {len(failures)}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
