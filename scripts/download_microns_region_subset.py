#!/usr/bin/env python3
"""Download MICrONS meshes from a spatial region, then convert to GLB/OBJ.

Workflow:
1) Query segmentation IDs inside a user-provided 3D region (in nanometers).
2) Download meshes for those IDs until a size target is reached.
3) Convert downloaded .npz meshes to GLB or OBJ.

Example:
  python ./download_microns_region_subset.py \
    --x-min-nm 347992 --x-max-nm 1447384 \
    --y-min-nm 300952 --y-max-nm 1116304 \
    --z-min-nm 594000 --z-max-nm 1114320 \
    --target-gb 4.0 --max-gb 4.3 \
    --format glb --separate \
    --download-dir ./tmp/microns_region_4gb_npz \
    --export-dir ./tmp/microns_region_4gb_glb
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from cloudvolume import CloudVolume

try:
    import trimesh
except ImportError as exc:
    raise ImportError("trimesh is required. Install with: pip install trimesh") from exc


DEFAULT_SEG_PATH = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-path", default=DEFAULT_SEG_PATH)
    parser.add_argument("--x-min-nm", type=float, required=True)
    parser.add_argument("--x-max-nm", type=float, required=True)
    parser.add_argument("--y-min-nm", type=float, required=True)
    parser.add_argument("--y-max-nm", type=float, required=True)
    parser.add_argument("--z-min-nm", type=float, required=True)
    parser.add_argument("--z-max-nm", type=float, required=True)
    parser.add_argument(
        "--discovery-mip",
        type=int,
        default=8,
        help="Mip used for regional ID discovery. Larger mip = lower resolution, faster.",
    )
    parser.add_argument(
        "--discovery-chunk-voxels",
        type=int,
        default=64,
        help="Chunk edge length (in voxels at discovery mip) used while scanning the bbox.",
    )
    parser.add_argument("--lod", type=int, default=0, help="Mesh LOD for download (0=highest static detail).")
    parser.add_argument("--target-gb", type=float, default=4.0)
    parser.add_argument("--max-gb", type=float, default=4.3)
    parser.add_argument("--max-meshes", type=int, default=500)
    parser.add_argument("--max-candidate-ids", type=int, default=20000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    
    script_dir = Path(__file__).parent
    parser.add_argument("--download-dir", help="Directory to store downloaded .npz files.")
    parser.add_argument("--export-dir", help="Directory to store converted meshes.")
    parser.add_argument("--format", choices=["glb", "obj"], default="glb")
    parser.add_argument("--separate", action="store_true", help="Export one file per mesh.")
    parser.add_argument(
        "--rescale-nm-to-um",
        action="store_true",
        default=True,
        help="Rescale vertices by 1/1000 for visualization tools like Blender.",
    )
    parser.add_argument("--no-rescale-nm-to-um", dest="rescale_nm_to_um", action="store_false")
    
    args = parser.parse_args()
    
    # Set dynamic defaults if not provided
    if args.download_dir is None:
        args.download_dir = str(script_dir / "microns_data" / f"microns_region_{int(args.target_gb)}gb_npz")
    if args.export_dir is None:
        args.export_dir = str(script_dir / "microns_data" / f"microns_region_{int(args.target_gb)}gb_glb")
        
    return args


def dir_size_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def format_gib(num_bytes: int) -> str:
    return f"{num_bytes / (1024 ** 3):.3f} GiB"


def discover_ids_in_region(args: argparse.Namespace, cv: CloudVolume) -> list[int]:
    cv_mip = cv[args.discovery_mip]
    res = np.asarray(cv_mip.resolution, dtype=np.float64)

    bbox_min_nm = np.array([args.x_min_nm, args.y_min_nm, args.z_min_nm], dtype=np.float64)
    bbox_max_nm = np.array([args.x_max_nm, args.y_max_nm, args.z_max_nm], dtype=np.float64)

    voxel_min = np.floor(bbox_min_nm / res).astype(np.int64)
    voxel_max = np.ceil(bbox_max_nm / res).astype(np.int64)

    if np.any(voxel_max <= voxel_min):
        raise ValueError("Invalid bbox: max must be greater than min on all axes")

    print(f"Discovery mip: {args.discovery_mip}")
    print(f"Discovery resolution (nm): {res.tolist()}")
    print(f"Discovery voxel bbox min: {voxel_min.tolist()}")
    print(f"Discovery voxel bbox max: {voxel_max.tolist()}")

    shape = voxel_max - voxel_min
    print(f"Discovery voxel shape: {shape.tolist()}")

    step = max(1, int(args.discovery_chunk_voxels))
    unique_ids: set[int] = set()
    chunk_count = 0

    for x0 in range(voxel_min[0], voxel_max[0], step):
        x1 = min(x0 + step, voxel_max[0])
        for y0 in range(voxel_min[1], voxel_max[1], step):
            y1 = min(y0 + step, voxel_max[1])
            for z0 in range(voxel_min[2], voxel_max[2], step):
                z1 = min(z0 + step, voxel_max[2])
                vol = cv_mip[x0:x1, y0:y1, z0:z1]
                vals = np.unique(np.asarray(vol))
                unique_ids.update(int(v) for v in vals if int(v) != 0)
                chunk_count += 1
                if chunk_count % 50 == 0:
                    print(f"  scanned chunks: {chunk_count}, unique IDs so far: {len(unique_ids)}")

    ids = sorted(unique_ids)

    if len(ids) > args.max_candidate_ids:
        ids = ids[: args.max_candidate_ids]

    if args.shuffle:
        random.Random(args.seed).shuffle(ids)

    print(f"Candidate IDs found in region: {len(ids)}")
    return ids


def download_meshes(
    args: argparse.Namespace,
    cv: CloudVolume,
    candidate_ids: list[int],
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    target_bytes = int(args.target_gb * (1024 ** 3))
    max_bytes = int(args.max_gb * (1024 ** 3))
    if max_bytes < target_bytes:
        raise ValueError("--max-gb must be >= --target-gb")

    initial_size = dir_size_bytes(out_dir)
    print(f"Download directory: {out_dir}")
    print(f"Initial size: {format_gib(initial_size)}")
    print(f"Target size:  {format_gib(target_bytes)}")
    print(f"Hard cap:     {format_gib(max_bytes)}")

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
            mesh = mesh_dict.get(root_id)
            if mesh is None:
                failures.append({"root_id": str(root_id), "error": "mesh_missing"})
                print("  missing mesh")
                continue

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
        "bbox_nm": {
            "x_min": args.x_min_nm,
            "x_max": args.x_max_nm,
            "y_min": args.y_min_nm,
            "y_max": args.y_max_nm,
            "z_min": args.z_min_nm,
            "z_max": args.z_max_nm,
        },
        "discovery_mip": args.discovery_mip,
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
    return summary


def load_npz_mesh(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    return data["vertices"], data["faces"]


def load_meshes_from_npz_dir(input_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    meshes: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for npz_file in sorted(input_dir.glob("*.npz")):
        try:
            root_id = int(npz_file.stem)
            meshes[root_id] = load_npz_mesh(npz_file)
        except Exception:  # noqa: BLE001
            continue
    return meshes


def export_meshes(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    export_dir: Path,
    export_format: str,
    separate: bool,
    rescale_nm_to_um: bool,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if not meshes:
        raise RuntimeError("No meshes loaded for export")

    if separate:
        print(f"Exporting {len(meshes)} meshes as separate .{export_format} files...")
        for root_id, (vertices, faces) in sorted(meshes.items()):
            verts = np.asarray(vertices, dtype=np.float32)
            if rescale_nm_to_um:
                verts = verts / 1000.0
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            out_file = export_dir / f"mesh_{root_id}.{export_format}"
            mesh.export(out_file, file_type=export_format)
        return

    scene = trimesh.Scene()
    for root_id, (vertices, faces) in sorted(meshes.items()):
        verts = np.asarray(vertices, dtype=np.float32)
        if rescale_nm_to_um:
            verts = verts / 1000.0
        mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        scene.add_geometry(mesh, node_name=f"mesh_{root_id}")

    out_file = export_dir / f"microns_region_combined.{export_format}"
    scene.export(out_file, file_type=export_format)


def main() -> None:
    args = parse_args()
    download_dir = Path(args.download_dir)
    export_dir = Path(args.export_dir)

    print("Initializing CloudVolume...")
    cv = CloudVolume(args.segmentation_path, progress=False, use_https=True)
    print("CloudVolume ready.")

    print("Discovering IDs in region...")
    candidate_ids = discover_ids_in_region(args, cv)
    if not candidate_ids:
        raise RuntimeError("No candidate IDs found in the specified region")

    print("Starting mesh download...")
    summary = download_meshes(args, cv, candidate_ids, download_dir)
    summary_file = download_dir / "download_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print("Loading downloaded meshes for export...")
    meshes = load_meshes_from_npz_dir(download_dir)
    print(f"Meshes loaded for export: {len(meshes)}")
    export_meshes(
        meshes=meshes,
        export_dir=export_dir,
        export_format=args.format,
        separate=args.separate,
        rescale_nm_to_um=args.rescale_nm_to_um,
    )

    print("\nDone.")
    print(f"Downloaded .npz directory: {download_dir}")
    print(f"Export directory: {export_dir}")
    print(f"Summary JSON: {summary_file}")


if __name__ == "__main__":
    main()