#!/usr/bin/env python3
"""Download MICrONS meshes intersecting a 3D region until a size target.

This workflow uses candidate root IDs (default: proofread SWC IDs), fetches each
mesh, and keeps only meshes whose mesh bounding box intersects the requested
region in nanometers.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import random
from pathlib import Path

import numpy as np
from cloudfiles import CloudFiles
from cloudvolume import CloudVolume

try:
    import trimesh
except ImportError as exc:
    raise ImportError("trimesh is required. Install with: pip install trimesh") from exc


DEFAULT_SEG_PATH = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
DEFAULT_ID_SOURCE = "gs://microns-static-links/skel/swc/proofread"
DEFAULT_NUCLEUS_CSV_BUCKET = "gs://iarpa_microns/minnie/minnie65"
DEFAULT_NUCLEUS_CSV_PATH = "nucleus_detection/nucleus_detection_v0.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation-path", default=DEFAULT_SEG_PATH)
    parser.add_argument("--id-source", default=DEFAULT_ID_SOURCE)
    parser.add_argument("--x-min-nm", type=float, required=True)
    parser.add_argument("--x-max-nm", type=float, required=True)
    parser.add_argument("--y-min-nm", type=float, required=True)
    parser.add_argument("--y-max-nm", type=float, required=True)
    parser.add_argument("--z-min-nm", type=float, required=True)
    parser.add_argument("--z-max-nm", type=float, required=True)
    parser.add_argument("--lod", type=int, default=0)
    parser.add_argument("--target-gb", type=float, default=4.0)
    parser.add_argument("--max-gb", type=float, default=4.3)
    parser.add_argument("--max-meshes", type=int, default=500)
    parser.add_argument("--candidate-limit", type=int, default=20000)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    
    script_dir = Path(__file__).parent
    parser.add_argument("--download-dir", help="Directory to store downloaded .npz files.")
    parser.add_argument("--export-dir", help="Directory to store converted meshes.")
    parser.add_argument("--format", choices=["glb", "obj"], default="glb")
    parser.add_argument("--separate", action="store_true", default=True)
    parser.add_argument("--no-rescale", action="store_true")
    parser.add_argument(
        "--include-nuclei",
        action="store_true",
        help="Add nucleus proxy meshes for downloaded neuron root IDs.",
    )
    parser.add_argument(
        "--nucleus-csv-bucket",
        default=DEFAULT_NUCLEUS_CSV_BUCKET,
        help="Bucket containing nucleus_detection CSV.",
    )
    parser.add_argument(
        "--nucleus-csv-path",
        default=DEFAULT_NUCLEUS_CSV_PATH,
        help="Path to nucleus detection CSV in nucleus-csv-bucket.",
    )
    parser.add_argument(
        "--nucleus-voxel-size-nm",
        nargs=3,
        type=float,
        default=[4.0, 4.0, 40.0],
        metavar=("VX", "VY", "VZ"),
        help="Voxel size to convert nucleus CSV coordinates to nm.",
    )
    parser.add_argument(
        "--nucleus-icosphere-subdivisions",
        type=int,
        default=2,
        help="Icosphere detail level for generated nucleus proxy meshes.",
    )
    parser.add_argument(
        "--nucleus-min-radius-um",
        type=float,
        default=1.0,
        help="Minimum nucleus proxy radius in micrometers.",
    )
    
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


def intersects_region(min_v: np.ndarray, max_v: np.ndarray, region_min: np.ndarray, region_max: np.ndarray) -> bool:
    return bool(np.all(max_v >= region_min) and np.all(min_v <= region_max))


def load_meshes_from_npz_dir(input_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for npz_file in sorted(input_dir.glob("*.npz")):
        try:
            data = np.load(npz_file)
            meshes[npz_file.stem] = (data["vertices"], data["faces"])
        except Exception:  # noqa: BLE001
            continue
    return meshes


def add_nucleus_proxy_meshes(
    neuron_root_ids: list[int],
    out_dir: Path,
    region_min: np.ndarray,
    region_max: np.ndarray,
    args: argparse.Namespace,
) -> list[dict[str, float | int]]:
    if not neuron_root_ids:
        return []

    cf = CloudFiles(args.nucleus_csv_bucket, use_https=True)
    raw = cf.get(args.nucleus_csv_path)
    if not raw:
        print("Could not load nucleus CSV; skipping nuclei.")
        return []

    root_set = set(int(r) for r in neuron_root_ids)
    voxel_size_nm = np.asarray(args.nucleus_voxel_size_nm, dtype=np.float64)
    nucleus_rows = csv.reader(io.StringIO(raw.decode("utf-8", errors="ignore")))

    added: list[dict[str, float | int]] = []
    for row in nucleus_rows:
        if len(row) < 8:
            continue
        try:
            nucleus_id = int(row[0])
            root_id = int(row[3])
            x, y, z = float(row[4]), float(row[5]), float(row[6])
            nucleus_volume_um3 = float(row[7])
        except ValueError:
            continue

        if root_id not in root_set:
            continue

        center_nm = np.array([x, y, z], dtype=np.float64) * voxel_size_nm
        if not intersects_region(center_nm, center_nm, region_min, region_max):
            continue

        # Approximate radius from reported nucleus volume and guard against tiny values.
        radius_um = ((3.0 * nucleus_volume_um3) / (4.0 * np.pi)) ** (1.0 / 3.0)
        radius_um = max(radius_um, float(args.nucleus_min_radius_um))
        radius_nm = radius_um * 1000.0

        sphere = trimesh.creation.icosphere(
            subdivisions=args.nucleus_icosphere_subdivisions,
            radius=radius_nm,
        )
        sphere.apply_translation(center_nm)

        out_file = out_dir / f"nucleus_{nucleus_id}_for_{root_id}.npz"
        np.savez(out_file, vertices=sphere.vertices, faces=sphere.faces)
        added.append(
            {
                "nucleus_id": nucleus_id,
                "root_id": root_id,
                "center_x_nm": float(center_nm[0]),
                "center_y_nm": float(center_nm[1]),
                "center_z_nm": float(center_nm[2]),
                "radius_um": float(radius_um),
            }
        )

    return added


def export_meshes(
    meshes: dict[str, tuple[np.ndarray, np.ndarray]],
    export_dir: Path,
    export_format: str,
    separate: bool,
    rescale_nm_to_um: bool,
) -> None:
    export_dir.mkdir(parents=True, exist_ok=True)
    if not meshes:
        raise RuntimeError("No meshes loaded for export")

    if separate:
        for mesh_key, (vertices, faces) in sorted(meshes.items()):
            verts = np.asarray(vertices, dtype=np.float32)
            if rescale_nm_to_um:
                verts = verts / 1000.0
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            out_file = export_dir / f"mesh_{mesh_key}.{export_format}"
            mesh.export(out_file, file_type=export_format)
        return

    scene = trimesh.Scene()
    for mesh_key, (vertices, faces) in sorted(meshes.items()):
        verts = np.asarray(vertices, dtype=np.float32)
        if rescale_nm_to_um:
            verts = verts / 1000.0
        scene.add_geometry(trimesh.Trimesh(vertices=verts, faces=faces), node_name=f"mesh_{mesh_key}")
    scene.export(export_dir / f"microns_region_combined.{export_format}", file_type=export_format)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.download_dir)
    export_dir = Path(args.export_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    region_min = np.array([args.x_min_nm, args.y_min_nm, args.z_min_nm], dtype=np.float64)
    region_max = np.array([args.x_max_nm, args.y_max_nm, args.z_max_nm], dtype=np.float64)

    target_bytes = int(args.target_gb * (1024 ** 3))
    max_bytes = int(args.max_gb * (1024 ** 3))
    if max_bytes < target_bytes:
        raise ValueError("--max-gb must be >= --target-gb")

    print("Loading candidate IDs...")
    candidate_ids = enumerate_ids_from_swc_bucket(args.id_source, args.candidate_limit)
    print(f"Candidate IDs loaded: {len(candidate_ids)}")
    if args.shuffle:
        random.Random(args.seed).shuffle(candidate_ids)

    cv = CloudVolume(args.segmentation_path, progress=False, use_https=True)

    downloaded: list[dict[str, int]] = []
    failures: list[dict[str, str]] = []
    skipped_non_intersect = 0

    print(f"Download directory: {out_dir}")
    print(f"Target size: {format_gib(target_bytes)} (hard cap {format_gib(max_bytes)})")

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

        print(f"[{idx}/{len(candidate_ids)}] check root_id={root_id} ...")
        try:
            mesh_dict = cv.mesh.get(root_id, lod=args.lod)
            mesh = mesh_dict.get(root_id)
            if mesh is None:
                failures.append({"root_id": str(root_id), "error": "mesh_missing"})
                continue

            verts = np.asarray(mesh.vertices)
            min_v = verts.min(axis=0)
            max_v = verts.max(axis=0)
            if not intersects_region(min_v, max_v, region_min, region_max):
                skipped_non_intersect += 1
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
            print(f"  kept and saved: +{format_gib(added)} (total {format_gib(after_size)})")
        except Exception as exc:  # noqa: BLE001
            failures.append({"root_id": str(root_id), "error": str(exc)})

    final_size = dir_size_bytes(out_dir)
    nuclei_added: list[dict[str, float | int]] = []
    if args.include_nuclei:
        print("Adding nucleus proxy meshes for downloaded neurons...")
        downloaded_root_ids = [item["root_id"] for item in downloaded]
        nuclei_added = add_nucleus_proxy_meshes(
            neuron_root_ids=downloaded_root_ids,
            out_dir=out_dir,
            region_min=region_min,
            region_max=region_max,
            args=args,
        )
        final_size = dir_size_bytes(out_dir)
        print(f"Added nucleus proxies: {len(nuclei_added)}")

    print(f"Final download size: {format_gib(final_size)}")
    print(f"Downloaded meshes: {len(downloaded)}")
    print(f"Skipped (outside region): {skipped_non_intersect}")
    print(f"Failures: {len(failures)}")

    summary = {
        "segmentation_path": args.segmentation_path,
        "id_source": args.id_source,
        "bbox_nm": {
            "x_min": args.x_min_nm,
            "x_max": args.x_max_nm,
            "y_min": args.y_min_nm,
            "y_max": args.y_max_nm,
            "z_min": args.z_min_nm,
            "z_max": args.z_max_nm,
        },
        "lod": args.lod,
        "target_gb": args.target_gb,
        "max_gb": args.max_gb,
        "final_bytes": final_size,
        "downloaded_count": len(downloaded),
        "nucleus_proxy_count": len(nuclei_added),
        "skipped_non_intersect": skipped_non_intersect,
        "failure_count": len(failures),
        "downloaded": downloaded,
        "nucleus_proxies": nuclei_added,
        "failures": failures,
    }
    summary_file = out_dir / "download_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2))

    print("Converting downloaded meshes...")
    meshes = load_meshes_from_npz_dir(out_dir)
    export_meshes(
        meshes=meshes,
        export_dir=export_dir,
        export_format=args.format,
        separate=args.separate,
        rescale_nm_to_um=not args.no_rescale,
    )
    print(f"Exported files to: {export_dir}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()