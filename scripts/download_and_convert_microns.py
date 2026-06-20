#!/usr/bin/env python3
"""Standalone script to download MICrONS meshes and convert to Blender GLB format.

No dependencies on the RaySpace repo—just install prerequisites and run.

Installation:
  pip install cloud-volume trimesh numpy

Usage:
  # Download 18 meshes and export to single GLB file
  python download_and_convert_microns.py --output microns_neurons.glb

  # Or download to a directory first, then convert separately
  python download_and_convert_microns.py --download-only --output-dir ./meshes_npz
  python download_and_convert_microns.py --convert-only --input-dir ./meshes_npz --output microns.glb
"""

from __future__ import annotations

import argparse
import colorsys
import json
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("trimesh required: pip install trimesh")


DEFAULT_MESH_SOURCE = "precomputed://gs://iarpa_microns/minnie/minnie65/seg_m1300"
DEFAULT_LOD = 0
DEFAULT_ROOT_IDS = [
    864691135569592300,
    864691135685661367,
    864691135361291591,
    864691136813553523,
    864691136025333561,
    864691135463999294,
    864691136335276211,
    864691135430460720,
    864691135777381805,
    864691135777521837,
    864691136052291827,
    864691135688375264,
    864691136012739747,
    864691135503367517,
    864691135114295961,
    864691136663371742,
    864691135359010904,
    864691135341516741,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download-and-convert",
        action="store_true",
        default=True,
        help="Download from GCS and convert to GLB (default).",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="Only download .npz files, save to --output-dir.",
    )
    parser.add_argument(
        "--convert-only",
        action="store_true",
        help="Only convert existing .npz files in --input-dir to GLB.",
    )
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "microns_data" / "microns_meshes_npz",
        help="Directory to store downloaded .npz files.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="Directory with .npz files to convert (used with --convert-only).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "microns_data" / "microns_neurons.glb",
        help="Output GLB filename.",
    )
    parser.add_argument(
        "--mesh-source",
        default=DEFAULT_MESH_SOURCE,
        help="CloudVolume segmentation source.",
    )
    parser.add_argument(
        "--lod",
        type=int,
        default=DEFAULT_LOD,
        help="Mesh level of detail.",
    )
    parser.add_argument(
        "--root-ids",
        nargs="+",
        type=int,
        default=DEFAULT_ROOT_IDS,
        help="Root IDs to download.",
    )
    return parser.parse_args()


def download_meshes(
    root_ids: list[int],
    mesh_source: str,
    lod: int,
    output_dir: Path,
) -> None:
    """Download meshes from CloudVolume and save as .npz files."""
    try:
        from cloudvolume import CloudVolume
    except ImportError:
        raise ImportError("cloudvolume required: pip install cloud-volume")

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(root_ids)} meshes from {mesh_source} (LOD={lod})...")
    print(f"Saving to {output_dir}\n")

    cv = CloudVolume(mesh_source, progress=False, use_https=True)
    succeeded = 0
    failed = 0

    for root_id in root_ids:
        try:
            mesh_dict = cv.mesh.get(root_id, lod=lod)
            mesh = mesh_dict.get(root_id)
            if mesh is None:
                print(f"  {root_id}: no mesh returned")
                failed += 1
                continue

            npz_file = output_dir / f"{root_id}.npz"
            np.savez(npz_file, vertices=mesh.vertices, faces=mesh.faces)
            succeeded += 1
            print(f"  {root_id}: {mesh.vertices.shape[0]} verts, {mesh.faces.shape[0]} faces -> {npz_file.name}")
        except Exception as e:
            print(f"  {root_id}: {e}")
            failed += 1

    print(f"\nDownloaded: {succeeded}, Failed: {failed}")


def load_npz_mesh(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from .npz file."""
    data = np.load(npz_path)
    return data["vertices"], data["faces"]


def load_meshes_from_dir(input_dir: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load all .npz files from directory."""
    meshes = {}
    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"Loading {len(npz_files)} .npz files from {input_dir}...")
    for npz_file in npz_files:
        try:
            root_id = int(npz_file.stem)
            vertices, faces = load_npz_mesh(npz_file)
            meshes[root_id] = (vertices, faces)
            print(f"  {root_id}: {vertices.shape[0]} verts, {faces.shape[0]} faces")
        except (ValueError, KeyError) as e:
            print(f"  {npz_file.name}: {e}")
    return meshes


def generate_colors(count: int) -> list[tuple[float, float, float]]:
    """Generate N distinct colors using HSL space."""
    colors = []
    for i in range(count):
        hue = i / count
        saturation = 0.7 + 0.2 * (i % 2)
        lightness = 0.5
        rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
        colors.append(rgb)
    return colors


def combine_meshes_to_glb(
    meshes: dict[int, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    """Combine meshes into a single GLB with per-mesh segmentation colors."""
    if not meshes:
        raise ValueError("No meshes to export")

    root_ids = sorted(meshes.keys())
    colors = generate_colors(len(root_ids))

    print(f"\nCombining {len(root_ids)} meshes into GLB...")
    scene = trimesh.Scene()

    for (root_id, (vertices, faces)), color in zip(meshes.items(), colors):
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        # Assign color to all vertices of this mesh
        mesh.visual.vertex_colors = [
            tuple(int(c * 255) for c in color) + (255,) for _ in range(len(vertices))
        ]
        scene.add_geometry(mesh, node_name=f"neuron_{root_id}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Exporting to {output_path}...")
    scene.export(output_path, file_type="glb")

    file_size_mb = output_path.stat().st_size / (1024 ** 2)
    print(f"✓ Saved {output_path} ({file_size_mb:.1f} MB)")
    print(f"\n✓ Open {output_path} in Blender!")
    print("  Each neuron is a separate object with a unique color for segmentation.")


def main() -> None:
    args = parse_args()

    if args.convert_only:
        input_dir = args.input_dir or args.output_dir
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        meshes = load_meshes_from_dir(input_dir)
        combine_meshes_to_glb(meshes, args.output)
    elif args.download_only:
        download_meshes(args.root_ids, args.mesh_source, args.lod, args.output_dir)
        print(f"\nTo convert to GLB, run:")
        print(f"  python download_and_convert_microns.py --convert-only --input-dir {args.output_dir} --output {args.output}")
    else:
        download_meshes(args.root_ids, args.mesh_source, args.lod, args.output_dir)
        meshes = load_meshes_from_dir(args.output_dir)
        combine_meshes_to_glb(meshes, args.output)


if __name__ == "__main__":
    main()
