#!/usr/bin/env python3
"""Convert MICrONS mesh subset (.npz files) to OBJ format.

Each mesh becomes a separate OBJ file (when using --separate) or combined into 
a single OBJ file. By default, coordinates are rescaled from nanometers to 
micrometers for better compatibility with 3D tools like Blender.

Usage:
  # Convert local .npz files to a directory of separate OBJ files
  python convert_microns_to_obj.py --input-dir ./microns_mesh_subset_lod0 --output ./microns_obj --separate

  # Convert local .npz files to a single combined OBJ
  python convert_microns_to_obj.py --input-dir ./microns_mesh_subset_lod0 --output microns_combined.obj
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

try:
    import trimesh
except ImportError:
    raise ImportError("trimesh is required. Install with: pip install trimesh")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing .npz mesh files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output OBJ filename (if combined) or output directory (if --separate).",
    )
    parser.add_argument(
        "--separate",
        action="store_true",
        help="Export each mesh as a separate OBJ file in the --output directory.",
    )
    parser.add_argument(
        "--rescale-nm-to-um",
        action="store_true",
        default=True,
        help="Rescale from nanometers to micrometers (default: True).",
    )
    parser.add_argument(
        "--no-rescale",
        action="store_false",
        dest="rescale_nm_to_um",
        help="Do not rescale coordinates.",
    )
    return parser.parse_args()


def load_npz_mesh(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from an .npz file."""
    data = np.load(npz_path)
    vertices = data["vertices"]
    faces = data["faces"]
    return vertices, faces


def export_meshes_individually(
    input_dir: Path,
    output_dir: Path,
    rescale: bool = True,
) -> None:
    """Load .npz files one by one and export as separate OBJ files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(input_dir.glob("*.npz"))
    
    print(f"Found {len(npz_files)} .npz files in {input_dir}")
    print(f"Exporting individually to {output_dir}...")
    if rescale:
        print("  (rescaling from nanometers to micrometers)")

    for npz_file in npz_files:
        try:
            root_id = npz_file.stem
            vertices, faces = load_npz_mesh(npz_file)
            
            verts = np.asarray(vertices, dtype=np.float32)
            if rescale:
                verts = verts / 1000.0
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            obj_file = output_dir / f"{root_id}.obj"
            mesh.export(obj_file)
            
            file_size_mb = obj_file.stat().st_size / (1024 ** 2)
            print(f"  ✓ {obj_file.name} ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Failed to convert {npz_file.name}: {e}")

    print(f"\n✓ Done! All files exported to {output_dir}")


def export_meshes_combined(
    input_dir: Path,
    output_file: Path,
    rescale: bool = True,
) -> None:
    """Combine all .npz files into a single OBJ file."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(input_dir.glob("*.npz"))
    
    print(f"Found {len(npz_files)} .npz files in {input_dir}")
    print(f"Combining into {output_file}...")
    if rescale:
        print("  (rescaling from nanometers to micrometers)")

    scene = trimesh.Scene()
    for npz_file in npz_files:
        try:
            root_id = npz_file.stem
            vertices, faces = load_npz_mesh(npz_file)
            
            verts = np.asarray(vertices, dtype=np.float32)
            if rescale:
                verts = verts / 1000.0
            
            mesh = trimesh.Trimesh(vertices=verts, faces=faces)
            scene.add_geometry(mesh, node_name=f"neuron_{root_id}")
            print(f"  added {root_id}")
        except Exception as e:
            print(f"  ✗ Failed to load {npz_file.name}: {e}")

    print(f"Exporting scene to {output_file} (this may take a while for large datasets)...")
    # For OBJ, trimesh scene export will include all geometries
    scene.export(output_file)
    print(f"✓ Done! Combined OBJ saved to {output_file}")


def main() -> None:
    args = parse_args()
    
    if not args.input_dir.exists():
        print(f"Error: Input directory {args.input_dir} does not exist.")
        return

    if args.separate:
        export_meshes_individually(args.input_dir, args.output, args.rescale_nm_to_um)
    else:
        export_meshes_combined(args.input_dir, args.output, args.rescale_nm_to_um)


if __name__ == "__main__":
    main()
