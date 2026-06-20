#!/usr/bin/env python3
"""
Generate cube datasets with a TARGET selectivity.
Given the cube sizes and desired selectivity, computes the appropriate universe extent.
"""

import argparse
import random
import math
from pathlib import Path
import sys
from typing import List, Tuple, Optional

# Import from generate_test_cubes
sys.path.append(str(Path(__file__).parent))
from generate_test_cubes import generate_cube_vertices, generate_cube_faces, write_obj_file


def compute_universe_for_selectivity(target_selectivity, min_size, max_size):
    """
    Compute universe extent (assuming cubic universe) to achieve target selectivity.
    
    For overlap (excluding containment), the selectivity approximately follows:
        selectivity ≈ [(2 * avg_size) / U]^3
    
    Solving for U:
        U ≈ (2 * avg_size) / selectivity^(1/3)
    
    Returns:
        Universe extent U (same for all dimensions)
    """
    avg_size = (min_size + max_size) / 2.0
    
    # Solve: target_selectivity = [(2 * avg_size) / U]^3
    # U = (2 * avg_size) / target_selectivity^(1/3)
    
    if target_selectivity <= 0:
        raise ValueError("Target selectivity must be positive")
    
    universe_extent = (2.0 * avg_size) / (target_selectivity ** (1.0/3.0))
    
    return universe_extent


def verify_selectivity_on_data(cubes_a, cubes_b, universe_extent, samples=100000):
    """
    Verify selectivity by sampling pairs directly from generated datasets.
    """
    matches = 0
    num_a = len(cubes_a)
    num_b = len(cubes_b)
    
    for _ in range(samples):
        # Pick random cube from A and B
        idx_a = random.randint(0, num_a - 1)
        idx_b = random.randint(0, num_b - 1)
        
        _, v_a, _ = cubes_a[idx_a]
        _, v_b, _ = cubes_b[idx_b]
        
        # In generate_cube_vertices, v[0] and v[6] are opposite corners
        # Corner 0: (min, min, min), Corner 6: (max, max, max)
        min_a = v_a[0]
        max_a = v_a[6]
        min_b = v_b[0]
        max_b = v_b[6]
        
        # AABB Overlap check
        overlap = True
        for i in range(3):
            if min_a[i] >= max_b[i] or min_b[i] >= max_a[i]:
                overlap = False
                break
        
        if not overlap:
            continue
            
        # Containment check (exclude if one strictly contains another)
        # A contains B?
        a_contains_b = True
        for i in range(3):
            if min_a[i] > min_b[i] or max_a[i] < max_b[i]:
                a_contains_b = False
                break
        
        if a_contains_b:
            continue
            
        # B contains A?
        b_contains_a = True
        for i in range(3):
            if min_b[i] > min_a[i] or max_b[i] < max_a[i]:
                b_contains_a = False
                break
                
        if b_contains_a:
            continue
            
        matches += 1
    
    return matches / samples


BBox = Tuple[float, float, float, float, float, float]  # (minx, miny, minz, maxx, maxy, maxz)


def verify_selectivity_on_bboxes(bboxes_a: List[BBox], bboxes_b: List[BBox], samples: int = 100000) -> float:
    """Verify selectivity by sampling pairs from lists of axis-aligned bounding boxes.

    Uses the same overlap-without-containment predicate as the original implementation.
    """
    if not bboxes_a or not bboxes_b:
        return 0.0

    matches = 0
    num_a = len(bboxes_a)
    num_b = len(bboxes_b)

    for _ in range(samples):
        min_ax, min_ay, min_az, max_ax, max_ay, max_az = bboxes_a[random.randint(0, num_a - 1)]
        min_bx, min_by, min_bz, max_bx, max_by, max_bz = bboxes_b[random.randint(0, num_b - 1)]

        # Overlap check
        if min_ax >= max_bx or min_bx >= max_ax:
            continue
        if min_ay >= max_by or min_by >= max_ay:
            continue
        if min_az >= max_bz or min_bz >= max_az:
            continue

        # Containment check (exclude if one strictly contains another)
        a_contains_b = (min_ax <= min_bx and min_ay <= min_by and min_az <= min_bz and
                        max_ax >= max_bx and max_ay >= max_by and max_az >= max_bz)
        if a_contains_b:
            continue

        b_contains_a = (min_bx <= min_ax and min_by <= min_ay and min_bz <= min_az and
                        max_bx >= max_ax and max_by >= max_ay and max_bz >= max_az)
        if b_contains_a:
            continue

        matches += 1

    return matches / samples


def _reservoir_maybe_add(sample: List[BBox], bbox: BBox, i_one_based: int, k: int) -> None:
    """Reservoir sampling: keep a uniform sample of size k from a stream."""
    if k <= 0:
        return
    if i_one_based <= k:
        sample.append(bbox)
        return
    j = random.randint(1, i_one_based)
    if j <= k:
        sample[j - 1] = bbox


def write_obj_file_streaming(
    filepath: Path,
    num_cubes: int,
    min_size: float,
    max_size: float,
    extent: float,
    seed: Optional[int],
    reservoir_size: int,
    progress_every: int = 100000,
) -> List[BBox]:
    """Stream-generate cubes directly into an OBJ, while keeping a reservoir sample of bboxes."""
    if seed is not None:
        random.seed(seed)

    cube_faces = generate_cube_faces()
    reservoir: List[BBox] = []

    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("# Generated cube mesh for Pierce testing\n")
        f.write(f"# Number of cubes: {num_cubes}\n\n")

        vertex_offset = 0

        for i in range(num_cubes):
            center_x = random.uniform(0, extent)
            center_y = random.uniform(0, extent)
            center_z = random.uniform(0, extent)
            size = random.uniform(min_size, max_size)
            half = size / 2.0

            minx = center_x - half
            miny = center_y - half
            minz = center_z - half
            maxx = center_x + half
            maxy = center_y + half
            maxz = center_z + half

            _reservoir_maybe_add(reservoir, (minx, miny, minz, maxx, maxy, maxz), i + 1, reservoir_size)

            vertices = generate_cube_vertices(center_x, center_y, center_z, size)

            f.write(f"o cube_{i}\n")
            for v in vertices:
                f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
            for face in cube_faces:
                f.write(
                    f"f {face[0] + vertex_offset + 1} {face[1] + vertex_offset + 1} {face[2] + vertex_offset + 1}\n"
                )
            f.write("\n")
            vertex_offset += 8

            if progress_every > 0 and (i + 1) % progress_every == 0:
                print(f"  ... wrote {i + 1:,} / {num_cubes:,} cubes to {filepath.name}")

    return reservoir


def generate_random_cube_data(num_cubes, min_size, max_size, extent, seed=None):
    """Generate random cube data within cubic extent."""
    if seed is not None:
        random.seed(seed)
    
    cubes_data = []
    cube_faces = generate_cube_faces()
    
    for i in range(num_cubes):
        center_x = random.uniform(0, extent)
        center_y = random.uniform(0, extent)
        center_z = random.uniform(0, extent)
        size = random.uniform(min_size, max_size)
        
        vertices = generate_cube_vertices(center_x, center_y, center_z, size)
        cubes_data.append((i, vertices, cube_faces))
    
    return cubes_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate cube datasets with target selectivity',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--num-cubes-a', '-na', type=int, required=True,
                        help='Number of cubes in dataset A')
    parser.add_argument('--num-cubes-b', '-nb', type=int, required=True,
                        help='Number of cubes in dataset B')
    parser.add_argument('--min-size', type=float, required=True,
                        help='Minimum cube edge length')
    parser.add_argument('--max-size', type=float, required=True,
                        help='Maximum cube edge length')
    parser.add_argument('--selectivity', '-s', type=float, required=True,
                        help='Target selectivity (probability that a random pair overlaps)')
    parser.add_argument('--output-a', '-oa', type=str, default='generated_files/Query/cubes_a.obj',
                        help='Output path for dataset A')
    parser.add_argument('--output-b', '-ob', type=str, default='generated_files/Query/cubes_b.obj',
                        help='Output path for dataset B')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    parser.add_argument('--verify-samples', type=int, default=100000,
                        help='Number of samples for verification (default: 100000)')
    parser.add_argument('--streaming-threshold', type=int, default=250000,
                        help='If num-cubes exceeds this, generate OBJ via streaming to reduce RAM')
    parser.add_argument('--reservoir-size', type=int, default=100000,
                        help='Reservoir sample size for on-data verification in streaming mode')
    parser.add_argument('--progress-every', type=int, default=100000,
                        help='Progress interval (cubes) for streaming writer')
    
    args = parser.parse_args()
    
    # Validate
    if args.num_cubes_a <= 0 or args.num_cubes_b <= 0:
        parser.error("Number of cubes must be positive")
    if args.min_size <= 0 or args.max_size <= 0:
        parser.error("Cube sizes must be positive")
    if args.min_size > args.max_size:
        parser.error("Minimum size cannot be greater than maximum size")
    if args.selectivity <= 0 or args.selectivity > 1:
        parser.error("Selectivity must be between 0 and 1")
    
    # Compute universe extent
    print(f"Computing universe extent for target selectivity {args.selectivity:.6f}...")
    universe_extent = compute_universe_for_selectivity(args.selectivity, args.min_size, args.max_size)
    
    print(f"\n{'=' * 60}")
    print(f"COMPUTED UNIVERSE EXTENT:")
    print(f"{'=' * 60}")
    print(f"  Target Selectivity:    {args.selectivity:.8f}")
    print(f"  Cube Size Range:       [{args.min_size}, {args.max_size}]")
    print(f"  Average Cube Size:     {(args.min_size + args.max_size)/2:.2f}")
    print(f"  Computed Universe:     {universe_extent:.2f} × {universe_extent:.2f} × {universe_extent:.2f}")
    print(f"{'=' * 60}")
    
    # Create output directories
    output_path_a = Path(args.output_a)
    output_path_b = Path(args.output_b)
    output_path_a.parent.mkdir(parents=True, exist_ok=True)
    output_path_b.parent.mkdir(parents=True, exist_ok=True)
    
    streaming_mode = (args.num_cubes_a > args.streaming_threshold) or (args.num_cubes_b > args.streaming_threshold)

    seed_b = (args.seed + 1) if args.seed is not None else None

    if streaming_mode:
        print(
            "\nStreaming mode enabled (avoids holding all cubes in RAM). "
            f"Verification uses reservoir samples of size {args.reservoir_size:,}."
        )

        print(f"\nWriting dataset A to (streaming): {output_path_a}")
        bboxes_a = write_obj_file_streaming(
            output_path_a,
            args.num_cubes_a,
            args.min_size,
            args.max_size,
            universe_extent,
            args.seed,
            reservoir_size=args.reservoir_size,
            progress_every=args.progress_every,
        )
        print(f"Successfully wrote {args.num_cubes_a:,} cubes")

        print(f"\nWriting dataset B to (streaming): {output_path_b}")
        bboxes_b = write_obj_file_streaming(
            output_path_b,
            args.num_cubes_b,
            args.min_size,
            args.max_size,
            universe_extent,
            seed_b,
            reservoir_size=args.reservoir_size,
            progress_every=args.progress_every,
        )
        print(f"Successfully wrote {args.num_cubes_b:,} cubes")

        print(f"\nVerifying selectivity on generated data (reservoir samples) with {args.verify_samples:,} samples...")
        verified_selectivity = verify_selectivity_on_bboxes(bboxes_a, bboxes_b, samples=args.verify_samples)
    else:
        # Generate datasets (in-memory)
        print(f"Generating dataset A ({args.num_cubes_a:,} cubes)...")
        cubes_data_a = generate_random_cube_data(
            args.num_cubes_a,
            args.min_size,
            args.max_size,
            universe_extent,
            args.seed
        )
        
        print(f"Generating dataset B ({args.num_cubes_b:,} cubes)...")
        cubes_data_b = generate_random_cube_data(
            args.num_cubes_b,
            args.min_size,
            args.max_size,
            universe_extent,
            seed_b
        )

        # Verify with sampling ON THE GENERATED DATA
        print(f"\nVerifying selectivity on generated data with {args.verify_samples:,} samples...")
        verified_selectivity = verify_selectivity_on_data(cubes_data_a, cubes_data_b, universe_extent, args.verify_samples)
    
    print(f"\n{'=' * 60}")
    print(f"SELECTIVITY VERIFICATION (on generated data):")
    print(f"{'=' * 60}")
    print(f"  Target Selectivity:    {args.selectivity:.8f}")
    print(f"  Verified Selectivity:  {verified_selectivity:.8f}")
    print(f"  Relative Error:        {abs(verified_selectivity - args.selectivity) / args.selectivity * 100:.2f}%")
    print(f"{'=' * 60}")
    
    # Compute expected intersections
    total_pairs = args.num_cubes_a * args.num_cubes_b
    expected_intersections = verified_selectivity * total_pairs
    
    print(f"\n{'=' * 60}")
    print(f"EXPECTED JOIN STATISTICS:")
    print(f"{'=' * 60}")
    print(f"  Dataset A Size:        {args.num_cubes_a:,}")
    print(f"  Dataset B Size:        {args.num_cubes_b:,}")
    print(f"  Total Pairs:           {total_pairs:,}")
    print(f"  Expected Intersections: {expected_intersections:,.0f}")
    print(f"{'=' * 60}\n")
    
    # Write outputs for in-memory mode
    if not streaming_mode:
        print(f"\nWriting dataset A to: {output_path_a}")
        write_obj_file(output_path_a, cubes_data_a)
        print(f"Successfully wrote {args.num_cubes_a:,} cubes")
        
        print(f"\nWriting dataset B to: {output_path_b}")
        write_obj_file(output_path_b, cubes_data_b)
        print(f"Successfully wrote {args.num_cubes_b:,} cubes")
    
    print(f"\n{'=' * 60}")
    print(f"GENERATION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"Universe: [0, {universe_extent:.2f}]³")
    print(f"Dataset A: {output_path_a}")
    print(f"Dataset B: {output_path_b}")
    print(f"{'=' * 60}")


if __name__ == '__main__':
    main()
