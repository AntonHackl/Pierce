#!/usr/bin/env python3
"""Shared cube mesh helpers used by the paper dataset generators."""


def generate_cube_vertices(center_x, center_y, center_z, size):
    half_size = size / 2.0
    return [
        (center_x - half_size, center_y - half_size, center_z - half_size),
        (center_x + half_size, center_y - half_size, center_z - half_size),
        (center_x + half_size, center_y + half_size, center_z - half_size),
        (center_x - half_size, center_y + half_size, center_z - half_size),
        (center_x - half_size, center_y - half_size, center_z + half_size),
        (center_x + half_size, center_y - half_size, center_z + half_size),
        (center_x + half_size, center_y + half_size, center_z + half_size),
        (center_x - half_size, center_y + half_size, center_z + half_size),
    ]


def generate_cube_faces():
    return [
        (0, 2, 1),
        (0, 3, 2),
        (4, 5, 6),
        (4, 6, 7),
        (0, 1, 5),
        (0, 5, 4),
        (1, 2, 6),
        (1, 6, 5),
        (2, 3, 7),
        (2, 7, 6),
        (3, 0, 4),
        (3, 4, 7),
    ]


def write_obj_file(filepath, cubes_data):
    with open(filepath, "w") as output:
        output.write("# Generated cube mesh for Pierce testing\n")
        output.write(f"# Number of cubes: {len(cubes_data)}\n\n")

        vertex_offset = 0
        for cube_id, vertices, faces in cubes_data:
            output.write(f"o cube_{cube_id}\n")
            for vertex in vertices:
                output.write(
                    f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n"
                )
            for face in faces:
                output.write(
                    f"f {face[0] + vertex_offset + 1} "
                    f"{face[1] + vertex_offset + 1} "
                    f"{face[2] + vertex_offset + 1}\n"
                )
            output.write("\n")
            vertex_offset += len(vertices)
