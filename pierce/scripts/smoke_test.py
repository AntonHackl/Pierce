#!/usr/bin/env python3
"""Exercise Pierce preprocessing, queries, and Python runners without baselines."""

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.adapters.base import run_command_streaming
from benchmarks.common.executables import pierce_executable
from pierce.scripts.generate_test_cubes import generate_cube_faces, generate_cube_vertices, write_obj_file


def make_cubes(specs):
    return [(i, generate_cube_vertices(*center, size), generate_cube_faces())
            for i, (center, size) in enumerate(specs)]


def write_raw_dt(path, cubes, large=False):
    """Write the 64-bit little-endian RAW DT format consumed by Pierce."""
    with path.open("wb") as output:
        if large and os.name == "nt":
            import ctypes
            from ctypes import wintypes
            import msvcrt

            control = ctypes.WinDLL("kernel32", use_last_error=True).DeviceIoControl
            control.argtypes = [wintypes.HANDLE, wintypes.DWORD, ctypes.c_void_p,
                                wintypes.DWORD, ctypes.c_void_p, wintypes.DWORD,
                                ctypes.POINTER(wintypes.DWORD), ctypes.c_void_p]
            control.restype = wintypes.BOOL
            returned = wintypes.DWORD()
            if not control(msvcrt.get_osfhandle(output.fileno()), 0x900C4,
                           None, 0, None, 0, ctypes.byref(returned), None):
                raise ctypes.WinError(ctypes.get_last_error())
        output.write(b"\x01")  # RAW
        for object_id, vertices, faces in cubes:
            triangles = [coordinate for face in faces for vertex in face for coordinate in vertices[vertex]]
            data = struct.pack(f"<{len(triangles)}f", *triangles)
            data += bytes(len(faces) * 2 * 4)  # Hausdorff distances
            payload_size = len(data) * 5
            if large and object_id == 0:
                payload_size = max(payload_size, (1 << 31) + 4096)
            output.write(struct.pack("<Q", payload_size))
            payload_start = output.tell()
            for _ in range(5):
                output.write(data)
            output.seek(payload_start + payload_size)
            output.write(struct.pack("<Q10f", 1, *([0.0] * 10)))
            low = [min(vertex[d] for vertex in vertices) for d in range(3)]
            high = [max(vertex[d] for vertex in vertices) for d in range(3)]
            core = [(low[d] + high[d]) / 2 for d in range(3)]
            output.write(struct.pack("<9f", *(low + high + core)))
            for lod in range(5):
                output.write(struct.pack("<QQ", lod * len(data), len(faces)))


def check_streaming(directory):
    log = directory / "streaming.log"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        stdout, stderr = run_command_streaming(
            [sys.executable, "-c", "import sys; print('out'); print('err', file=sys.stderr)"],
            timeout=10, log_path=str(log), prefix="[test]")
        assert stdout.strip() == "out" and stderr.strip() == "err"
        assert "[test] out" in log.read_text() and "[test] err" in log.read_text()
        try:
            run_command_streaming([sys.executable, "-c", "raise SystemExit(7)"], timeout=10)
        except subprocess.CalledProcessError as error:
            assert error.returncode == 7
        else:
            raise AssertionError("Nonzero child exit was not reported")
        for code in ("import time; time.sleep(10)",
                     "import os,time; os.close(1); os.close(2); time.sleep(10)"):
            try:
                run_command_streaming([sys.executable, "-c", code], timeout=0.3)
            except subprocess.TimeoutExpired:
                pass
            else:
                raise AssertionError("Child timeout was not enforced")
    print("PASS Python stdout/stderr streaming, logging, errors, and timeouts")


def read_pairs(path):
    with path.open(newline="") as stream:
        return {(int(row[0]), int(row[1])) for row in list(csv.reader(stream))[1:]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "tmp" / "windows-smoke")
    parser.add_argument("--large-dt", action="store_true", help="Also preprocess a sparse DT file above 2 GiB")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    pierce_dir = REPO_ROOT / "pierce"
    preprocess = pierce_executable(pierce_dir, "preprocess", "pierce_preprocess")
    executables = ["pierce_overlap_two_pass", "pierce_overlap", "pierce_intersection", "pierce_containment"]
    for executable in [preprocess] + [pierce_executable(pierce_dir, "query", name) for name in executables]:
        if not executable.is_file():
            raise FileNotFoundError(executable)
    check_streaming(directory)

    a = make_cubes([((0, 0, 0), 4), ((10, 0, 0), 2)])
    b = make_cubes([((1.7, 0.4, 0.2), 2), ((0.2, 0.1, 0.3), 0.6), ((20, 0, 0), 1)])
    far = make_cubes([((30, 0, 0), 1)])
    for name, cubes in (("a", a), ("b", b), ("far", far)):
        write_obj_file(directory / f"{name}.obj", cubes)
        write_raw_dt(directory / f"{name}.dt", cubes)

    def run(command, name, cwd=directory):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            stdout, stderr = run_command_streaming([str(part) for part in command], timeout=120,
                                                  log_path=str(directory / f"{name}.log"), cwd=str(cwd))
        return stdout, stderr

    def preprocess_input(source, name):
        target = directory / f"{name}.pre"
        run([preprocess, "--dataset", source, "--output-geometry", target,
             "--output-timing", directory / f"{name}.json", "--generate-grid", "--grid-cell-size", "2"], name)
        assert target.is_file() and target.stat().st_size > 64
        return target

    sources = {}
    for extension in ("obj", "dt"):
        for name in ("a", "b", "far"):
            sources[name, extension] = preprocess_input(directory / f"{name}.{extension}", f"{name}_{extension}")
    print("PASS OBJ and RAW DT preprocessing")

    expected = [{(0, 0)}, {(0, 0)}, {(0, 0), (0, 1)}, {(0, 1)}]
    filenames = ["mesh_overlap_results.csv", "pairs.csv", "pairs.csv", "mesh_containment_results.csv"]
    for extension in ("obj", "dt"):
        for disjoint in (False, True):
            for name, pairs, filename in zip(executables, expected, filenames):
                case = f"{name}_{extension}_{'disjoint' if disjoint else 'mixed'}"
                case_dir = directory / case
                case_dir.mkdir(exist_ok=True)
                command = [pierce_executable(pierce_dir, "query", name),
                           "--mesh1", sources["a", extension],
                           "--mesh2", sources["far" if disjoint else "b", extension],
                           "--output", case_dir / "timing.json", "--runs", "1", "--warmup-runs", "0"]
                if name in ("pierce_overlap", "pierce_intersection"):
                    command += ["--pairs-output", case_dir / filename]
                if name == "pierce_overlap":
                    command += ["--hash-table-size", "4096"]
                run(command, case, cwd=case_dir)
                actual = read_pairs(case_dir / filename)
                assert actual == (set() if disjoint else pairs), f"{case}: {actual} != {pairs}"
                assert json.loads((case_dir / "timing.json").read_text())
                print(f"PASS {case}: {sorted(actual)}")

    # Exercise the actual Python adapter entry points and executable resolution.
    from benchmarks.overlap.adapters.pierce_adapter import PierceAdapter
    from benchmarks.predicates.adapters.intersection import PierceIntersectionAdapter
    from benchmarks.predicates.adapters.containment import PierceContainmentAdapter
    adapter_dir = directory / "adapters"
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for adapter_class, method, options, count_key, count in (
                (PierceAdapter, "run_overlap", {"mode": "exact"}, "num_intersections", 1),
                (PierceAdapter, "run_overlap", {"mode": "direct_estimation", "hash_table_size": 4096}, "num_intersections", 1),
                (PierceIntersectionAdapter, "run_intersection", {}, "num_intersections", 2),
                (PierceContainmentAdapter, "run_containment", {}, "num_containments", 1)):
            adapter = adapter_class(str(pierce_dir), preprocessed_dir=str(adapter_dir),
                                    timings_dir=str(adapter_dir), grid_cell_size=2, warmup_runs=0, **options)
            for name in ("a", "b"):
                adapter.preprocess_from_source(str(directory / f"{name}.obj"), str(directory / f"{name}.obj"),
                                               log_dir=str(adapter_dir))
            old_cwd = Path.cwd()
            try:
                os.chdir(adapter_dir)
                result = getattr(adapter, method)(str(directory / "a.obj"), str(directory / "b.obj"),
                                                  num_runs=1, timeout=120, log_dir=str(adapter_dir))
            finally:
                os.chdir(old_cwd)
            assert "error" not in result, result
            assert result[count_key] == count, result
    print("PASS two-pass/estimated overlap, intersection, and containment Python adapters")

    if args.large_dt:
        large = directory / "large.dt"
        write_raw_dt(large, a[:1], large=True)
        assert large.stat().st_size > (1 << 31)
        output = preprocess_input(large, "large_dt")
        with output.open("rb") as stream:
            magic, version, vertices, triangles = struct.unpack("<IIQQ", stream.read(24))
        assert magic == 0x52334442 and version == 4 and vertices == 8 and triangles == 12
        print(f"PASS DT above 2 GiB ({large.stat().st_size} bytes): 8 vertices, 12 triangles")
    print(f"Pierce smoke checks passed. Logs: {directory}")


if __name__ == "__main__":
    main()
