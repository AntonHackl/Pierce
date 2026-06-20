import subprocess
import time
import json
import re
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path
from benchmarks.common.adapters.base import IntersectionBenchmarkAdapter, run_command_streaming


class PierceIntersectionAdapter(IntersectionBenchmarkAdapter):
    def __init__(
        self,
        pierce_dir: str,
        mode: str = "estimated",
        preprocessed_dir: str = "preprocessed",
        timings_dir: str = "timings",
        grid_cell_size: int = 10,
        warmup_runs: int = 2,
    ):
        normalized_mode = "estimated" if mode == "two_pass" else mode
        super().__init__(f"Pierce_{normalized_mode}")
        self.pierce_dir = Path(pierce_dir)
        self.mode = normalized_mode
        self.preprocessed_dir = Path(preprocessed_dir)
        self.timings_dir = Path(timings_dir)
        self.grid_cell_size = grid_cell_size
        self.warmup_runs = warmup_runs

        self.timings_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

        query_bin_dir = self.pierce_dir / "query" / "build" / "bin"
        if self.mode in ("estimated", "estimate_only"):
            self.executable = query_bin_dir / "pierce_intersection"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        self.preprocess_exec = self.pierce_dir / "preprocess" / "build" / "bin" / "pierce_preprocess"

    def _get_preprocessed_path(self, file_path: str) -> Path:
        input_path = Path(file_path)
        # Use a token that includes the grid size to ensure we re-preprocess if it changes
        grid_token = str(self.grid_cell_size).replace(".", "_")
        return self.preprocessed_dir / f"{input_path.stem}_g{grid_token}.pre"

    def check_preprocessed(self, file_path: str) -> bool:
        return self._get_preprocessed_path(file_path).exists()

    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        source_path = Path(source_file)
        dt_path = Path(dt_file)

        output_geometry = self._get_preprocessed_path(dt_file)
        output_timing = self.timings_dir / (dt_path.stem + f'_g{str(self.grid_cell_size).replace(".", "_")}_timing.json')

        mode = "dt" if source_path.suffix == ".dt" else "mesh"

        cmd = [
            str(self.preprocess_exec),
            "--mode", mode,
            "--dataset", str(source_path),
            "--output-geometry", str(output_geometry),
            "--output-timing", str(output_timing),
            "--generate-grid",
            "--grid-cell-size", str(self.grid_cell_size)
        ]

        print(f"[{self.name}] Preprocessing {source_path.name} (output: {dt_path.name}) with grid (resolution={self.grid_cell_size})...")
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = adapter_log_dir / f"preprocess_{dt_path.stem}_{int(time.time())}.log"
            run_command_streaming(cmd, timeout=None, log_path=str(log_path), prefix=f"[{self.name}]")
        else:
            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")

    def run_intersection(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
        extra_args: Optional[list[str]] = None,
    ) -> Dict[str, Any]:
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        p1 = self._get_preprocessed_path(file1)
        p2 = self._get_preprocessed_path(file2)

        f1 = str(p1) if p1.exists() else file1
        f2 = str(p2) if p2.exists() else file2

        runtimes = []
        breakdown_accum = {}
        num_obj1 = 0
        num_obj2 = 0
        num_intersections = 0
        universe_extents1 = [0.0, 0.0, 0.0]
        universe_extents2 = [0.0, 0.0, 0.0]

        print(f"[{self.name}] Running benchmark...")

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "estimated":
            expected_prefixes = [
                "selectivity estimation",
                "raytrace_hash_",
                "raytrace_overlap_hash_",
                "raytrace_containment_hash_",
                "execute hash query",
                "download results",
                "query",
                "compact_hash_table_pairs",
            ]
        else:
            expected_prefixes = ["selectivity estimation"]

        for run_idx in range(num_runs):
            json_output = self.timings_dir / f"timing_{self.mode}_{int(time.time())}_{run_idx}.json"

            cmd = [
                str(self.executable),
                "--mesh1", f1,
                "--mesh2", f2,
                "--output", str(json_output)
            ]

            if extra_args:
                cmd.extend(extra_args)

            if self.mode == "estimate_only":
                cmd.append("--estimate-only")

            try:
                log_path = None
                if adapter_log_dir is not None:
                    log_path = str(adapter_log_dir / f"run_{run_idx:03d}.log")

                stdout_text, stderr_text = run_command_streaming(
                    cmd,
                    timeout=timeout,
                    log_path=log_path,
                    prefix=f"[{self.name}]",
                )

                # Parse summary from stdout on the first run
                if run_idx == 0:
                    lines = stdout_text.splitlines()
                    def parse_vec3(l):
                        return [float(p.strip()) for p in l.split("[")[1].split("]")[0].split(",")]

                    for i, line in enumerate(lines):
                        if "Mesh1 objects:" in line:
                            num_obj1 = int(line.split(":")[1].strip())
                        elif "Mesh2 objects:" in line:
                            num_obj2 = int(line.split(":")[1].strip())
                        elif "Unique intersecting object pairs:" in line:
                            num_intersections = int(line.split(":")[1].strip())
                        elif "Actual Intersection Pairs:" in line:
                            num_intersections = int(line.split(":")[1].strip())
                        elif "Hash Table Query found" in line:
                            num_intersections = int(line.split("found")[1].split("unique")[0].strip())
                        elif "Final Estimated Pairs:" in line:
                            try:
                                num_intersections = int(line.split(":", 1)[1].strip())
                            except ValueError: pass
                        elif "Universe Extents:" in line:
                            try:
                                ext = parse_vec3(line)
                                universe_extents1 = ext
                                universe_extents2 = ext
                            except Exception: pass
                        elif "Mesh1 Universe Min:" in line and (i+1) < len(lines) and "Mesh1 Universe Max:" in lines[i+1]:
                            try:
                                v_min = parse_vec3(line)
                                v_max = parse_vec3(lines[i+1])
                                universe_extents1 = [v_max[j] - v_min[j] for j in range(3)]
                            except Exception: pass
                        elif "Mesh2 Universe Min:" in line and (i+1) < len(lines) and "Mesh2 Universe Max:" in lines[i+1]:
                            try:
                                v_min = parse_vec3(line)
                                v_max = parse_vec3(lines[i+1])
                                universe_extents2 = [v_max[j] - v_min[j] for j in range(3)]
                            except Exception: pass

                if not json_output.exists():
                    return {"error": f"Timing JSON not found at {json_output}. Output:\n{stdout_text + stderr_text}"}

                with open(json_output, 'r') as f:
                    data = json.load(f)

                phases = data.get("phases", {})
                phase_values = {}
                for key, phase_data in phases.items():
                    normalized_key = re.sub(r"_\d+$", "", key.lower())
                    phase_values[normalized_key] = phase_values.get(normalized_key, 0.0) + phase_data.get("duration_ms", 0.0)

                has_detailed_raytrace = any(k.startswith("raytrace_") for k in phase_values.keys())

                if self.mode == "estimated":
                    # Sum all relevant phases to get the total query time.
                    # This ensures that query_time and breakdown sum are always identical.
                    components = [
                        "selectivity estimation",
                        "raytrace_hash_mesh1tomesh2",
                        "raytrace_hash_mesh2tomesh1",
                        "raytrace_overlap_hash_mesh1tomesh2",
                        "raytrace_overlap_hash_mesh2tomesh1",
                        "raytrace_containment_hash_mesh1tomesh2",
                        "raytrace_containment_hash_mesh2tomesh1",
                        "compact_hash_table_pairs",
                        "download results",
                    ]
                    query_time = sum(phase_values.get(c, 0.0) for c in components)
                else:
                    query_time = phase_values.get("selectivity estimation", 0.0)

                found = query_time > 0.0

                for normalized_key, duration in phase_values.items():
                    if not any(normalized_key.startswith(prefix) for prefix in expected_prefixes):
                        continue
                    if has_detailed_raytrace and normalized_key in ("query", "execute hash query"):
                        continue
                    if normalized_key not in breakdown_accum:
                        breakdown_accum[normalized_key] = []
                    breakdown_accum[normalized_key].append(duration)

                if not found:
                    return {"error": f"Expected timing phases not found in {json_output}"}

                runtimes.append(query_time)

            except subprocess.TimeoutExpired:
                print(f"[{self.name}] Timeout reached ({timeout}s)")
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"Pierce failed with exit code {e.returncode}: {e.stderr}"}
            except json.JSONDecodeError:
                return {"error": "Failed to parse timing JSON"}
            finally:
                if json_output.exists():
                    json_output.unlink()

        if not runtimes:
            return {"error": "No timing results collected for Pierce"}

        breakdown_stats = {}
        for phase, times in breakdown_accum.items():
            breakdown_stats[phase] = np.mean(times)

        return {
            "mean": np.mean(runtimes),
            "min": np.min(runtimes),
            "max": np.max(runtimes),
            "std": np.std(runtimes),
            "raw_times": runtimes,
            "breakdown": breakdown_stats,
            "num_obj1": num_obj1,
            "num_obj2": num_obj2,
            "num_intersections": num_intersections,
            "universe_extents1": universe_extents1,
            "universe_extents2": universe_extents2
        }
