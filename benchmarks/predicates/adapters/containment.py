import json
import re
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from benchmarks.common.adapters.base import ContainmentBenchmarkAdapter, run_command_streaming


class PierceContainmentAdapter(ContainmentBenchmarkAdapter):
    def __init__(
        self,
        pierce_dir: str,
        preprocessed_dir: str = "preprocessed",
        timings_dir: str = "timings",
        grid_cell_size: int = 10,
        warmup_runs: int = 1,
        include_overlap_pairs: bool = False,
    ):
        super().__init__("Pierce")
        self.pierce_dir = Path(pierce_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.timings_dir = Path(timings_dir)
        self.grid_cell_size = grid_cell_size
        self.warmup_runs = warmup_runs
        self.include_overlap_pairs = include_overlap_pairs

        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        self.timings_dir.mkdir(parents=True, exist_ok=True)

        query_bin_dir = self.pierce_dir / "query" / "build" / "bin"
        self.executable = query_bin_dir / "pierce_containment"
        self.preprocess_exec = self.pierce_dir / "preprocess" / "build" / "bin" / "pierce_preprocess"

    def _get_preprocessed_path(self, file_path: str) -> Path:
        input_path = Path(file_path)
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
            "--grid-cell-size", str(self.grid_cell_size),
        ]

        print(f"[{self.name}] Preprocessing {source_path.name} (output: {dt_path.name})...")
        log_path = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = str(adapter_log_dir / f"preprocess_{dt_path.stem}_{int(time.time())}.log")

        run_command_streaming(cmd, timeout=None, log_path=log_path, prefix=f"[{self.name}]")

    def run_containment(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
        extra_args: Optional[list] = None,
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
        num_containments = 0
        num_overlaps = 0
        num_reported_pairs = 0

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        for run_idx in range(num_runs):
            json_output = self.timings_dir / f"timing_containment_{int(time.time())}_{run_idx}.json"
            cmd = [
                str(self.executable),
                "--mesh1", f1,
                "--mesh2", f2,
                "--output", str(json_output),
                "--runs", "1",
                "--warmup-runs", str(self.warmup_runs),
                "--no-export",
            ]
            if extra_args:
                cmd.extend(extra_args)
            if self.include_overlap_pairs:
                cmd.append("--include-overlap-pairs")

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
                output = stdout_text + stderr_text

                if run_idx == 0:
                    match = re.search(r"A objects:\s*(\d+)", output)
                    if match:
                        num_obj1 = int(match.group(1))
                    match = re.search(r"B objects:\s*(\d+)", output)
                    if match:
                        num_obj2 = int(match.group(1))
                    match = re.search(r"Containment pairs \(B in A\):\s*(\d+)", output)
                    if match:
                        num_containments = int(match.group(1))
                    match = re.search(r"Overlap/Touch pairs \(A vs B\):\s*(\d+)", output)
                    if match:
                        num_overlaps = int(match.group(1))
                    match = re.search(r"Reported pairs:\s*(\d+)", output)
                    if match:
                        num_reported_pairs = int(match.group(1))

                if not json_output.exists():
                    return {"error": f"Timing JSON not found at {json_output}"}

                with open(json_output, "r", encoding="utf-8") as f:
                    data = json.load(f)

                phases = data.get("phases", {})
                phase_values = {}
                for key, phase_data in phases.items():
                    normalized_key = re.sub(r"_\d+$", "", key.lower())
                    phase_values[normalized_key] = phase_values.get(normalized_key, 0.0) + phase_data.get("duration_ms", 0.0)

                # Explicitly sum components to ensure consistency with breakdown
                components = [
                    "selectivity estimation",
                    "raytrace_overlap_hash_mesh1tomesh2",
                    "raytrace_overlap_hash_mesh2tomesh1",
                    "raytrace_containment_hash_mesh1tomesh2",
                    "raytrace_containment_hash_mesh2tomesh1",
                    "compact_hash_table_pairs (containment)",
                    "compact_hash_table_pairs (overlap)",
                    "download results",
                ]
                query_time = sum(phase_values.get(c, 0.0) for c in components)
                if query_time <= 0.0:
                    # Fallback
                    query_time = phase_values.get("query", 0.0) + phase_values.get("download results", 0.0)

                if query_time <= 0.0:
                    return {"error": f"Expected query timing not found in {json_output}"}

                runtimes.append(query_time)

                for phase, duration in phase_values.items():
                    breakdown_accum.setdefault(phase, []).append(duration)

            except subprocess.TimeoutExpired:
                return {"error": f"Timeout reached ({timeout}s)"}
            except subprocess.CalledProcessError as e:
                return {"error": f"Pierce containment failed with exit code {e.returncode}: {e.stderr}"}
            except json.JSONDecodeError:
                return {"error": "Failed to parse timing JSON"}
            finally:
                if json_output.exists():
                    json_output.unlink()

        if not runtimes:
            return {"error": "No timing results collected"}

        breakdown_stats = {
            phase: {
                "mean": float(statistics.mean(times)),
                "min": float(min(times)),
                "max": float(max(times)),
                "std": float(statistics.pstdev(times) if len(times) > 1 else 0.0),
            }
            for phase, times in breakdown_accum.items()
            if times
        }

        std = statistics.pstdev(runtimes) if len(runtimes) > 1 else 0.0
        return {
            "mean": float(statistics.mean(runtimes)),
            "min": float(min(runtimes)),
            "max": float(max(runtimes)),
            "std": float(std),
            "raw_times": runtimes,
            "num_obj1": int(num_obj1),
            "num_obj2": int(num_obj2),
            "num_containments": int(num_containments),
            "num_overlaps": int(num_overlaps),
            "num_reported_pairs": int(num_reported_pairs or num_containments),
            "breakdown": breakdown_stats,
        }
