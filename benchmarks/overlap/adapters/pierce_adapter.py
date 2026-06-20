import subprocess
import time
import json
import re
import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path
from .base import OverlapBenchmarkAdapter, run_command_streaming

class PierceAdapter(OverlapBenchmarkAdapter):
    def __init__(
        self,
        pierce_dir: str,
        mode: str = "exact",
        preprocessed_dir: str = "preprocessed",
        timings_dir: str = "timings",
        grid_cell_size: float = 5.0,
        warmup_runs: int = 10,
        track_hash_contention: bool = False,
        use_alpha_correction: bool = True,
        hash_table_size: Optional[int] = None,
        hash_table_free_mem_fraction: Optional[float] = None,
        overlap_max_iterations: int = 100,
    ):
        """
        mode: 'exact' or 'direct_estimation'
        grid_cell_size: resolution for grid generation (default: 10)
        hash_table_size: manually override hash table slot count for direct_estimation mode
        hash_table_free_mem_fraction: fraction of currently free GPU memory to use for hash table size in direct_estimation mode
        """
        super().__init__(f"Pierce_{mode}")
        self.pierce_dir = Path(pierce_dir)
        self.mode = mode
        self.preprocessed_dir = Path(preprocessed_dir)
        self.timings_dir = Path(timings_dir)
        self.grid_cell_size = grid_cell_size
        self.warmup_runs = warmup_runs
        self.track_hash_contention = track_hash_contention
        self.use_alpha_correction = use_alpha_correction
        self.hash_table_size = hash_table_size
        self.hash_table_free_mem_fraction = hash_table_free_mem_fraction
        self.overlap_max_iterations = overlap_max_iterations
        # Ensure directories exist
        self.timings_dir.mkdir(parents=True, exist_ok=True)
        self.preprocessed_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine executable based on mode
        # Binaries are in query/build/bin
        query_bin_dir = self.pierce_dir / "query" / "build" / "bin"
        if self.mode == "exact":
            self.executable = query_bin_dir / "pierce_overlap_two_pass"
        elif self.mode == "direct_estimation":
            self.executable = query_bin_dir / "pierce_overlap"
        elif self.mode == "estimated":
            self.executable = query_bin_dir / "pierce_intersection"
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

        # Preprocess binary is in preprocess/build/bin
        self.preprocess_exec = self.pierce_dir / "preprocess" / "build" / "bin" / "pierce_preprocess"
        
    def _get_preprocessed_path(self, file_path: str) -> Path:
        input_path = Path(file_path)
        grid_token = str(self.grid_cell_size).replace(".", "_")
        return self.preprocessed_dir / f"{input_path.stem}_g{grid_token}.pre"

    def check_preprocessed(self, file_path: str) -> bool:
        """Check if .pre file exists for the given .dt or .obj file in preprocessed dir."""
        return self._get_preprocessed_path(file_path).exists()

    def preprocess(self, file_path: str):
        """Run the Pierce preprocessing tool including grid generation."""
        self.preprocess_from_source(file_path, file_path)
    
    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        """Run preprocessing using a source file (.obj) but naming outputs based on dt_file."""
        source_path = Path(source_file)
        dt_path = Path(dt_file)
        
        # Output files are named based on dt_file for consistency, stored in PREPROCESSED_DIR
        output_geometry = self._get_preprocessed_path(dt_file)
        output_timing = self.timings_dir / (dt_path.stem + f'_g{str(self.grid_cell_size).replace(".", "_")}_timing.json')
        
        # Determine mode based on source file extension
        mode = "dt" if source_path.suffix == ".dt" else "mesh"
        
        # Preprocess dataset with grid generation
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
            # Stream to terminal without logging
            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")

    def run_overlap(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
        query_direction: str = "both",
        pairs_output: Optional[str] = None,
        estimate_only: bool = False,
        overlap_max_iterations: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Execute the overlap join query."""
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}

        # Use preprocessed files if they exist in the preprocessed directory
        p1 = self._get_preprocessed_path(file1)
        p2 = self._get_preprocessed_path(file2)
        
        f1 = str(p1) if p1.exists() else file1
        f2 = str(p2) if p2.exists() else file2

        runtimes = []
        breakdown_accum = {} # key: phase name, value: list of durations
        num_obj1 = 0
        num_obj2 = 0
        num_intersections = 0
        raw_estimated_pairs = 0
        final_estimated_pairs = 0
        universe_extents1 = [0.0, 0.0, 0.0]
        universe_extents2 = [0.0, 0.0, 0.0]
        hash_accesses = 0
        hash_contentions = 0
        contention_pct = 0.0
        actual_hash_table_size = 0
        hash_table_allocated_bytes = 0
        result_buffer_capacity = 0
        result_buffer_allocated_bytes = 0
        result_buffer_used_bytes = 0
        
        print(f"[{self.name}] Running benchmark...")

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        if self.mode == "exact":
            expected_prefixes = [
                "raytrace_",
                "gpu deduplication",
                "download results",
                "query",
            ]
        elif self.mode in ("direct_estimation", "estimated"):
            # For direct_estimation/estimated mode, include selectivity estimation in query time
            expected_prefixes = [
                "selectivity estimation",
                "raytrace_hash_",
                "download results",
                "query",
                "compact_hash_table_pairs",
            ]

        # Execute num_runs times, each with warmup
        for run_idx in range(num_runs):
            json_output = self.timings_dir / f"timing_{self.mode}_{int(time.time())}_{run_idx}.json"
            
            cmd = [
                str(self.executable),
                "--mesh1", f1,
                "--mesh2", f2,
                "--runs", "1",
                "--warmup-runs", str(self.warmup_runs),
                "--no-export",
                "--output", str(json_output)
            ]

            if self.mode == "direct_estimation":
                cmd.extend(["--query-direction", query_direction])
                if not self.use_alpha_correction:
                    cmd.append("--no-alpha-correction")
                if self.track_hash_contention:
                    cmd.append("--track-hash-contention")
                if self.hash_table_size is not None:
                    cmd.extend(["--hash-table-size", str(self.hash_table_size)])
                elif self.hash_table_free_mem_fraction is not None:
                    cmd.extend(["--hash-table-free-mem-fraction", str(self.hash_table_free_mem_fraction)])
            
            if self.mode in ("direct_estimation", "estimated"):
                if pairs_output and run_idx == (num_runs - 1):
                    cmd.extend(["--pairs-output", str(pairs_output)])
                if estimate_only:
                    cmd.append("--estimate-only")
                
                # Pass max iterations
                max_iter = overlap_max_iterations if overlap_max_iterations is not None else self.overlap_max_iterations
                cmd.extend(["--overlap-max-iterations", str(max_iter)])
                
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
                
                # Parse summary from stdout.
                # Object counts/universe extents are taken from run 0, while pair/hash counters are
                # updated from every run (last value wins) to reflect the latest measured execution.
                lines = stdout_text.splitlines()

                def parse_vec3(l):
                    return [float(p.strip()) for p in l.split("[")[1].split("]")[0].split(",")]

                for i, line in enumerate(lines):
                    if run_idx == 0:
                        if "Mesh1 objects:" in line:
                            num_obj1 = int(line.split(":")[1].strip())
                        elif "Mesh2 objects:" in line:
                            num_obj2 = int(line.split(":")[1].strip())
                        elif "Unique object pairs:" in line:
                            num_intersections = int(line.split(":")[1].strip())
                        elif "Universe Extents:" in line:
                            try:
                                ext = parse_vec3(line)
                                universe_extents1 = ext
                                universe_extents2 = ext
                            except Exception:
                                pass
                        elif "Mesh1 Universe Min:" in line and (i + 1) < len(lines) and "Mesh1 Universe Max:" in lines[i + 1]:
                            try:
                                v_min = parse_vec3(line)
                                v_max = parse_vec3(lines[i + 1])
                                universe_extents1 = [v_max[j] - v_min[j] for j in range(3)]
                            except Exception:
                                pass
                        elif "Mesh2 Universe Min:" in line and (i + 1) < len(lines) and "Mesh2 Universe Max:" in lines[i + 1]:
                            try:
                                v_min = parse_vec3(line)
                                v_max = parse_vec3(lines[i + 1])
                                universe_extents2 = [v_max[j] - v_min[j] for j in range(3)]
                            except Exception:
                                pass

                    if "Raw Potential Pairs:" in line:
                        try:
                            raw_estimated_pairs = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Final Estimated Pairs:" in line:
                        try:
                            final_estimated_pairs = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Hash Table Query found" in line:
                        try:
                            num_intersections = int(line.split("found")[1].split("unique")[0].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Using Direct Estimated Hash Table Size:" in line:
                        try:
                            actual_hash_table_size = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Using Free GPU Memory Hash Table Size:" in line:
                        try:
                            actual_hash_table_size = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Using Manual Hash Table Size:" in line:
                        try:
                            actual_hash_table_size = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Hash Table Allocated Bytes:" in line:
                        try:
                            hash_table_allocated_bytes = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Result Buffer Capacity:" in line:
                        try:
                            result_buffer_capacity = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Result Buffer Allocated Bytes:" in line:
                        try:
                            result_buffer_allocated_bytes = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass
                    elif "Result Buffer Used Bytes:" in line:
                        try:
                            result_buffer_used_bytes = int(line.split(":", 1)[1].strip())
                        except (ValueError, IndexError):
                            pass

                # Parse contention from any run (last value wins)
                for line in lines:
                    if "Hash contention (run" in line and "):" in line:
                        try:
                            # Format: "Hash contention (run N): X/Y accesses (Z%)"
                            counts_part = line.split("):", 1)[1].strip()
                            # counts_part: "X/Y accesses (Z%)"
                            slash_part = counts_part.split(" accesses")[0].strip()
                            hash_contentions = int(slash_part.split("/")[0])
                            hash_accesses = int(slash_part.split("/")[1])
                            pct_str = counts_part.split("(")[1].split("%")[0].strip()
                            contention_pct = float(pct_str)
                        except (ValueError, IndexError):
                            pass

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

                if self.mode == "exact":
                    # Sum all relevant phases to get the total query time.
                    query_time = sum(v for k, v in phase_values.items() if k.startswith("raytrace_"))
                    query_time += phase_values.get("gpu deduplication", 0.0)
                    query_time += phase_values.get("download results", 0.0)
                elif self.mode in ("estimated", "direct_estimation"):
                    # Sum all relevant phases to get the total query time.
                    components = [
                        "selectivity estimation",
                        "raytrace_hash_mesh1tomesh2",
                        "raytrace_hash_mesh2tomesh1",
                        "raytrace_overlap_hash_mesh1tomesh2",
                        "raytrace_overlap_hash_mesh2tomesh1",
                        "compact_hash_table_pairs",
                        "download results",
                    ]
                    query_time = sum(phase_values.get(c, 0.0) for c in components)
                    if query_time <= 0.0:
                         # Fallback to 'query' or 'execute hash query' if components not found
                         query_time = phase_values.get("execute hash query", 0.0) or phase_values.get("query", 0.0)
                elif self.mode == "estimated":
                    # Sum all relevant phases for intersection_estimated
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
                    if query_time <= 0.0:
                         query_time = phase_values.get("execute hash query", 0.0) or phase_values.get("query", 0.0)
                else:
                    # In direct_estimation mode, if no components found, fallback
                    query_time = phase_values.get("execute hash query", 0.0) or phase_values.get("query", 0.0)

                found = query_time > 0.0

                for normalized_key, duration in phase_values.items():
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

        # Calculate mean breakdown
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
            "raw_estimated_pairs": raw_estimated_pairs,
            "final_estimated_pairs": final_estimated_pairs,
            "universe_extents1": universe_extents1,
            "universe_extents2": universe_extents2,
            "hash_accesses": hash_accesses,
            "hash_contentions": hash_contentions,
            "contention_pct": contention_pct,
            "actual_hash_table_size": actual_hash_table_size,
            "hash_table_allocated_bytes": hash_table_allocated_bytes,
            "result_buffer_capacity": result_buffer_capacity,
            "result_buffer_allocated_bytes": result_buffer_allocated_bytes,
            "result_buffer_used_bytes": result_buffer_used_bytes,
        }
