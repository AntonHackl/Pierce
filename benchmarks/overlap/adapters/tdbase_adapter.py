import subprocess
import time
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from .base import OverlapBenchmarkAdapter, run_command_streaming
from benchmarks.common.adapters.tdbase_common import (
    TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    parse_tdbase_run_metrics,
    query_time_for_mode,
    validate_tdbase_timing_mode,
)

class TDBaseAdapter(OverlapBenchmarkAdapter):
    def __init__(
        self,
        tdbase_dir: str,
        preprocessed_dir: Optional[str] = None,
        threads: Optional[int] = None,
        compute_threads: int = 1,
        query_timing_mode: str = TDBASE_TIMING_MODE_INDEX_COMPUTE_EVALUATE,
    ):
        super().__init__("TDBase")
        self.tdbase_dir = Path(tdbase_dir)
        direct_build_dir = self.tdbase_dir / "build"

        tdbase_candidates = [
            direct_build_dir / "tdbase",
        ]
        obj_to_dt_candidates = [
            direct_build_dir / "obj_to_dt",
        ]

        self.executable_candidates = [p for p in tdbase_candidates if p.exists()]
        self.executable = self.executable_candidates[0] if self.executable_candidates else tdbase_candidates[0]
        self.obj_to_dt_exec = next((p for p in obj_to_dt_candidates if p.exists()), obj_to_dt_candidates[0])
        self.preprocessed_dir = Path(preprocessed_dir) if preprocessed_dir else None
        self.threads = threads
        self.compute_threads = compute_threads
        self.query_timing_mode = validate_tdbase_timing_mode(query_timing_mode)
        
        if self.preprocessed_dir:
            self.preprocessed_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _is_buffer_overflow_failure(returncode: int, text: str) -> bool:
        lower = text.lower()
        if "buffer overflow" in lower or "stack smashing" in lower:
            return True
        # SIGABRT often surfaces as -6 when glibc aborts after corruption checks.
        return returncode == -6

    def _build_command(
        self,
        executable: Path,
        file1: str,
        file2: str,
        *,
        threads: Optional[int],
        compute_threads: int,
        lods: List[int],
        use_gpu: bool,
    ) -> List[str]:
        cmd = [
            str(executable),
            "join",
            "-q", "intersect",
            "--tile1", file1,
            "--tile2", file2,
        ]
        if threads:
            cmd.extend(["-t", str(threads)])
        cmd.extend(["--cn", str(compute_threads)])
        for lod in lods:
            cmd.extend(["-l", str(lod)])
        if use_gpu:
            cmd.append("-g")
        return cmd

    def check_preprocessed(self, file_path: str) -> bool:
        """Check if .dt file exists for the given file path in preprocessed dir."""
        input_path = Path(file_path)
        if self.preprocessed_dir:
             dt_file = self.preprocessed_dir / input_path.with_suffix('.dt').name
        else:
             dt_file = input_path.with_suffix('.dt')
        return dt_file.exists()

    def preprocess_from_source(self, source_file: str, dt_file: str, log_dir: Optional[str] = None):
        """Convert .obj to .dt using obj_to_dt tool."""
        source_path = Path(source_file)
        dt_path = Path(dt_file)

        if self.preprocessed_dir:
            output_dt = self.preprocessed_dir / dt_path.with_suffix(".dt").name
        else:
            output_dt = dt_path.with_suffix(".dt")
            
        # Ensure output directory exists (if not using preprocessed_dir, or if it was just created)
        output_dt.parent.mkdir(parents=True, exist_ok=True)
        
        # If it's already a .dt file, just copy it to the preprocessed dir if needed
        if source_path.suffix == '.dt':
            if source_path.resolve() != output_dt.resolve():
                import shutil
                print(f"[{self.name}] Copying {source_path.name} to {output_dt.name}...")
                shutil.copyfile(source_path, output_dt)
            return

        if not self.obj_to_dt_exec.exists():
            print(f"[{self.name}] Error: obj_to_dt tool not found at {self.obj_to_dt_exec}")
            return

        if source_path.suffix == ".obj":
            cmd = [
                str(self.obj_to_dt_exec),
                "--obj",
                str(source_path),
                "--output",
                str(output_dt.parent),
                "--output-name",
                output_dt.stem,
            ]
        elif source_path.suffix == ".off":
            cmd = [
                str(self.obj_to_dt_exec),
                "--off",
                str(source_path),
                "--output",
                str(output_dt.parent),
                "--output-name",
                output_dt.stem,
            ]
        else:
            raise RuntimeError(
                f"[{self.name}] Unsupported source format for obj_to_dt conversion: {source_path.suffix}"
            )

        print(f"[{self.name}] Converting {source_path.name} to {output_dt.name}...")
        
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)
            log_path = adapter_log_dir / f"preprocess_{dt_path.stem}_{int(time.time())}.log"
            run_command_streaming(cmd, timeout=None, log_path=str(log_path), prefix=f"[{self.name}]")
        else:
            run_command_streaming(cmd, timeout=None, log_path=None, prefix=f"[{self.name}]")

    def run_overlap(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run tdbase overlap join."""
        if not self.executable.exists():
            return {"error": f"Executable not found: {self.executable}"}
            
        # Determine actual input files (use preprocessed if available)
        input_path1 = Path(file1)
        input_path2 = Path(file2)
        f1 = file1
        f2 = file2
        
        if self.preprocessed_dir:
            p1 = self.preprocessed_dir / input_path1.with_suffix('.dt').name
            p2 = self.preprocessed_dir / input_path2.with_suffix('.dt').name
            p1_exists = p1.exists()
            p2_exists = p2.exists()
            if p1_exists and p2_exists:
                f1 = str(p1)
                f2 = str(p2)
            elif p1_exists or p2_exists:
                # Avoid mixing preprocessed/raw TDBase inputs; this has proven unstable.
                print(
                    f"[{self.name}] Mixed preprocessed/raw inputs detected "
                    f"(have_pre1={p1_exists}, have_pre2={p2_exists}). "
                    f"Falling back to raw files for both tiles."
                )

        # Run TDBase once per run with multiple -l flags (progressive LODs).
        lods = [20, 40, 60, 80, 100]
        runtimes = []
        preprocessing_times = []
        loading_times = []
        run_metrics = []
        print(f"[{self.name}] Running TDBase with LODs {lods} (GPU), timing mode={self.query_timing_mode}")

        adapter_log_dir = None
        if log_dir:
            adapter_log_dir = Path(log_dir) / self.name
            adapter_log_dir.mkdir(parents=True, exist_ok=True)

        fallback_threads = 1 if (self.threads is None or self.threads > 1) else self.threads
        fallback_compute_threads = 1

        for run_idx in range(num_runs):
            attempts: List[Tuple[str, Path, Optional[int], int, bool]] = [
                ("primary", self.executable, self.threads, self.compute_threads, True),
            ]
            for alt_exec in self.executable_candidates:
                if alt_exec != self.executable:
                    attempts.append(("alt_binary", alt_exec, self.threads, self.compute_threads, True))
                    break
            attempts.append(("conservative_gpu", self.executable, fallback_threads, fallback_compute_threads, True))
            attempts.append(("conservative_cpu", self.executable, fallback_threads, fallback_compute_threads, False))

            # Deduplicate while preserving order.
            dedup: Dict[Tuple[str, int, int, bool], Tuple[str, Path, Optional[int], int, bool]] = {}
            for label, exe, th, cn, gpu in attempts:
                key = (str(exe), th or 0, cn, gpu)
                if key not in dedup:
                    dedup[key] = (label, exe, th, cn, gpu)
            attempts = list(dedup.values())

            last_error = None
            output = ""
            for attempt_idx, (attempt_label, exe, th, cn, use_gpu) in enumerate(attempts):
                cmd = self._build_command(exe, f1, f2, threads=th, compute_threads=cn, lods=lods, use_gpu=use_gpu)
                try:
                    log_path = None
                    if adapter_log_dir is not None:
                        suffix = "" if attempt_idx == 0 else f"_{attempt_label}"
                        log_path = str(adapter_log_dir / f"run_{run_idx:03d}{suffix}.log")
                    if attempt_idx > 0:
                        print(
                            f"[{self.name}] Retrying run {run_idx} via {attempt_label}: "
                            f"exe={exe} threads={th or 'default'} cn={cn} gpu={use_gpu}"
                        )
                    stdout_text, stderr_text = run_command_streaming(
                        cmd,
                        timeout=timeout,
                        log_path=log_path,
                        prefix=f"[{self.name}]",
                    )
                    output = stdout_text + stderr_text
                    metrics = parse_tdbase_run_metrics(output)
                    query_time_ms = query_time_for_mode(metrics, self.query_timing_mode)
                    runtimes.append(query_time_ms)
                    preprocessing_times.append(metrics["preprocessing_ms"])
                    loading_times.append(metrics["loading_ms"])
                    run_metrics.append(
                        {
                            **metrics,
                            "query_time_ms": query_time_ms,
                            "query_timing_mode": self.query_timing_mode,
                            "attempt": attempt_label,
                            "executable": str(exe),
                            "threads": th,
                            "compute_threads": cn,
                            "gpu": use_gpu,
                        }
                    )
                    last_error = None
                    break
                except subprocess.TimeoutExpired:
                    print(f"[{self.name}] Timeout reached ({timeout}s)")
                    return {"error": f"Timeout reached ({timeout}s)"}
                except subprocess.CalledProcessError as e:
                    err_text = (e.stderr or "") + "\n" + (e.output or "")
                    last_error = f"TDBase failed with exit code {e.returncode}: {e.stderr}"
                    if attempt_idx + 1 < len(attempts) and self._is_buffer_overflow_failure(e.returncode, err_text):
                        print(f"[{self.name}] Detected buffer-overflow-style failure; trying safer fallback...")
                        continue
                    return {"error": last_error}
                except RuntimeError as e:
                    last_error = str(e)
                    if attempt_idx + 1 < len(attempts):
                        print(f"[{self.name}] Parse/runtime failure; trying safer fallback...")
                        continue
                    print(f"[{self.name}] Error: {e}. Result:\n{output}")
                    return {"error": last_error}

            if last_error is not None:
                return {"error": last_error}

        if not runtimes:
            return {"error": "No timing results collected for TDBase"}

        # Return aggregate stats over the runs (each run processed all LODs)
        mean_prep = float(np.mean(preprocessing_times)) if preprocessing_times else 0.0
        mean_loading = float(np.mean(loading_times)) if loading_times else 0.0
        return {
            "mean": float(np.mean(runtimes)),
            "min": float(np.min(runtimes)),
            "max": float(np.max(runtimes)),
            "std": float(np.std(runtimes)),
            "raw_times": [float(x) for x in runtimes],
            "mean_preprocessing": mean_prep,
            "mean_loading": mean_loading,
            "run_metrics": run_metrics,
            "query_timing_mode": self.query_timing_mode,
            "lods": lods,
            "gpu": True
        }
