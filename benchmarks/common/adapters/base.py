from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import subprocess
import sys
import time
import selectors
from pathlib import Path
from typing import List, Tuple


def run_command_streaming(
    cmd: List[str],
    *,
    timeout: Optional[float] = None,
    log_path: Optional[str] = None,
    prefix: Optional[str] = None,
) -> Tuple[str, str]:
    """Run a command while streaming output to terminal and optionally logging to file.

    Returns (stdout_text, stderr_text). Raises CalledProcessError/TimeoutExpired on failure.
    """
    log_fh = None
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        log_fh = open(log_path, "w", encoding="utf-8")
        log_fh.write("COMMAND:\n")
        log_fh.write(" ".join(cmd) + "\n\n")

    start = time.time()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []

    sel = selectors.DefaultSelector()
    assert proc.stdout is not None
    assert proc.stderr is not None
    sel.register(proc.stdout, selectors.EVENT_READ, data="stdout")
    sel.register(proc.stderr, selectors.EVENT_READ, data="stderr")

    try:
        while sel.get_map():
            if timeout is not None and (time.time() - start) > timeout:
                proc.kill()
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)

            for key, _ in sel.select(timeout=0.1):
                stream_name = key.data
                line = key.fileobj.readline()
                if line == "":
                    try:
                        sel.unregister(key.fileobj)
                    except Exception:
                        pass
                    continue

                if stream_name == "stdout":
                    out_line = f"{prefix} {line}" if prefix else line
                    stdout_parts.append(line)
                    sys.stdout.write(out_line)
                    sys.stdout.flush()
                else:
                    out_line = f"{prefix} {line}" if prefix else line
                    stderr_parts.append(line)
                    sys.stderr.write(out_line)
                    sys.stderr.flush()

                if log_fh is not None:
                    log_fh.write(out_line)
                    log_fh.flush()

        return_code = proc.wait()
        stdout_text = "".join(stdout_parts)
        stderr_text = "".join(stderr_parts)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=stdout_text, stderr=stderr_text)
        return stdout_text, stderr_text
    finally:
        try:
            sel.close()
        except Exception:
            pass
        if log_fh is not None:
            log_fh.flush()
            log_fh.close()


class OverlapBenchmarkAdapter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_overlap(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the overlap join between two files.
        Returns a dictionary containing timing results and statistics.
        """
        pass


class IntersectionBenchmarkAdapter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_intersection(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the intersection join between two files.
        Returns a dictionary containing timing results and statistics.
        """
        pass


class ContainmentBenchmarkAdapter(ABC):
    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def run_containment(
        self,
        file1: str,
        file2: str,
        num_runs: int,
        timeout: Optional[float] = None,
        log_dir: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run strict containment join (B in A) between two files.
        Returns a dictionary containing timing results and statistics.
        """
        pass
