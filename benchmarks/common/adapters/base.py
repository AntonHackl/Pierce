from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

import subprocess
import sys
import time
import queue
import threading
from pathlib import Path
from typing import List, Tuple


def run_command_streaming(
    cmd: List[str],
    *,
    timeout: Optional[float] = None,
    log_path: Optional[str] = None,
    prefix: Optional[str] = None,
    cwd: Optional[str] = None,
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

    start = time.monotonic()
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            cwd=cwd,
        )
    except Exception:
        if log_fh is not None:
            log_fh.close()
        raise

    stdout_parts: List[str] = []
    stderr_parts: List[str] = []

    assert proc.stdout is not None
    assert proc.stderr is not None
    # Windows selectors cannot monitor subprocess pipes. Separate readers also
    # ensure that a full stderr pipe cannot block a child writing to stdout.
    events = queue.Queue()

    def read_stream(stream, name):
        try:
            for line in stream:
                events.put((name, line))
        except Exception as error:
            events.put((name, error))
        finally:
            events.put((name, None))

    readers = [
        threading.Thread(target=read_stream, args=(proc.stdout, "stdout"), daemon=True),
        threading.Thread(target=read_stream, args=(proc.stderr, "stderr"), daemon=True),
    ]
    for reader in readers:
        reader.start()

    try:
        open_streams = 2
        while open_streams:
            if timeout is not None and (time.monotonic() - start) > timeout:
                raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout)
            try:
                stream_name, line = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if line is None:
                open_streams -= 1
                continue
            if isinstance(line, Exception):
                raise line
            out_line = f"{prefix} {line}" if prefix else line
            if stream_name == "stdout":
                stdout_parts.append(line)
                sys.stdout.write(out_line)
                sys.stdout.flush()
            else:
                stderr_parts.append(line)
                sys.stderr.write(out_line)
                sys.stderr.flush()
            if log_fh is not None:
                log_fh.write(out_line)
                log_fh.flush()

        remaining = None if timeout is None else max(0.0, timeout - (time.monotonic() - start))
        return_code = proc.wait(timeout=remaining)
        stdout_text = "".join(stdout_parts)
        stderr_text = "".join(stderr_parts)

        if return_code != 0:
            raise subprocess.CalledProcessError(return_code, cmd, output=stdout_text, stderr=stderr_text)
        return stdout_text, stderr_text
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        for reader in readers:
            reader.join(timeout=1.0)
        proc.stdout.close()
        proc.stderr.close()
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
