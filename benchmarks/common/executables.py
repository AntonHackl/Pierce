"""Locate Pierce executables from single- and multi-configuration builds."""

import os
from pathlib import Path


def pierce_executable(pierce_dir: Path, component: str, name: str) -> Path:
    filename = name + (".exe" if os.name == "nt" else "")
    bin_dir = Path(pierce_dir) / component / "build" / "bin"
    candidates = [bin_dir / filename]
    candidates.extend(bin_dir / config / filename for config in ("Release", "RelWithDebInfo", "Debug"))
    return next((path for path in candidates if path.is_file()), candidates[0])
