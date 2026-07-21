from __future__ import annotations

import re
import struct
from pathlib import Path


BINARY_FILE_MAGIC = 0x52334442


def expected_binary_geometry_version(repo_root: Path) -> int | None:
    header_path = repo_root / "pierce" / "common" / "include" / "BinaryIO.h"
    try:
        text = header_path.read_text(encoding="utf-8")
    except OSError:
        return None

    match = re.search(r"BINARY_FILE_VERSION\s*=\s*(\d+)", text)
    return int(match.group(1)) if match else None


def binary_geometry_version(path: Path) -> int | None:
    try:
        with path.open("rb") as f:
            data = f.read(8)
    except OSError:
        return None
    if len(data) < 8:
        return None

    magic, version = struct.unpack("<II", data)
    if magic != BINARY_FILE_MAGIC:
        return None
    return version


def is_current_binary_geometry(path: Path, repo_root: Path) -> bool:
    expected_version = expected_binary_geometry_version(repo_root)
    if expected_version is None:
        return path.exists()
    return binary_geometry_version(path) == expected_version
