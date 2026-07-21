#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _load_result(compare_dir: Path, label: str) -> dict:
    path = compare_dir / f"{label}.results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result for {label}: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _direct_rows(payload: dict) -> dict[int, dict]:
    results = payload["results"]
    counts = results["counts"]
    direct = results["direct_estimation"]
    rows = {}
    for idx, nu in enumerate(counts):
        rows[int(nu)] = {
            "query_ms": direct["mean"][idx],
            "std_ms": direct["std"][idx],
            "num_gpus_active": (
                direct.get("num_gpus_active", [None] * len(counts))[idx]
                if isinstance(direct.get("num_gpus_active"), list)
                else None
            ),
            "num_obj1": results.get("num_obj1", [None] * len(counts))[idx],
            "num_obj2": results.get("num_obj2", [None] * len(counts))[idx],
            "num_intersections": results.get("num_intersections", [None] * len(counts))[idx],
        }
    return rows


def _fmt(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare H100 and RTX Pro 6000 Pierce large_nu_v query timings.")
    parser.add_argument("compare_dir", type=Path)
    parser.add_argument("--h100-label", default="h100")
    parser.add_argument("--rtx-label", default="rtx_pro_6000")
    args = parser.parse_args()

    compare_dir = args.compare_dir
    h100_rows = _direct_rows(_load_result(compare_dir, args.h100_label))
    rtx_rows = _direct_rows(_load_result(compare_dir, args.rtx_label))
    common_nu = sorted(set(h100_rows) & set(rtx_rows))
    if not common_nu:
        raise SystemExit("No common nu counts found between the two result files.")

    rows = []
    for nu in common_nu:
        h = h100_rows[nu]
        r = rtx_rows[nu]
        h_ms = h["query_ms"]
        r_ms = r["query_ms"]
        speedup = (r_ms / h_ms) if h_ms and r_ms else None
        rows.append(
            {
                "nu": nu,
                "num_nuclei": h["num_obj2"],
                "num_intersections": h["num_intersections"],
                "h100_query_ms": h_ms,
                "h100_std_ms": h["std_ms"],
                "rtx_pro_6000_query_ms": r_ms,
                "rtx_pro_6000_std_ms": r["std_ms"],
                "rtx_over_h100_speedup": speedup,
                "h100_active_gpus": h["num_gpus_active"],
                "rtx_active_gpus": r["num_gpus_active"],
            }
        )

    csv_path = compare_dir / "h100_vs_rtx_pro_6000_large_nu_v.csv"
    json_path = compare_dir / "h100_vs_rtx_pro_6000_large_nu_v.json"
    md_path = compare_dir / "h100_vs_rtx_pro_6000_large_nu_v.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=2)

    lines = [
        "# H100 vs RTX Pro 6000 Pierce large_nu_v Query Comparison",
        "",
        "| nu | nuclei | intersections | H100 ms | RTX Pro 6000 ms | RTX/H100 |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {nu} | {num_nuclei} | {num_intersections} | {h100} | {rtx} | {speedup} |".format(
                nu=row["nu"],
                num_nuclei=row["num_nuclei"],
                num_intersections=row["num_intersections"],
                h100=_fmt(row["h100_query_ms"]),
                rtx=_fmt(row["rtx_pro_6000_query_ms"]),
                speedup=_fmt(row["rtx_over_h100_speedup"]),
            )
        )
    lines.extend(
        [
            "",
            f"CSV: `{csv_path}`",
            f"JSON: `{json_path}`",
        ]
    )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path)


if __name__ == "__main__":
    main()
