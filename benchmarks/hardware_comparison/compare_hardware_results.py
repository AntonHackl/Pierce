#!/usr/bin/env python3
"""Merge H100 and RTX Pro 6000 single-/multi-GPU benchmark results."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


GPU_COUNTS = (1, 2)


def _load_result(compare_dir: Path, label: str) -> dict[str, Any]:
    path = compare_dir / f"{label}.results.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing result for {label}: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _number(value: Any) -> float | None:
    return float(value) if isinstance(value, int | float) else None


def _index_results(payload: dict[str, Any]) -> tuple[dict[tuple[str, str, int], dict[str, Any]], list[str], list[str]]:
    indexed: dict[tuple[str, str, int], dict[str, Any]] = {}
    datasets = list(payload.get("metadata", {}).get("datasets", []))
    queries = list(payload.get("metadata", {}).get("queries", []))
    for result in payload.get("results", []):
        dataset = result.get("dataset")
        gpu_count = result.get("gpu_count")
        if not isinstance(dataset, str) or gpu_count not in GPU_COUNTS:
            continue
        if dataset not in datasets:
            datasets.append(dataset)
        result_queries = result.get("queries", {})
        query_names = list(result_queries) if result_queries else queries
        for query in query_names:
            if query not in queries:
                queries.append(query)
            query_result = result_queries.get(query, {})
            error = query_result.get("error") or result.get("error")
            indexed[(dataset, query, gpu_count)] = {
                "status": "error" if error else result.get("status", "unknown"),
                "mean_ms": _number(query_result.get("mean")),
                "std_ms": _number(query_result.get("std")),
                "num_gpus_active": query_result.get("num_gpus_active"),
                "error": error,
            }
    return indexed, datasets, queries


def _value(record: dict[str, Any] | None, key: str) -> Any:
    return record.get(key) if record else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def _field_names(h100_label: str, rtx_label: str) -> list[str]:
    fields = ["dataset", "query"]
    for label in (h100_label, rtx_label):
        for gpu_count in GPU_COUNTS:
            prefix = f"{label}_{gpu_count}gpu"
            fields.extend([
                f"{prefix}_status",
                f"{prefix}_mean_ms",
                f"{prefix}_std_ms",
                f"{prefix}_active_gpus",
                f"{prefix}_error",
            ])
        fields.append(f"{label}_multi_gpu_speedup")
    fields.extend([
        "rtx_over_h100_1gpu_speedup",
        "rtx_over_h100_2gpu_speedup",
    ])
    return fields


def _build_rows(
    h100_index: dict[tuple[str, str, int], dict[str, Any]],
    rtx_index: dict[tuple[str, str, int], dict[str, Any]],
    datasets: list[str],
    queries: list[str],
    h100_label: str,
    rtx_label: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset in datasets:
        for query in queries:
            h100_records = {count: h100_index.get((dataset, query, count)) for count in GPU_COUNTS}
            rtx_records = {count: rtx_index.get((dataset, query, count)) for count in GPU_COUNTS}
            if not any((*h100_records.values(), *rtx_records.values())):
                continue
            row: dict[str, Any] = {"dataset": dataset, "query": query}
            for label, records in ((h100_label, h100_records), (rtx_label, rtx_records)):
                for gpu_count, record in records.items():
                    prefix = f"{label}_{gpu_count}gpu"
                    row[f"{prefix}_status"] = _value(record, "status") or "unavailable"
                    row[f"{prefix}_mean_ms"] = _value(record, "mean_ms")
                    row[f"{prefix}_std_ms"] = _value(record, "std_ms")
                    row[f"{prefix}_active_gpus"] = _value(record, "num_gpus_active")
                    row[f"{prefix}_error"] = _value(record, "error")
                row[f"{label}_multi_gpu_speedup"] = _ratio(
                    _value(records[1], "mean_ms"), _value(records[2], "mean_ms")
                )
            row["rtx_over_h100_1gpu_speedup"] = _ratio(
                _value(rtx_records[1], "mean_ms"), _value(h100_records[1], "mean_ms")
            )
            row["rtx_over_h100_2gpu_speedup"] = _ratio(
                _value(rtx_records[2], "mean_ms"), _value(h100_records[2], "mean_ms")
            )
            rows.append(row)
    return rows


def _format(value: Any, *, speedup: bool = False) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        rendered = f"{value:.3f}"
        return f"{rendered}x" if speedup else rendered
    return str(value)


def _write_markdown(path: Path, rows: list[dict[str, Any]], h100_label: str, rtx_label: str) -> None:
    lines = [
        "# H100 vs RTX Pro 6000: Single- vs. Multi-GPU Query Comparison",
        "",
        "Runtime values are steady-query mean milliseconds. Speedups are `1 GPU / 2 GPU` "
        "within a hardware type, and `RTX / H100` across hardware types.",
        "",
        "| Dataset | Query | H100 1 GPU ms | H100 2 GPU ms | H100 speedup | RTX 1 GPU ms | RTX 2 GPU ms | RTX speedup | RTX/H100 1 GPU | RTX/H100 2 GPU |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {dataset} | {query} | {h1} | {h2} | {hs} | {r1} | {r2} | {rs} | {x1} | {x2} |".format(
                dataset=row["dataset"],
                query=row["query"],
                h1=_format(row[f"{h100_label}_1gpu_mean_ms"]),
                h2=_format(row[f"{h100_label}_2gpu_mean_ms"]),
                hs=_format(row[f"{h100_label}_multi_gpu_speedup"], speedup=True),
                r1=_format(row[f"{rtx_label}_1gpu_mean_ms"]),
                r2=_format(row[f"{rtx_label}_2gpu_mean_ms"]),
                rs=_format(row[f"{rtx_label}_multi_gpu_speedup"], speedup=True),
                x1=_format(row["rtx_over_h100_1gpu_speedup"], speedup=True),
                x2=_format(row["rtx_over_h100_2gpu_speedup"], speedup=True),
            )
        )
    lines.append("")
    lines.append("`N/A` denotes an unavailable variant, including the intentionally multi-GPU-only MICRONS 16 GB dataset.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare H100 and RTX Pro 6000 single-/multi-GPU query results.")
    parser.add_argument("compare_dir", type=Path)
    parser.add_argument("--h100-label", default="h100")
    parser.add_argument("--rtx-label", default="rtx_pro_6000")
    parser.add_argument("--output-prefix", default="h100_vs_rtx_pro_6000_single_vs_multi_gpu")
    args = parser.parse_args()

    compare_dir = args.compare_dir
    h100_index, h100_datasets, h100_queries = _index_results(_load_result(compare_dir, args.h100_label))
    rtx_index, rtx_datasets, rtx_queries = _index_results(_load_result(compare_dir, args.rtx_label))
    datasets = list(dict.fromkeys([*h100_datasets, *rtx_datasets]))
    queries = list(dict.fromkeys([*h100_queries, *rtx_queries]))
    rows = _build_rows(h100_index, rtx_index, datasets, queries, args.h100_label, args.rtx_label)
    if not rows:
        raise SystemExit("No comparable dataset/query results found in the supplied result files.")

    fields = _field_names(args.h100_label, args.rtx_label)
    csv_path = compare_dir / f"{args.output_prefix}.csv"
    json_path = compare_dir / f"{args.output_prefix}.json"
    markdown_path = compare_dir / f"{args.output_prefix}.md"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "h100_label": args.h100_label,
                "rtx_label": args.rtx_label,
                "speedup_definitions": {
                    "multi_gpu": "one_gpu_mean_ms / two_gpu_mean_ms",
                    "rtx_over_h100": "rtx_mean_ms / h100_mean_ms",
                },
                "rows": rows,
            },
            handle,
            indent=2,
        )
    _write_markdown(markdown_path, rows, args.h100_label, args.rtx_label)
    print(markdown_path)


if __name__ == "__main__":
    main()
