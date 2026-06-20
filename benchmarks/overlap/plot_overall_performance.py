#!/usr/bin/env python3
import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.common.scenario_utils import create_benchmark_run_layout, write_json
from benchmarks.common.viz_utils import apply_side_by_side_style, set_log_timing_axis_limits, style_for, PAPER_SIDE_BY_SIDE_FIGSIZE, make_legend_bold


@dataclass
class GroupResult:
    group_name: str
    run_dir: Path
    run_timestamp: str
    selector_value: str
    approach_to_mean: Dict[str, float]
    result_size: Optional[int] = None


def _large_nu_v_group_name(nu: int) -> str:
    return r"Vessel $\bowtie$ Nuclei$_1$"


def _large_nu_nn_group_name(nu: int) -> str:
    return r"Nuclei$_2$ $\bowtie$ Nuclei$_3$"


def _parse_ts_from_run_dir(run_dir: Path) -> str:
    parts = run_dir.name.split("_")
    if len(parts) >= 2:
        return "_".join(parts[-2:])
    return run_dir.name


def _valid_mean(value) -> Optional[float]:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v <= 0.0:
        return None
    return v


def _valid_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        v = int(value)
    except (TypeError, ValueError):
        return None
    return v if v >= 0 else None


def _approach_order_key(name: str) -> Tuple[int, str]:
    preferred = ["pierce", "cgal", "touch", "tdbase", "exact", "direct_estimation", "estimated", "estimated_mem10"]
    if name in preferred:
        return (preferred.index(name), name)
    return (len(preferred), name)

def _canonicalize_approaches(raw: Dict[str, float]) -> Dict[str, float]:
    out: Dict[str, float] = {}

    # Represent RaySpace with one canonical series to avoid duplicate legend entries
    # and empty slots when one internal mode is missing for a given dataset group.
    if raw.get("pierce") is not None:
        out["pierce"] = raw["pierce"]
    for k in ("direct_estimation", "estimated", "exact", "estimated_mem10"):
        v = raw.get(k)
        if v is not None:
            out["pierce"] = v
            break

    for src, dst in (("cgal", "cgal"), ("touch", "touch"), ("tdbase", "tdbase")):
        v = raw.get(src)
        if v is not None:
            out[dst] = v
    return out


def _extract_latest_complete_nu_profile(
    path: Path,
    *,
    dataset_profile: str,
    group_name_factory,
    required_approach: Optional[str] = None,
) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    if md.get("dataset_profile") != dataset_profile:
        return None

    results = payload.get("results", {})
    counts = results.get("counts", [])
    if not counts:
        return None

    approaches = results.get("enabled_approaches") or md.get("approaches") or []
    if not approaches:
        return None

    selected_idx = None
    approach_to_mean: Dict[str, float] = {}
    result_size = None
    for idx in sorted(range(len(counts)), key=lambda i: counts[i], reverse=True):
        candidate_means: Dict[str, float] = {}
        for app in approaches:
            arr = (results.get(app) or {}).get("mean", [])
            if idx >= len(arr):
                continue
            mean = _valid_mean(arr[idx])
            if mean is not None:
                candidate_means[app] = mean
        if required_approach is not None and required_approach not in candidate_means:
            continue
        if not candidate_means:
            continue

        selected_idx = idx
        approach_to_mean = candidate_means
        sizes = results.get("result_sizes", [])
        if idx < len(sizes):
            result_size = _valid_int(sizes[idx])
        if result_size is None:
            intersection_sizes = results.get("num_intersections", [])
            if idx < len(intersection_sizes):
                result_size = _valid_int(intersection_sizes[idx])
        if result_size is None:
            return None
        break

    if selected_idx is None:
        return None

    run_dir = path.parent
    return GroupResult(
        group_name=group_name_factory(counts[selected_idx]),
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value=f"nu={counts[selected_idx]}",
        approach_to_mean=approach_to_mean,
        result_size=result_size,
    )


def _extract_latest_complete_nu_large(path: Path) -> Optional[GroupResult]:
    return _extract_latest_complete_nu_profile(
        path,
        dataset_profile="large_nu_v",
        group_name_factory=_large_nu_v_group_name,
    )


def _extract_latest_complete_nu_nn_large(path: Path) -> Optional[GroupResult]:
    return _extract_latest_complete_nu_profile(
        path,
        dataset_profile="large_nu_nn",
        group_name_factory=_large_nu_nn_group_name,
    )

def _extract_latest_tdbase_supported_nu_large(path: Path) -> Optional[GroupResult]:
    return _extract_latest_complete_nu_profile(
        path,
        dataset_profile="large_nu_v",
        group_name_factory=_large_nu_v_group_name,
        required_approach="tdbase",
    )


def _extract_latest_tdbase_supported_nu_nn_large(path: Path) -> Optional[GroupResult]:
    return _extract_latest_complete_nu_profile(
        path,
        dataset_profile="large_nu_nn",
        group_name_factory=_large_nu_nn_group_name,
        required_approach="tdbase",
    )


def _extract_microns_4gb(path: Path) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    md = payload.get("metadata", {})
    scenario = md.get("scenario")
    if scenario not in ("microns_overlap", None):
        return None

    results = payload.get("results", [])
    if not isinstance(results, list):
        return None

    row = next((r for r in results if r.get("size_gb") == 4), None)
    if row is None:
        return None

    approaches = md.get("approaches") or [k for k in row.keys() if k not in {"size_gb", "size_bytes_a", "size_bytes_b"}]
    approach_to_mean: Dict[str, float] = {}
    for app in approaches:
        res = row.get(app)
        if not isinstance(res, dict):
            continue
        if res.get("error"):
            continue
        mean = _valid_mean(res.get("mean"))
        if mean is not None:
            approach_to_mean[app] = mean

    if not approach_to_mean:
        return None

    # Prefer true intersection cardinality from direct_estimation/exact outputs.
    # Older microns runs stored row["result_size"] from estimated pairs, which is incorrect.
    result_size = None
    for app in ("direct_estimation", "exact", "pierce", "estimated"):
        res = row.get(app)
        if isinstance(res, dict):
            result_size = _valid_int(res.get("num_intersections"))
            if result_size is not None:
                break
    if result_size is None:
        result_size = _valid_int(row.get("result_size"))

    run_dir = path.parent
    return GroupResult(
        group_name=r"Neurons$_1$ $\bowtie$ Neurons$_2$",
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value="size_gb=4",
        approach_to_mean=approach_to_mean,
        result_size=result_size,
    )


def _extract_cube_largest(path: Path) -> Optional[GroupResult]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return None

    counts = results.get("counts", [])
    if not counts:
        return None

    max_count = max(counts)
    idx = counts.index(max_count)
    candidate_approaches = [
        a for a in ["exact", "estimated", "cgal", "touch", "direct_estimation", "tdbase"]
        if a in results
    ]

    approach_to_mean: Dict[str, float] = {}
    for app in candidate_approaches:
        arr = (results.get(app) or {}).get("mean", [])
        if idx >= len(arr):
            continue
        mean = _valid_mean(arr[idx])
        if mean is not None:
            approach_to_mean[app] = mean

    if not approach_to_mean:
        return None

    sizes = results.get("result_sizes", [])
    result_size = _valid_int(sizes[idx]) if idx < len(sizes) else None
    if result_size is None:
        return None

    run_dir = path.parent
    return GroupResult(
        group_name=r"Cubes$_1$ $\bowtie$ Cubes$_2$",
        run_dir=run_dir,
        run_timestamp=_parse_ts_from_run_dir(run_dir),
        selector_value=f"count={max_count}",
        approach_to_mean=approach_to_mean,
        result_size=result_size,
    )


def _pick_latest_usable(runs_root: Path, prefix: str, extractor) -> GroupResult:
    run_dirs = sorted(runs_root.glob(f"{prefix}_*"), reverse=True)
    for run_dir in run_dirs:
        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue
        try:
            extracted = extractor(result_file)
        except Exception as exc:
            print(f"Skipping {result_file}: failed to parse ({exc})")
            continue
        if extracted is not None:
            return extracted
    raise RuntimeError(f"No usable run found for {prefix}")

def _pick_latest_usable_with_required_approach(
    runs_root: Path,
    prefix: str,
    extractor,
    required_approach: str,
) -> GroupResult:
    run_dirs = sorted(runs_root.glob(f"{prefix}_*"), reverse=True)
    for run_dir in run_dirs:
        result_file = run_dir / "results.json"
        if not result_file.exists():
            continue
        try:
            extracted = extractor(result_file)
        except Exception as exc:
            print(f"Skipping {result_file}: failed to parse ({exc})")
            continue
        if extracted is None:
            continue
        canonical = _canonicalize_approaches(extracted.approach_to_mean)
        if required_approach in canonical:
            extracted.approach_to_mean = canonical
            return extracted
    raise RuntimeError(f"No usable run found for {prefix} with approach '{required_approach}'")


def _plot_grouped_bars(groups: List[GroupResult], output_base: Path) -> None:
    apply_side_by_side_style()
    plt.rcParams["hatch.linewidth"] = 1.2

    for g in groups:
        g.approach_to_mean = _canonicalize_approaches(g.approach_to_mean)

    all_approaches = {a for g in groups for a in g.approach_to_mean.keys()}
    # Ensure TDBase is included in the comparison even if missing from all selected runs,
    # so we can show it as 'missing/failed' rather than just absent.
    all_approaches.add("tdbase")
    all_approaches = sorted(all_approaches, key=_approach_order_key)

    num_apps = len(all_approaches)
    x = np.arange(len(groups), dtype=float)
    width = 0.8 / max(1, num_apps)

    fig, ax = plt.subplots(figsize=(6.2, 4.1))
    for gi, g in enumerate(groups):
        for pi, app in enumerate(all_approaches):
            offset = (pi - (num_apps - 1) / 2.0) * width
            st = style_for(app)

            if app in g.approach_to_mean:
                ax.bar(
                    x[gi] + offset,
                    g.approach_to_mean[app],
                    width=width,
                    color=st["color"],
                    alpha=0.92,
                    hatch=st.get("hatch", ""),
                    edgecolor="black",
                    linewidth=0.6,
                )
            elif app == "tdbase":
                # For TDBase, if it's missing, draw an 'X' to indicate it couldn't run
                # rather than just leaving an empty space or skipping it.
                # Draw a very short 'failure' bar so the X has a base and stays within the plot area
                ax.bar(x[gi] + offset, 1.0, width=width, color=st["color"], alpha=0.2, edgecolor="black", linewidth=0.5)
                ax.text(
                    x[gi] + offset,
                    1.1,  # Slightly above the 1.0 floor for visibility
                    "X",
                    ha="center",
                    va="bottom",
                    color=st["color"],
                    fontsize=22,
                    fontweight="bold",
                )

    ax.set_yscale("log")
    all_vals = [v for g in groups for v in g.approach_to_mean.values()]
    set_log_timing_axis_limits(ax, all_vals, floor=1.0)
    _, current_top = ax.get_ylim()
    ax.set_ylim(top=current_top * 3.0)
    ax.set_ylabel("Query Time (ms)", fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [g.group_name.replace(r" $\bowtie$ ", "\n" + r"$\bowtie$" + " ") for g in groups],
        fontsize=11,
    )
    ax.tick_params(axis="y", labelsize=14)
    ax.grid(False)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.6)
    legend_handles = [
        Patch(
            facecolor=style_for(app)["color"],
            edgecolor="black",
            hatch=style_for(app).get("hatch", ""),
            linewidth=0.6,
            alpha=0.92,
            label=style_for(app)["label"],
        )
        for app in all_approaches
    ]
    make_legend_bold(
        ax,
        legend_handles,
        [handle.get_label() for handle in legend_handles],
        loc="upper left",
        fontsize=11,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=2.6,
        handleheight=1.4,
        handletextpad=0.8,
    )

    plt.tight_layout(pad=0.6)
    plt.savefig(f"{output_base}.png", dpi=300)
    plt.savefig(f"{output_base}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create grouped bar chart for overlap overall performance using latest usable runs: "
            "large nu (highest TDBase-supported nu from the latest usable large_nu_v run), "
            "large nu nn (highest TDBase-supported nu from the latest usable large_nu_nn run), "
            "MICrONS (4GB), cube scalability (largest dataset)."
        )
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=SCRIPT_DIR / "runs",
        help="Directory that contains overlap run folders.",
    )
    args = parser.parse_args()

    runs_root = args.runs_root

    group_nu = _pick_latest_usable(runs_root, "overlap_nu_scalability", _extract_latest_tdbase_supported_nu_large)
    group_nu_nn = _pick_latest_usable(runs_root, "overlap_nu_scalability", _extract_latest_tdbase_supported_nu_nn_large)
    group_microns = _pick_latest_usable_with_required_approach(
        runs_root, "overlap_microns", _extract_microns_4gb, "pierce"
    )
    group_cube = _pick_latest_usable(runs_root, "overlap_cube_scalability", _extract_cube_largest)

    groups = [group_nu, group_nu_nn, group_microns, group_cube]

    run_layout = create_benchmark_run_layout(SCRIPT_DIR, "overlap_overall_performance")
    figures_dir = Path(run_layout["figures_dir"])
    output_base = figures_dir / f"mesh_overlap_overall_performance_{run_layout['timestamp']}"

    _plot_grouped_bars(groups, output_base)

    summary_payload = {
        "metadata": {
            "scenario": "overlap_overall_performance",
            "timestamp": run_layout["timestamp"],
            "run_name": run_layout["run_name"],
        },
        "groups": [
            {
                "group_name": g.group_name,
                "selector": g.selector_value,
                "source_run": str(g.run_dir),
                "source_run_name": g.run_dir.name,
                "source_timestamp": g.run_timestamp,
                "result_size": g.result_size,
                "results_ms": g.approach_to_mean,
            }
            for g in groups
        ],
    }
    write_json(Path(run_layout["results_json"]), summary_payload)

    print(f"Saved figure: {output_base}.png")
    print(f"Saved figure: {output_base}.pdf")
    for g in groups:
        print(f"{g.group_name}: {g.run_dir.name} ({g.selector_value}), approaches={sorted(g.approach_to_mean.keys(), key=_approach_order_key)}")


if __name__ == "__main__":
    main()
