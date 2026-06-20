import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from pathlib import Path


def _register_matplotlib_cm_fonts():
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    for font_name in (
        "cmr10.ttf",
        "cmb10.ttf",
        "cmmi10.ttf",
        "cmsy10.ttf",
        "cmex10.ttf",
        "cmtt10.ttf",
    ):
        font_path = font_dir / font_name
        if font_path.exists():
            font_manager.fontManager.addfont(str(font_path))


_register_matplotlib_cm_fonts()

PAPER_FIGSIZE = (10.0, 7.2)
PAPER_WIDE_FIGSIZE = PAPER_FIGSIZE
PAPER_SIDE_BY_SIDE_FIGSIZE = (8.0, 4.8)

APPROACH_PALETTE = [
    "#9ebcda", # TOUCH
    "#8856a7", # Pierce
    "#8c96c6", # Face (cgal)
    "#810f7c", # TDBase
]

QUERY_PALETTE = {
    "overlap": "#e0ecf4",
    "intersection": "#9ebcda",
    "containment": "#8856a7",
}

APPROACH_STYLES = {
    "pierce": {"label": "Pierce", "color": APPROACH_PALETTE[1], "marker": "o", "hatch": "\\", "linestyle": "-"},
    "exact": {"label": "Pierce (Two Pass)", "color": APPROACH_PALETTE[1], "marker": "o", "hatch": "/", "linestyle": "-"},
    "direct_estimation": {"label": "Pierce", "color": APPROACH_PALETTE[1], "marker": "s", "hatch": "\\", "linestyle": "--"},
    "estimated": {"label": "Pierce (Estimated)", "color": APPROACH_PALETTE[1], "marker": "^", "hatch": "\\", "linestyle": ":"},
    "estimated_mem10": {"label": "Pierce (10 GB fixed)", "color": APPROACH_PALETTE[1], "marker": "D", "hatch": "x", "linestyle": "-."},
    "cgal": {"label": "Face", "color": APPROACH_PALETTE[2], "marker": "v", "hatch": "-", "linestyle": "-"},
    "touch": {"label": "TOUCH", "color": APPROACH_PALETTE[0], "marker": "p", "hatch": "+", "linestyle": "-"},
    "tdbase": {"label": "TDBase", "color": APPROACH_PALETTE[3], "marker": "P", "hatch": ".", "linestyle": "-."},
}

HATCH_PATTERNS = ["/", "\\", "x", "-", "+", ".", "o", "*"]


def apply_paper_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["cmb10", "cmr10"],
        "font.weight": "bold",
        "font.size": 16,
        "mathtext.fontset": "cm",
        "axes.formatter.use_mathtext": True,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.labelsize": 17,
        "axes.titlesize": 17,
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.linewidth": 1.6,
        "xtick.labelsize": 15,
        "ytick.labelsize": 15,
        "legend.fontsize": 14,
        "legend.title_fontsize": 14,
        "lines.linewidth": 2.8,
        "lines.markersize": 9,
        "axes.grid": False,
    })

def apply_side_by_side_style():
    apply_paper_style()
    plt.rcParams.update({
        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "legend.title_fontsize": 10,
        "lines.linewidth": 2.2,
        "lines.markersize": 7,
    })

def make_legend_bold(ax, *args, **kwargs):
    legend = ax.legend(*args, **kwargs)
    if legend is None:
        return None
    for text in legend.get_texts():
        text.set_fontweight("bold")
    title = legend.get_title()
    if title is not None:
        title.set_fontweight("bold")
    return legend


def style_for(approach: str):
    return APPROACH_STYLES.get(approach, {"label": approach, "color": "#444444", "marker": "o", "hatch": "oo"})


def query_style_for(query: str):
    styles = {
        "overlap": {"label": "Overlap", "color": QUERY_PALETTE["overlap"], "hatch": "/"},
        "intersection": {"label": "Intersection", "color": QUERY_PALETTE["intersection"], "hatch": "\\"},
        "containment": {"label": "Containment", "color": QUERY_PALETTE["containment"], "hatch": "x"},
    }
    return styles.get(query, {"label": query.capitalize(), "color": "#444444", "hatch": "oo"})


def hatch_for_index(idx: int) -> str:
    return HATCH_PATTERNS[idx % len(HATCH_PATTERNS)]


def plot_mean_series(ax, xs, ys, approach: str):
    st = style_for(approach)
    ax.plot(xs, ys, linestyle=st.get("linestyle", "-"), marker=st["marker"], color=st["color"], label=st["label"])

def set_log_timing_axis_limits(ax, values, *, floor: float = 1.0, padding_factor: float = 0.8):
    """Set a readable lower bound for log-scale timing axes.

    Lower bound rule: start at 10^0 ms or slightly below the observed minimum,
    whichever is lower.
    """
    positive = [
        float(v)
        for v in values
        if isinstance(v, (int, float)) and np.isfinite(v) and float(v) > 0.0
    ]
    if not positive:
        return

    observed_min = min(positive)
    lower_bound = min(float(floor), observed_min * float(padding_factor))
    if lower_bound <= 0.0:
        lower_bound = min(float(floor), observed_min)
    ax.set_ylim(bottom=lower_bound)


def generate_scalability_figure(results, approaches, figures_dir: Path, timestamp: str,
                                scenario_name: str, x_axis_key: str, x_axis_label: str,
                                y_axis_label: str, title: str):
    """Generate a line plot for scalability from successful runs."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    plt.figure(figsize=PAPER_FIGSIZE)
    has_any_series = False

    all_y_vals = []
    for approach in approaches:
        x_vals = []
        y_vals = []
        for row in results:
            res = row.get(approach)
            if not isinstance(res, dict) or "error" in res:
                continue
            mean = res.get("mean")
            if mean is None:
                continue
            x_vals.append(row.get(x_axis_key))
            y_vals.append(mean)

        if not x_vals:
            continue

        has_any_series = True
        sorted_points = sorted(zip(x_vals, y_vals), key=lambda t: (t[0] if t[0] is not None else 0))
        xs = [p[0] for p in sorted_points]
        ys = [p[1] for p in sorted_points]
        all_y_vals.extend(ys)
        plot_mean_series(plt.gca(), xs, ys, approach)

    if not has_any_series:
        print(f"No successful approach results available; skipping {scenario_name} scalability figure.")
        plt.close()
        return

    plt.yscale("log")
    set_log_timing_axis_limits(plt.gca(), all_y_vals)
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.grid(False)
    make_legend_bold(plt.gca())
    plt.tight_layout()

    output_base = figures_dir / f"{scenario_name}_scalability_{timestamp}"
    plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {output_base}.png")


def generate_breakdown_figure(results, approaches, figures_dir: Path, timestamp: str,
                              scenario_name: str, x_axis_key: str, x_axis_label: str,
                              y_axis_label: str, title: str):
    """Generate a stacked bar chart showing the runtime breakdown."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    apply_paper_style()

    for approach in approaches:
        x_labels = []
        breakdown_keys = set()
        data_points = []

        for row in results:
            res = row.get(approach)
            if not isinstance(res, dict) or "error" in res or "breakdown" not in res:
                continue
            x_labels.append(str(row.get(x_axis_key)))
            data_points.append(res["breakdown"])
            breakdown_keys.update(res["breakdown"].keys())

        if not data_points:
            continue

        sorted_keys = sorted(list(breakdown_keys))

        # Preferred intersection phase order (bottom -> top in stacked bars).
        phase_order = [
            "selectivity estimation",
            "raytrace_overlap_hash_mesh1tomesh2",
            "raytrace_overlap_hash_mesh2tomesh1",
            "raytrace_containment_hash_mesh1tomesh2",
            "raytrace_containment_hash_mesh2tomesh1",
            "compact_hash_table_pairs",
        ]
        phase_labels = {
            "selectivity estimation": "Selectivity Estimation",
            "raytrace_overlap_hash_mesh1tomesh2": "Edge (M1->M2)",
            "raytrace_overlap_hash_mesh2tomesh1": "Edge (M2 -> M1)",
            "raytrace_containment_hash_mesh1tomesh2": "Containment (M1-M2)",
            "raytrace_containment_hash_mesh2tomesh1": "Containment (M2 -> M1)",
            "compact_hash_table_pairs": "Download results",
            "download results": "Download results",
        }

        present_preferred = [k for k in phase_order if k in breakdown_keys]
        remaining = [k for k in sorted_keys if k not in present_preferred]
        display_order = present_preferred + remaining

        plt.figure(figsize=PAPER_WIDE_FIGSIZE)
        bottom = np.zeros(len(x_labels))
        colors = plt.cm.tab10.colors

        # Draw in bottom->top order so the first phase is at the bottom.
        for i, key in enumerate(display_order):
            vals = np.array([dp.get(key, 0.0) for dp in data_points])
            plt.bar(
                x_labels,
                vals,
                bottom=bottom,
                label=phase_labels.get(key, key),
                color=colors[i % len(colors)],
                hatch=hatch_for_index(i),
                edgecolor="black",
                linewidth=0.5,
            )
            bottom += vals

        plt.xlabel(x_axis_label)
        plt.ylabel(y_axis_label)
        make_legend_bold(plt.gca(), bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(False)
        plt.tight_layout()

        output_base = figures_dir / f"{scenario_name}_breakdown_{approach}_{timestamp}"
        plt.savefig(f"{output_base}.png", dpi=300, bbox_inches="tight")
        plt.savefig(f"{output_base}.pdf", bbox_inches="tight")
        plt.close()
        print(f"Saved figure: {output_base}.png")
