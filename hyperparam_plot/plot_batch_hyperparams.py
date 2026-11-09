import argparse
from pathlib import Path
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def rgb(red, green, blue):
    return red / 255, green / 255, blue / 255


OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_PATHS = {
    "VGG8": {
        "en": OUTPUT_DIR / "batch_size_hyperparams_vgg8_en.pdf",
        "zh": OUTPUT_DIR / "batch_size_hyperparams_vgg8.pdf",
    },
    "ResNet8×4": {
        "en": OUTPUT_DIR / "batch_size_hyperparams_resnet8x4_en.pdf",
        "zh": OUTPUT_DIR / "batch_size_hyperparams_resnet8x4.pdf",
    },
}

AXIS_LABELS = {
    "en": ("Batch size", "Accuracy (%)"),
    "zh": ("批大小", "准确率（%）"),
}
CHINESE_FONT_FAMILIES = ["SimSun", "Songti SC", "STSong"]

BATCH_VALUES = np.array([16, 32, 64, 128, 256])
X_POSITIONS = np.arange(len(BATCH_VALUES))

VGG8_ACCURACY_RUNS = np.array(
    [
        [71.05, 70.85, 70.79, 70.32, 70.66],
        [71.59, 71.61, 71.74, 71.80, 71.50],
        [71.89, 71.82, 71.45, 71.90, 71.34],
        [71.95, 71.78, 71.68, 71.23, 71.18],
        [70.82, 70.65, 70.62, 71.14, 70.73],
    ]
)

RESNET8X4_ACCURACY_RUNS = np.array(
    [
        [72.11, 72.18, 72.32, 72.43, 72.20],
        [72.99, 72.98, 73.32, 73.26, 73.44],
        [73.31, 73.25, 73.17, 73.46, 73.41],
        [73.06, 72.61, 72.70, 73.02, 73.19],
        [72.34, 71.94, 72.29, 71.86, 72.14],
    ]
)

VGG8_BASELINE_ACCURACY_RUNS = [
    [69.67, 69.27, 69.95],
    [70.75, 70.38, 70.64],
    [70.59, 70.94, 70.51, 70.52, 70.15],
    [70.44, 69.70, 70.10],
    [68.80, 68.27, 69.19],
]

RESNET8X4_BASELINE_ACCURACY_RUNS = [
    [72.00, 72.22, 71.81],
    [72.16, 72.37, 72.45],
    [72.63, 72.50, 72.52, 72.37],
    [71.95, 71.95, 71.79],
    [70.93, 70.79, 70.36],
]

FIG_SIZE = (5.5, 5.0)
LINE_WIDTH = 3.0
ERROR_LINE_WIDTH = 2
CAP_SIZE = 5
MARKER_SIZE = 8

PLOT_CONFIGS = [
    {
        "label": "VGG8",
        "runs": VGG8_ACCURACY_RUNS,
        "marker": "o",
        "color": rgb(82, 137, 185),
        "marker_face_color": rgb(82, 137, 185),
        "baseline_runs": VGG8_BASELINE_ACCURACY_RUNS,
        "output_paths": OUTPUT_PATHS["VGG8"],
        "zorder": 3,
    },
    {
        "label": "ResNet8×4",
        "runs": RESNET8X4_ACCURACY_RUNS,
        "marker": "s",
        "color": rgb(197, 57, 50),
        "marker_face_color": rgb(197, 57, 50),
        "baseline_runs": RESNET8X4_BASELINE_ACCURACY_RUNS,
        "output_paths": OUTPUT_PATHS["ResNet8×4"],
        "zorder": 5,
    },
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang", choices=AXIS_LABELS, default="en")
    return parser.parse_args(argv)


def configure_style(lang="en"):
    sns.set_theme(style="darkgrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"


def mean_and_std(runs):
    return (
        np.array([np.mean(values) for values in runs]),
        np.array([np.std(values, ddof=1) for values in runs]),
    )


def validate_data_lengths():
    for config in PLOT_CONFIGS:
        for key in ("runs", "baseline_runs"):
            if len(config[key]) != len(BATCH_VALUES):
                raise ValueError(
                    f"{config['label']} {key} has {len(config[key])} rows, "
                    f"but BATCH_VALUES has {len(BATCH_VALUES)} values."
                )


def set_axis_labels(ax, lang):
    x_label, y_label = AXIS_LABELS[lang]
    label_style = {"fontsize": 20, "fontweight": "bold"}
    if lang == "zh":
        label_style["fontfamily"] = CHINESE_FONT_FAMILIES
    ax.set_xlabel(x_label, **label_style)
    ax.set_ylabel(y_label, **label_style)


def plot_batch_hyperparams(lang="en"):
    validate_data_lengths()

    for config in PLOT_CONFIGS:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=600, facecolor="white")
        means, stds = mean_and_std(config["runs"])
        baseline_means, baseline_stds = mean_and_std(config["baseline_runs"])
        y_min = min((means - stds).min(), (baseline_means - baseline_stds).min())
        y_max = max((means + stds).max(), (baseline_means + baseline_stds).max())

        ax.plot(
            X_POSITIONS,
            means,
            linestyle="-",
            marker=config["marker"],
            color=config["color"],
            linewidth=LINE_WIDTH,
            markersize=MARKER_SIZE,
            markerfacecolor=config["marker_face_color"],
            markeredgecolor=config["marker_face_color"],
            markeredgewidth=0,
            label=config["label"],
            zorder=config["zorder"],
        )

        baseline_label = f"{config['label']} Baseline"
        ax.plot(
            X_POSITIONS,
            baseline_means,
            color=config["color"],
            linestyle="--",
            linewidth=2.4,
            label=baseline_label,
            zorder=2,
        )

        for series_means, series_stds, zorder in (
            (means, stds, config["zorder"] + 1),
            (baseline_means, baseline_stds, 3),
        ):
            for index, x_position in enumerate(X_POSITIONS):
                ax.errorbar(
                    x_position,
                    series_means[index],
                    yerr=series_stds[index],
                    fmt="none",
                    ecolor=config["color"],
                    elinewidth=ERROR_LINE_WIDTH,
                    capsize=CAP_SIZE,
                    capthick=ERROR_LINE_WIDTH,
                    zorder=zorder,
                )

        y_margin = max((y_max - y_min) * 0.15, 0.25)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)

        set_axis_labels(ax, lang)
        ax.set_xticks(X_POSITIONS)
        ax.set_xticklabels(BATCH_VALUES)
        ax.tick_params(axis="both", labelsize=16)

        ax.yaxis.grid(True, linestyle="-", color="white", linewidth=1.2)
        ax.xaxis.grid(True, linestyle="-", color="white", linewidth=1.2)
        ax.set_axisbelow(True)

        for spine in ax.spines.values():
            spine.set_visible(False)

        handles, labels = ax.get_legend_handles_labels()
        legend_order = [config["label"], baseline_label]
        ordered_handles = [handles[labels.index(label)] for label in legend_order if label in labels]
        ordered_labels = [label for label in legend_order if label in labels]
        ax.legend(ordered_handles, ordered_labels, loc="best", fontsize=14, frameon=True, edgecolor="black")
        fig.tight_layout()

        output_path = config["output_paths"][lang]
        fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"Saved: {output_path}")


def main(argv=None):
    args = parse_args(argv)
    configure_style(args.lang)
    plot_batch_hyperparams(args.lang)


if __name__ == "__main__":
    main()
