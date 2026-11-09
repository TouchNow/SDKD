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
        "en": OUTPUT_DIR / "inarank_beta_hyperparams.pdf",
        "zh": OUTPUT_DIR / "inarank_beta_hyperparams_zh.pdf",
    },
    "ResNet8×4": {
        "en": OUTPUT_DIR / "inarank_beta_hyperparams_resnet8x4.pdf",
        "zh": OUTPUT_DIR / "inarank_beta_hyperparams_resnet8x4_zh.pdf",
    },
}

AXIS_LABELS = {
    "en": (r"Value of $\beta$", "Accuracy (%)"),
    "zh": (r"$\beta$ 的取值", "准确率（%）"),
}
CHINESE_FONT_FAMILIES = ["SimSun", "Songti SC", "STSong"]

HYPERPARAM_VALUES = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])

VGG8_ACCURACY_RUNS = np.array(
    [
        [71.84, 71.91, 72.08, 71.75, 71.89],
        [71.89, 71.82, 71.45, 71.90, 71.34],
        [71.54, 72.19, 71.60, 71.88, 71.96],
        [71.17, 71.50, 71.61, 71.59, 71.44],
        [71.33, 71.50, 71.30, 71.61, 71.56],
        [70.99, 71.07, 71.19, 70.80, 70.53],
    ]
)

RESNET8X4_ACCURACY_RUNS = np.array(
    [
        [73.51, 73.06, 73.38, 73.33, 73.14],
        [73.31, 73.25, 73.17, 73.46, 73.41],
        [73.43, 73.38, 73.12, 73.19, 73.10],
        [72.82, 72.86, 73.31, 73.22, 73.05],
        [72.93, 72.57, 72.69, 72.98, 72.75],
        [72.15, 72.37, 71.99, 72.34, 71.99],
    ]
)

VGG8_BASELINE_ACCURACY = 70.542
RESNET8X4_BASELINE_ACCURACY = 72.50

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
        "baseline_accuracy": VGG8_BASELINE_ACCURACY,
        "output_paths": OUTPUT_PATHS["VGG8"],
        "zorder": 3,
    },
    {
        "label": "ResNet8×4",
        "runs": RESNET8X4_ACCURACY_RUNS,
        "marker": "s",
        "color": rgb(197, 57, 50),
        "marker_face_color": rgb(197, 57, 50),
        "baseline_accuracy": RESNET8X4_BASELINE_ACCURACY,
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
    return runs.mean(axis=1), runs.std(axis=1, ddof=1)


def validate_data_lengths():
    for config in PLOT_CONFIGS:
        if len(config["runs"]) != len(HYPERPARAM_VALUES):
            raise ValueError(
                f"{config['label']} has {len(config['runs'])} rows, "
                f"but HYPERPARAM_VALUES has {len(HYPERPARAM_VALUES)} values."
            )


def set_axis_labels(ax, lang):
    x_label, y_label = AXIS_LABELS[lang]
    label_style = {"fontsize": 20, "fontweight": "bold"}
    if lang == "zh":
        label_style["fontfamily"] = CHINESE_FONT_FAMILIES
    ax.set_xlabel(x_label, **label_style)
    ax.set_ylabel(y_label, **label_style)


def plot_beta_hyperparams(lang="en"):
    validate_data_lengths()

    for config in PLOT_CONFIGS:
        fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=600, facecolor="white")
        means, stds = mean_and_std(config["runs"])
        y_min = min((means - stds).min(), config["baseline_accuracy"])
        y_max = max((means + stds).max(), config["baseline_accuracy"])

        ax.plot(
            HYPERPARAM_VALUES,
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

        for index, x_value in enumerate(HYPERPARAM_VALUES):
            ax.errorbar(
                x_value,
                means[index],
                yerr=stds[index],
                fmt="none",
                ecolor=config["color"],
                elinewidth=ERROR_LINE_WIDTH,
                capsize=CAP_SIZE,
                capthick=ERROR_LINE_WIDTH,
                zorder=config["zorder"] + 1,
            )

        baseline_label = f"{config['label']} Baseline"
        ax.axhline(
            config["baseline_accuracy"],
            color=config["color"],
            linestyle="--",
            linewidth=2.4,
            label=baseline_label,
            zorder=2,
        )

        y_margin = max((y_max - y_min) * 0.15, 0.25)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        set_axis_labels(ax, lang)
        ax.set_xticks(HYPERPARAM_VALUES)
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
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="best",
            fontsize=14,
            frameon=True,
            edgecolor="black",
        )
        fig.tight_layout()

        output_path = config["output_paths"][lang]
        fig.savefig(output_path, format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.05)
        plt.close(fig)

        print(f"Saved: {output_path}")


def main(argv=None):
    args = parse_args(argv)
    configure_style(args.lang)
    plot_beta_hyperparams(args.lang)


if __name__ == "__main__":
    main()
