import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import torch

from distiller_zoo.AHTKD import probability_boundaries

DEFAULT_DATA_PATH = "/root/autodl-fs/data/cifar-100-python"
DEFAULT_OUTPUT_PATH = "figure/non_target_rank_distribution.png"


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    model_key: str
    checkpoint_path: str


DEFAULT_MODEL = ModelSpec(
    "ResNet56",
    "resnet56",
    "/root/autodl-tmp/kd/pretrained/resnet56-(72.34).pth",
)


def normalize_cifar_root(data_path):
    normalized = os.path.normpath(os.path.expanduser(data_path))
    if os.path.basename(normalized) == "cifar-100-python":
        return os.path.dirname(normalized)
    return normalized


def single_sample_probabilities(logits, target):
    if logits.ndim != 1:
        raise ValueError("logits must be one-dimensional")

    target = int(target)
    class_indices = torch.arange(logits.numel(), device=logits.device)
    global_probabilities = torch.softmax(logits.float(), dim=0)
    non_target_mask = class_indices != target
    non_target_probabilities = global_probabilities[non_target_mask]
    non_target_indices = class_indices[non_target_mask]
    sorted_probabilities, order = torch.sort(
        non_target_probabilities, descending=True
    )
    return (
        global_probabilities,
        sorted_probabilities,
        non_target_indices[order],
    )


def piecewise_boundary_fit(probabilities):
    if probabilities.ndim != 1 or probabilities.numel() < 2:
        raise ValueError(
            "probabilities must be one-dimensional with at least 2 ranks"
        )

    boundary = int(
        probability_boundaries(probabilities.unsqueeze(0)).item()
    )
    working_dtype = (
        torch.float64
        if probabilities.dtype == torch.float64
        else torch.float32
    )
    values = probabilities.to(dtype=working_dtype)
    ranks = torch.arange(
        1,
        values.numel() + 1,
        device=values.device,
        dtype=working_dtype,
    )
    design = torch.stack(
        [
            torch.ones_like(ranks),
            ranks,
            torch.clamp(ranks - boundary, min=0),
        ],
        dim=1,
    )
    coefficients = torch.linalg.pinv(design) @ values
    return boundary, design @ coefficients


def select_random_sample(dataset, seed):
    if len(dataset) == 0:
        raise ValueError("dataset must contain at least one sample")

    generator = torch.Generator().manual_seed(int(seed))
    sample_index = int(
        torch.randint(len(dataset), (1,), generator=generator).item()
    )
    image, target = dataset[sample_index]
    return (
        image.unsqueeze(0),
        torch.tensor([int(target)], dtype=torch.long),
        sample_index,
    )


def annotation_ranks(rank_count, boundary, leading_count=8):
    ranks = set(range(1, min(rank_count, leading_count) + 1))
    ranks.update(
        rank
        for rank in (boundary - 1, boundary, boundary + 1)
        if 1 <= rank <= rank_count
    )
    return sorted(ranks)


def draw_piecewise_boundary(
        axis, curve, fitted, boundary, label_ranks):
    import numpy as np

    curve = np.asarray(curve, dtype=np.float64)
    fitted = np.asarray(fitted, dtype=np.float64)
    ranks = np.arange(1, len(curve) + 1)

    axis.axvspan(
        0.5,
        boundary + 0.5,
        color="#4477AA",
        alpha=0.07,
        label="Head ranks",
    )
    axis.axvspan(
        boundary + 0.5,
        len(curve) + 0.5,
        color="#999999",
        alpha=0.07,
        label="Tail ranks",
    )
    probability_line, = axis.plot(
        ranks,
        curve,
        color="#4477AA",
        linewidth=1.1,
        marker="o",
        markersize=3.2,
        markeredgewidth=0,
        label="Teacher probability",
        zorder=3,
    )
    head_fit, = axis.plot(
        ranks[:boundary],
        fitted[:boundary],
        color="#D9822B",
        linewidth=2.2,
        label="Head fit",
        zorder=4,
    )
    tail_fit, = axis.plot(
        ranks[boundary - 1:],
        fitted[boundary - 1:],
        color="#D9822B",
        linewidth=2.2,
        linestyle=":",
        label="Tail fit",
        zorder=4,
    )
    boundary_line = axis.axvline(
        boundary,
        color="#B22222",
        linestyle="--",
        linewidth=1.5,
        zorder=5,
    )
    boundary_marker, = axis.plot(
        boundary,
        curve[boundary - 1],
        color="#B22222",
        marker="*",
        markersize=12,
        linestyle="None",
        markeredgecolor="white",
        markeredgewidth=0.8,
        label="Boundary k",
        zorder=6,
    )

    probability_label_box = axis.text(
        0.98,
        0.04,
        rank_probability_table(curve, None, label_ranks),
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=6.5,
        color="#4D4D4D",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
        zorder=7,
    )

    boundary_label = axis.annotate(
        "k={}\n(last Head rank)".format(boundary),
        xy=(boundary, curve[boundary - 1]),
        xytext=(16, 34),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
        fontweight="bold",
        color="#B22222",
        arrowprops={"arrowstyle": "->", "color": "#B22222"},
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#B22222",
            "linewidth": 0.8,
        },
        zorder=8,
    )
    return {
        "probability_line": probability_line,
        "head_fit": head_fit,
        "tail_fit": tail_fit,
        "boundary_line": boundary_line,
        "boundary_marker": boundary_marker,
        "probability_label_box": probability_label_box,
        "boundary_label": boundary_label,
    }


def network_output_path(output_path, spec):
    output = Path(output_path)
    return output.with_name(
        "{}_{}{}".format(output.stem, spec.model_key, output.suffix)
    )


def extract_state_dict(checkpoint):
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "Plot the AHTKD boundary process for one single CIFAR-100 "
            "test image using the ResNet56 teacher."
        )
    )
    parser.add_argument(
        "--data-path",
        default=DEFAULT_DATA_PATH,
        help=(
            "CIFAR-100 root or the extracted cifar-100-python directory "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT_PATH,
        help=(
            "base output path; the ResNet56 model key is appended "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="seed used to select one random test image (default: %(default)s)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:0 (default: %(default)s)",
    )
    return parser


def resolve_device(device_name):
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_test_dataset(data_path):
    from torchvision import datasets, transforms

    data_root = normalize_cifar_root(data_path)
    extracted_dir = os.path.join(data_root, "cifar-100-python")
    if not os.path.isdir(extracted_dir):
        raise FileNotFoundError(
            "CIFAR-100 directory not found: {}".format(extracted_dir)
        )

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5071, 0.4867, 0.4408],
                std=[0.2675, 0.2565, 0.2761],
            ),
        ]
    )
    return datasets.CIFAR100(
        root=data_root,
        train=False,
        transform=transform,
        download=False,
    )


def load_model(spec, device):
    from models import model_dict

    if not os.path.isfile(spec.checkpoint_path):
        raise FileNotFoundError(
            "{} checkpoint not found: {}".format(
                spec.display_name, spec.checkpoint_path
            )
        )

    model = model_dict[spec.model_key](num_classes=100)
    checkpoint = torch.load(spec.checkpoint_path, map_location="cpu")
    model.load_state_dict(extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def _probability_text(probability):
    if probability >= 1e-3:
        return "{:.4f}".format(probability)
    return "{:.2e}".format(probability)


def rank_probability_table(probabilities, class_indices, label_ranks):
    lines = ["Selected probabilities"]
    for rank in label_ranks:
        probability = probabilities[rank - 1]
        if class_indices is None:
            lines.append(
                "r={}, p={}".format(
                    rank, _probability_text(probability)
                )
            )
        else:
            lines.append(
                "r={}, c={}, p={}".format(
                    rank,
                    int(class_indices[rank - 1]),
                    _probability_text(probability),
                )
            )
    return "\n".join(lines)


def _style_probability_axis(axis):
    axis.set_ylim(bottom=0)
    axis.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.65)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_sample_process(
        spec,
        global_probabilities,
        sorted_probabilities,
        sorted_indices,
        target,
        prediction,
        sample_index,
        output_path,
):
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    boundary, fitted = piecewise_boundary_fit(sorted_probabilities)
    global_values = global_probabilities.detach().cpu().numpy()
    sorted_values = sorted_probabilities.detach().cpu().numpy()
    sorted_class_indices = sorted_indices.detach().cpu().numpy()
    fitted_values = fitted.detach().cpu().numpy()
    target = int(target)
    prediction = int(prediction)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(16.0, 5.2), constrained_layout=True
    )
    class_indices = np.arange(len(global_values))
    target_colors = np.full(len(global_values), "#7FA8C9", dtype=object)
    target_colors[target] = "#B22222"
    axes[0].bar(
        class_indices,
        global_values,
        width=0.82,
        color=target_colors,
        linewidth=0,
    )
    axes[0].annotate(
        "target c={}\np={}".format(
            target, _probability_text(global_values[target])
        ),
        xy=(target, global_values[target]),
        xytext=(12, 28),
        textcoords="offset points",
        color="#B22222",
        fontsize=8,
        fontweight="bold",
        arrowprops={"arrowstyle": "->", "color": "#B22222"},
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "edgecolor": "#B22222",
            "linewidth": 0.8,
        },
    )
    axes[0].set_title("(a) Global softmax over all classes")
    axes[0].set_xlabel("Class index")
    axes[0].set_ylabel("Teacher probability")
    axes[0].set_xlim(-0.8, len(global_values) - 0.2)
    axes[0].set_xticks([0, 20, 40, 60, 80, len(global_values) - 1])
    _style_probability_axis(axes[0])

    ranks = np.arange(1, len(sorted_values) + 1)
    axes[1].plot(
        ranks,
        sorted_values,
        color="#4477AA",
        linewidth=1.3,
        marker="o",
        markersize=3.0,
        markeredgewidth=0,
    )
    panel_two_labels = annotation_ranks(
        len(sorted_values), boundary, leading_count=5
    )
    axes[1].text(
        0.98,
        0.97,
        rank_probability_table(
            sorted_values, sorted_class_indices, panel_two_labels
        ),
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=6.5,
        color="#4D4D4D",
        bbox={
            "boxstyle": "round,pad=0.25",
            "facecolor": "white",
            "edgecolor": "#BFBFBF",
            "linewidth": 0.7,
            "alpha": 0.94,
        },
    )
    axes[1].set_title("(b) Remove target, then sort non-target classes")
    axes[1].set_xlabel("Non-target rank")
    axes[1].set_ylabel("Global-softmax probability")
    axes[1].set_xlim(1, len(sorted_values))
    axes[1].set_xticks([1, 10, 20, 40, 60, 80, len(sorted_values)])
    _style_probability_axis(axes[1])

    draw_piecewise_boundary(
        axes[2],
        sorted_values,
        fitted_values,
        boundary,
        annotation_ranks(len(sorted_values), boundary, leading_count=3),
    )
    axes[2].set_title("(c) Piecewise fit and AHTKD boundary")
    axes[2].set_xlabel("Non-target rank")
    axes[2].set_ylabel("Global-softmax probability")
    axes[2].set_xlim(1, len(sorted_values))
    axes[2].set_xticks([1, 10, 20, 40, 60, 80, len(sorted_values)])
    _style_probability_axis(axes[2])
    axes[2].legend(loc="upper right", fontsize=6.6, ncol=2)

    fig.suptitle(
        (
            "{} | CIFAR-100 test index {} | target {} | "
            "prediction {} | k={}"
        ).format(
            spec.display_name,
            sample_index,
            target,
            prediction,
            boundary,
        ),
        fontsize=11,
        fontweight="bold",
    )
    output = network_output_path(output_path, spec)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    vector_outputs = []
    for suffix in (".svg", ".pdf"):
        vector_output = output.with_suffix(suffix)
        fig.savefig(vector_output, bbox_inches="tight", facecolor="white")
        vector_outputs.append(str(vector_output))
    plt.close(fig)
    return {
        "boundary": boundary,
        "output_path": str(output),
        "vector_outputs": vector_outputs,
    }


def main():
    args = build_parser().parse_args()
    device = resolve_device(args.device)
    print("Using device: {}".format(device))

    test_dataset = build_test_dataset(args.data_path)
    image, target, sample_index = select_random_sample(
        test_dataset, args.seed
    )
    target_index = int(target.item())
    print(
        "Selected CIFAR-100 test index {} with target class {} "
        "using seed {}".format(sample_index, target_index, args.seed)
    )

    print("Evaluating {}...".format(DEFAULT_MODEL.display_name))
    model = load_model(DEFAULT_MODEL, device)
    with torch.inference_mode():
        logits = model(image.to(device, non_blocking=True))[0]
    prediction = int(logits.argmax().item())
    global_probabilities, sorted_probabilities, sorted_indices = (
        single_sample_probabilities(logits, target_index)
    )
    metadata = plot_sample_process(
        DEFAULT_MODEL,
        global_probabilities,
        sorted_probabilities,
        sorted_indices,
        target=target_index,
        prediction=prediction,
        sample_index=sample_index,
        output_path=args.output,
    )

    print(
        "Prediction: {} ({})".format(
            prediction,
            "correct" if prediction == target_index else "incorrect",
        )
    )
    print("Boundary k: {}".format(metadata["boundary"]))
    print("Sorted non-target probabilities:")
    for rank, (class_index, probability) in enumerate(
            zip(
                sorted_indices.detach().cpu().tolist(),
                sorted_probabilities.detach().cpu().tolist(),
            ),
            start=1,
    ):
        print(
            "  rank {:02d}: class {:02d}, probability {:.8e}".format(
                rank, class_index, probability
            )
        )
    print("Saved figure to {}".format(metadata["output_path"]))

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
