import argparse
import math
import os
import re
from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import torch
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from dataset.cifar100 import _get_data_root, mean, stdv
from models import model_dict


@dataclass(frozen=True)
class MethodSpec:
    name: str
    checkpoint_path: str


def parse_methods(values: Sequence[str]) -> List[MethodSpec]:
    if not values:
        raise ValueError("at least one --method NAME=CHECKPOINT is required")

    methods = []
    names = set()
    for value in values:
        if "=" not in value:
            raise ValueError("--method must use NAME=CHECKPOINT format")
        name, checkpoint_path = (part.strip() for part in value.split("=", 1))
        if not name or not checkpoint_path:
            raise ValueError("--method name and checkpoint path cannot be empty")
        if name in names:
            raise ValueError("duplicate method name: {}".format(name))
        methods.append(MethodSpec(name, checkpoint_path))
        names.add(name)
    return methods


def normalize_checkpoint_state(checkpoint: object) -> Mapping[str, torch.Tensor]:
    state = checkpoint.get("model", checkpoint) if isinstance(checkpoint, Mapping) else checkpoint
    if not isinstance(state, Mapping) or not all(
        isinstance(name, str) and isinstance(value, torch.Tensor)
        for name, value in state.items()
    ):
        raise ValueError("checkpoint must contain a model state dictionary")
    if not any(name.startswith("module.") for name in state):
        return state
    return {
        name[7:] if name.startswith("module.") else name: value
        for name, value in state.items()
    }


def normalize_cifar_root(data_path: str) -> str:
    normalized = os.path.normpath(data_path)
    if os.path.basename(normalized) == "cifar-100-python":
        return os.path.dirname(normalized)
    return normalized


def build_test_loader(
    data_path: str, batch_size: int, num_workers: int
) -> DataLoader:
    data_root = normalize_cifar_root(data_path)
    extracted_path = os.path.join(data_root, "cifar-100-python")
    if not os.path.isdir(extracted_path):
        raise FileNotFoundError(
            "offline CIFAR-100 data not found: {}".format(extracted_path)
        )
    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=stdv),
        ]),
    )
    return DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    if not re.fullmatch(r"cuda(?::[0-9]+)?", device_name):
        raise ValueError(
            "device must be auto, cpu, cuda, or cuda:N; got {}".format(
                device_name
            )
        )
    if not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if ":" in device_name:
        device_index = int(device_name.split(":", 1)[1])
        if device_index >= torch.cuda.device_count():
            raise ValueError("CUDA device {} is not available".format(device_name))
    return torch.device(device_name)


def load_resnet8x4(checkpoint_path: str, device: torch.device) -> nn.Module:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError("checkpoint not found: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = model_dict["resnet8x4"](num_classes=100)
    model.load_state_dict(normalize_checkpoint_state(checkpoint), strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model.to(device)


def extract_features(
    model: nn.Module, loader: Iterable, device: torch.device
) -> Tuple[numpy.ndarray, numpy.ndarray]:
    all_features = []
    all_labels = []
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            images = images.float().to(device)
            features, _ = model(images, is_feat=True)
            all_features.append(features[-2].flatten(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    if not all_features:
        raise ValueError("CIFAR-100 test loader is empty")
    return numpy.concatenate(all_features), numpy.concatenate(all_labels)


def fit_tsne(features: numpy.ndarray, seed: int) -> numpy.ndarray:
    return TSNE(random_state=seed).fit_transform(features)


def class_colors(num_classes: int = 100) -> numpy.ndarray:
    colormaps = (
        "tab20",
        "tab20b",
        "tab20c",
        "Set3",
        "Set2",
        "Accent",
        "Paired",
    )
    colors = []
    seen = set()
    for colormap_name in colormaps:
        colormap = plt.get_cmap(colormap_name)
        for index in range(colormap.N):
            color = colormap(index)
            key = tuple(numpy.round(color, 8))
            if key not in seen:
                colors.append(color)
                seen.add(key)
    fallback = plt.get_cmap("hsv")
    for index in range(num_classes):
        if len(colors) >= num_classes:
            break
        color = fallback(index / float(num_classes))
        key = tuple(numpy.round(color, 8))
        if key not in seen:
            colors.append(color)
            seen.add(key)
    return numpy.asarray(colors[:num_classes])


def subplot_grid(panel_count: int) -> Tuple[int, int]:
    if panel_count < 1:
        raise ValueError("panel count must be positive")
    columns = min(3, int(math.ceil(math.sqrt(panel_count))))
    rows = int(math.ceil(panel_count / float(columns)))
    return rows, columns


def plot_comparison(
    results: Sequence[Tuple[str, numpy.ndarray, numpy.ndarray]],
    output_path: str,
) -> matplotlib.figure.Figure:
    if not results:
        raise ValueError("at least one t-SNE result is required")

    rows, columns = subplot_grid(len(results))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(5.0 * columns, 5.0 * rows),
        squeeze=False,
    )
    palette = class_colors()
    flat_axes = axes.ravel()
    for axis, (name, features, labels) in zip(flat_axes, results):
        for class_index in range(100):
            mask = labels == class_index
            axis.scatter(
                features[mask, 0],
                features[mask, 1],
                color=palette[class_index],
                s=1,
                alpha=0.4,
            )
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_box_aspect(1)
    for axis in flat_axes[len(results):]:
        axis.set_visible(False)

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare ResNet8x4 knowledge-distillation checkpoints with direct "
            "t-SNE on all CIFAR-100 test features."
        )
    )
    parser.add_argument(
        "--method",
        dest="methods",
        action="append",
        required=True,
        metavar="NAME=CHECKPOINT",
        help="method title and checkpoint path; repeat for each method",
    )
    parser.add_argument(
        "--data-path",
        default=_get_data_root(),
        help="CIFAR-100 root or extracted cifar-100-python directory",
    )
    parser.add_argument(
        "--output",
        default="figure/tsne_distillation_comparison.png",
        help="output path for the multi-panel figure",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a concrete device such as cuda:0",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random_state supplied to every independent t-SNE fit",
    )
    return parser


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    methods = parse_methods(args.methods)
    device = resolve_device(args.device)
    loader = build_test_loader(
        args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    results = []
    reference_labels = None
    for method in methods:
        print("[{}] extracting CIFAR-100 features".format(method.name))
        model = load_resnet8x4(method.checkpoint_path, device)
        features, labels = extract_features(model, loader, device)
        if reference_labels is None:
            reference_labels = labels
        elif not numpy.array_equal(labels, reference_labels):
            raise ValueError(
                "CIFAR-100 sample labels differ for method {}".format(method.name)
            )
        print("[{}] fitting direct t-SNE".format(method.name))
        embedded = fit_tsne(features, args.seed)
        results.append((method.name, embedded, labels))
        del model, features

    figure = plot_comparison(results, args.output)
    plt.close(figure)
    print("Saved t-SNE comparison to {}".format(args.output))


if __name__ == "__main__":
    main()
