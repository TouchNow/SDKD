#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Validate the teacher score as a relative recognition-difficulty proxy.

Place this file in CTKD-main, sync it to the server, and run:
    CUDA_VISIBLE_DEVICES=0 python analyze_sdkd_difficulty.py

To redraw the latest completed analysis without running model inference:
    python analyze_sdkd_difficulty.py --plot-from
To redraw a specific result directory or its bins.csv:
    python analyze_sdkd_difficulty.py --plot-from /path/to/result/bins.csv
The figure is English only and saved as difficulty_analysis.pdf at dpi=500.

Defaults use the server paths supplied for this experiment. A relative path is
looked up under the working directory, the home directory, and /, in that order.
No training, random augmentation, or dataset download is performed.

Models: teacher=resnet32x4; baseline=resnet8x4; SDKD=resnet8x4.
Data: all 10,000 CIFAR-100 test examples, including teacher mistakes.
Score: (target logit - mean of all 100 logits) / population standard deviation.
Bins: sort all test examples by that score, then split into 10 equal-size bins.
Primary statistic: sample-level Spearman(score, baseline cross entropy), with a
paired example-bootstrap percentile confidence interval. SDKD is supplementary.

The script imports models/resnet.py from YOUR project, rather than defining a
second ResNet. Parameters are loaded strictly after removing wrapper prefixes.
Only use checkpoint/pickle files from your own trusted experiment.

Dependencies: existing CTKD PyTorch environment, numpy, scipy, matplotlib, seaborn.
Compatible with Python 3.8+ and PyTorch 1.8+; torchvision is not needed.

Reference for the original CTKD model and test preprocessing:
https://github.com/zhengli97/CTKD/blob/master/models/resnet.py
https://github.com/zhengli97/CTKD/blob/master/dataset/cifar100.py
"""

import argparse
import csv
import importlib.util
import json
import os
import pickle
import random
import sys
import tempfile
import time
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "matplotlib-cache"))

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset


SERVER_PROJECT = "autodl-tmp/CTKD-main"
DEFAULT_MEAN = (0.5071, 0.4867, 0.4408)
DEFAULT_STD = (0.2675, 0.2565, 0.2761)
FIG_SIZE = (5.5, 5.0)
FIG_DPI = 500
LINE_WIDTH = 1.6
ERROR_LINE_WIDTH = 0.9
CAP_SIZE = 2.5
MARKER_SIZE = 4.5
MARKER_EDGE_WIDTH = 1.0
OUTPUT_FILENAME = "difficulty_analysis.pdf"
X_LABEL = "Teacher score bin"
Y_LABEL = "Error (%)"


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project-root", default=SERVER_PROJECT)
    parser.add_argument("--model-file", default=None,
                        help="Alternative ResNet definition file; default: PROJECT/models/resnet.py")
    parser.add_argument("--teacher-ckpt", default=SERVER_PROJECT + "/save/resnet32x4-(79.42).pth")
    parser.add_argument("--baseline-ckpt", default=SERVER_PROJECT + "/save/vanilla/resnet8x4_last.pth")
    parser.add_argument("--sdkd-ckpt", default=SERVER_PROJECT + "/save/sdkd/resnet8x4-(76.63).pth")
    parser.add_argument("--data-root", default="autodl-fs/data/cifar-100-python")
    parser.add_argument("--teacher-arch", default="resnet32x4")
    parser.add_argument("--student-arch", default="resnet8x4")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--bootstrap", type=int, default=1000,
                        help="Paired sample bootstrap repetitions for Spearman 95%% CI; 0 disables CI")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--mean", nargs=3, type=float, default=DEFAULT_MEAN)
    parser.add_argument("--std", nargs=3, type=float, default=DEFAULT_STD)
    parser.add_argument("--output-dir", default=None,
                        help="Default: PROJECT/output/difficulty_analysis_YYYYMMDD_HHMMSS")
    parser.add_argument("--plot-from", nargs="?", const="latest", default=None,
                        help="Redraw bins.csv or a result directory; omit the value to use the latest result")
    args = parser.parse_args()
    if args.batch_size < 1 or args.cpu_threads < 1:
        parser.error("batch-size and cpu-threads must be positive")
    if not 2 <= args.bins <= 10000 or args.bootstrap < 0:
        parser.error("bins must be between 2 and 10000; bootstrap must be nonnegative")
    if any(value <= 0 for value in args.std):
        parser.error("Every input normalization standard deviation must be positive")
    return args


def resolve_path(raw, extra_roots=()):
    path = Path(raw).expanduser()
    if path.is_absolute():
        candidates = [path]
    else:
        candidates = [Path.cwd() / path, Path.home() / path, Path("/") / path]
        candidates.extend(Path(root) / path for root in extra_roots)
    unique = list(dict.fromkeys(candidate.resolve() for candidate in candidates))
    for candidate in unique:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Cannot locate {!r}. Tried: {}".format(str(raw), "; ".join(map(str, unique)))
    )


def resolve_project_root(args):
    script_directory = Path(__file__).resolve().parent
    if args.project_root == SERVER_PROJECT and (script_directory / "models" / "resnet.py").is_file():
        return script_directory
    return resolve_path(args.project_root)


def import_model_file(model_file):
    # Loading this file directly avoids unrelated model imports in models/__init__.py.
    spec = importlib.util.spec_from_file_location("_sdkd_analysis_cifar_resnet", str(model_file))
    if spec is None or spec.loader is None:
        raise ImportError("Cannot import model definition: {}".format(model_file))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class CIFAR100Test(Dataset):
    """Read the original CIFAR Python test file and retain its stable example IDs."""

    def __init__(self, data_root, mean=DEFAULT_MEAN, std=DEFAULT_STD):
        test_file = data_root / "test"
        if not test_file.is_file():
            test_file = data_root / "cifar-100-python" / "test"
        if not test_file.is_file():
            raise FileNotFoundError("Cannot find CIFAR-100 test file below {}".format(data_root))
        with test_file.open("rb") as handle:
            record = pickle.load(handle, encoding="latin1")
        raw_data = record.get("data", record.get(b"data"))
        raw_labels = record.get("fine_labels", record.get(b"fine_labels"))
        if raw_data is None or raw_labels is None:
            raise ValueError("Expected original CIFAR-100 fields data and fine_labels")
        data = np.asarray(raw_data)
        labels = np.asarray(raw_labels, dtype=np.int64)
        if data.shape != (10000, 3072) or labels.shape != (10000,):
            raise ValueError("Expected the FULL CIFAR-100 test split (10000, 3072); got {} / {}".format(
                data.shape, labels.shape))
        if data.dtype != np.uint8:
            raise ValueError("CIFAR input must contain uint8 pixels")
        if not np.array_equal(np.bincount(labels, minlength=100), np.full(100, 100)):
            raise ValueError("Expected 100 test examples per fine-grained class")
        self.test_file = test_file
        self.images = torch.from_numpy(data.reshape(-1, 3, 32, 32).copy())
        self.labels = torch.from_numpy(labels.copy())
        self.mean = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = self.images[index].float().div(255.0)
        image = (image - self.mean) / self.std
        return image, self.labels[index], index


def tensor_dictionaries(obj, role, location="checkpoint", depth=0):
    """Traverse common checkpoint wrappers, without considering optimizer tensors."""
    if depth > 6:
        return
    if isinstance(obj, torch.nn.Module):
        yield location + ".state_dict()", obj.state_dict()
        return
    if not isinstance(obj, Mapping):
        return
    tensors = {key: value for key, value in obj.items()
               if isinstance(key, str) and torch.is_tensor(value)}
    if tensors:
        yield location, tensors
    preferred = (["teacher", "model_t", "teacher_state_dict"] if role == "teacher" else
                 ["student", "model_s", "student_state_dict"])
    preferred += ["model", "state_dict", "model_state_dict", "net", "network", "module"]
    keys = list(dict.fromkeys(preferred + [key for key in obj if isinstance(key, str)]))
    for key in keys:
        if key in {"optimizer", "optimizer_state_dict", "scheduler", "scaler"}:
            continue
        if key in obj and isinstance(obj[key], (Mapping, torch.nn.Module)):
            yield from tensor_dictionaries(obj[key], role, location + "." + key, depth + 1)


def load_checkpoint_strict(model, checkpoint_path, role):
    # These are the user's own checkpoints; weights_only=False also supports
    # original training metadata. Older PyTorch does not expose that argument.
    try:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    expected = model.state_dict()
    expected_keys = set(expected)
    anchor = next(iter(expected))
    closest = None
    best_score = -1
    for location, state in tensor_dictionaries(checkpoint, role):
        prefixes = {""}
        for key in state:
            if key.endswith("." + anchor):
                prefixes.add(key[:-len(anchor)])
        role_tokens = ("teacher", "model_t") if role == "teacher" else ("student", "model_s")
        prefixes = sorted(prefixes, key=lambda prefix: (
            not any(token in prefix for token in role_tokens), len(prefix), prefix))
        for prefix in prefixes:
            candidate = {key[len(prefix):]: value for key, value in state.items()
                         if key.startswith(prefix)}
            missing = expected_keys - set(candidate)
            # BatchNorm counters were absent from some older checkpoints.
            missing_required = {key for key in missing if not key.endswith("num_batches_tracked")}
            unexpected = set(candidate) - expected_keys
            bad_shape = [key for key in expected_keys & set(candidate)
                         if tuple(candidate[key].shape) != tuple(expected[key].shape)]
            score = len(expected_keys & set(candidate)) - len(missing_required) - len(unexpected) - len(bad_shape)
            if score > best_score:
                best_score = score
                closest = (location, prefix, sorted(missing_required), sorted(unexpected), sorted(bad_shape))
            if not missing_required and not unexpected and not bad_shape:
                # Fill only legacy non-parameter BN counters, then use strict=True.
                candidate.update({key: expected[key] for key in missing})
                model.load_state_dict(candidate, strict=True)
                return {"checkpoint": str(checkpoint_path), "container": location,
                        "removed_prefix": prefix, "tensor_count": len(candidate),
                        "legacy_bn_counters": sorted(missing)}
    raise RuntimeError(
        "Checkpoint is incompatible with the requested architecture: {}\n"
        "Closest (container, prefix, missing, unexpected, shape mismatch): {}\n"
        "Use the same model definition as training; partial loading is deliberately disabled.".format(
            checkpoint_path, closest))


@torch.no_grad()
def collect_logits(model, loader, device, name):
    model.to(device).eval()
    logits_parts, label_parts, index_parts = [], [], []
    for step, (images, labels, indices) in enumerate(loader, 1):
        result = model(images.to(device, non_blocking=True))
        if isinstance(result, (tuple, list)):
            result = result[-1]  # Supports CTKD's optional (features, logits) output.
        if not torch.is_tensor(result) or result.shape != (len(images), 100):
            raise ValueError("{} output must be [batch, 100] raw logits".format(name))
        if not torch.isfinite(result).all().item():
            raise ValueError("{} produced non-finite logits".format(name))
        logits_parts.append(result.detach().float().cpu())
        label_parts.append(labels.cpu())
        index_parts.append(indices.cpu())
        if step == 1 or step % 20 == 0 or step == len(loader):
            print("[INFER] {} batch={}/{}".format(name, step, len(loader)), flush=True)
    logits, labels, indices = map(torch.cat, (logits_parts, label_parts, index_parts))
    if not torch.equal(indices, torch.arange(len(loader.dataset))):
        raise RuntimeError("Test sample ordering changed; shuffle=False is required")
    accuracy = 100.0 * logits.argmax(1).eq(labels).float().mean().item()
    print("[ACCURACY] {}={:.2f}% samples={}".format(name, accuracy, len(labels)), flush=True)
    return logits, labels, indices


def make_sample_records(logits, labels, indices, bins):
    teacher, baseline, sdkd = (logits[name] for name in ("teacher", "baseline", "sdkd"))
    mean = teacher.mean(dim=1)
    std = ((teacher - mean[:, None]).pow(2).mean(dim=1)).sqrt()  # Population std; ddof=0.
    if (std <= 1e-12).any().item():
        raise ValueError("Teacher produced a constant or nearly constant logit vector; score is undefined")
    target = teacher.gather(1, labels[:, None]).squeeze(1)
    score = ((target - mean) / std).numpy().astype(np.float64)
    # Stable sorting breaks exact score ties by the original CIFAR example index.
    # Ties may span neighboring bins; the sample-level Spearman test retains ties.
    order = np.argsort(score, kind="mergesort")
    groups = np.array_split(order, bins)
    bin_id = np.empty(len(score), dtype=np.int64)
    for group_id, group in enumerate(groups, 1):
        bin_id[group] = group_id
    columns = {
        "sample_index": indices.numpy(), "label": labels.numpy(),
        "teacher_score": score, "teacher_target_logit": target.numpy(),
        "teacher_target_probability": teacher.softmax(1).gather(1, labels[:, None]).squeeze(1).numpy(),
        "teacher_logit_mean": mean.numpy(), "teacher_logit_std_population": std.numpy(),
        "teacher_prediction": teacher.argmax(1).numpy(),
        "teacher_error": teacher.argmax(1).ne(labels).numpy().astype(np.int64),
        "baseline_prediction": baseline.argmax(1).numpy(),
        "baseline_error": baseline.argmax(1).ne(labels).numpy().astype(np.int64),
        "baseline_ce": F.cross_entropy(baseline, labels, reduction="none").numpy(),
        "sdkd_prediction": sdkd.argmax(1).numpy(),
        "sdkd_error": sdkd.argmax(1).ne(labels).numpy().astype(np.int64),
        "sdkd_ce": F.cross_entropy(sdkd, labels, reduction="none").numpy(),
        "teacher_score_bin": bin_id,
    }
    if not all(np.isfinite(value).all() for value in columns.values()):
        raise ValueError("Non-finite per-sample statistics")
    return columns, groups


def wilson_percent(errors, total):
    """95% Wilson interval for a bin's marginal classification error rate."""
    z = 1.959963984540054
    proportion = errors / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * np.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)) / denominator
    return 100.0 * (center - half), 100.0 * (center + half)


def summarize_bins(columns, groups):
    rows = []
    for group_id, group in enumerate(groups, 1):
        scores = columns["teacher_score"][group]
        row = {"bin": group_id, "samples": len(group), "teacher_score_min": float(scores.min()),
               "teacher_score_max": float(scores.max()), "teacher_score_mean": float(scores.mean())}
        for name in ("baseline", "sdkd"):
            errors = int(columns[name + "_error"][group].sum())
            low, high = wilson_percent(errors, len(group))
            row.update({name + "_errors": errors, name + "_error_percent": 100.0 * errors / len(group),
                        name + "_error_ci95_low": low, name + "_error_ci95_high": high,
                        name + "_ce_mean": float(columns[name + "_ce"][group].mean())})
        rows.append(row)
        print("[BIN] {}/{} n={} score=[{:.4f}, {:.4f}] baseline_error={:.2f}% sdkd_error={:.2f}%".format(
            group_id, len(groups), len(group), row["teacher_score_min"], row["teacher_score_max"],
            row["baseline_error_percent"], row["sdkd_error_percent"]), flush=True)
    return rows


def spearman_summary(score, loss, repeats, seed):
    if np.ptp(score) == 0 or np.ptp(loss) == 0:
        raise ValueError("Spearman correlation is undefined for a constant score or loss")
    rho, pvalue = spearmanr(score, loss)
    interval = None
    if repeats:
        rng = np.random.default_rng(seed)
        bootstrap_rho = []
        for _ in range(repeats):
            indices = rng.integers(0, len(score), size=len(score))
            if np.ptp(score[indices]) == 0 or np.ptp(loss[indices]) == 0:
                continue
            value = float(spearmanr(score[indices], loss[indices])[0])
            if np.isfinite(value):
                bootstrap_rho.append(value)
        if len(bootstrap_rho) < max(1, int(0.95 * repeats)):
            raise RuntimeError("Too many degenerate bootstrap replicates")
        interval = [float(value) for value in np.percentile(bootstrap_rho, [2.5, 97.5])]
    return {"rho": float(rho), "p_value_asymptotic_two_sided": float(pvalue),
            "ci95_paired_example_bootstrap": interval,
            "bootstrap_requested": repeats,
            "bootstrap_valid": len(bootstrap_rho) if repeats else 0,
            "samples": len(score)}


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rgb(red, green, blue):
    return red / 255, green / 255, blue / 255


def configure_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    sns.set_theme(style="darkgrid")
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def validate_plot_rows(rows):
    if len(rows) < 2:
        raise ValueError("At least two teacher score bins are required")
    bins = [row["bin"] for row in rows]
    if bins != list(range(1, len(rows) + 1)):
        raise ValueError("Bin numbers must be consecutive and ordered from 1")
    for row in rows:
        if row["samples"] < 1:
            raise ValueError("Each bin must contain examples")
        for name in ("baseline", "sdkd"):
            mean = row[name + "_error_percent"]
            low = row[name + "_error_ci95_low"]
            high = row[name + "_error_ci95_high"]
            if not np.isfinite([mean, low, high]).all():
                raise ValueError("Non-finite bin error statistics")
            if not (-1e-8 <= low <= mean + 1e-8 and mean <= high + 1e-8 <= 100.0 + 2e-8):
                raise ValueError("Invalid classification error or confidence interval")


def save_figure(rows, output_dir):
    """Draw the former left panel using the supplied paper figure style.

    Error bars retain the existing 95% Wilson confidence intervals. They are
    not the standard deviations across training runs used in the reference plot.
    """
    validate_plot_rows(rows)
    configure_style()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=FIG_SIZE, dpi=FIG_DPI, facecolor="white")
    x_positions = np.arange(len(rows))
    bin_labels = [row["bin"] for row in rows]
    plot_configs = [
        {"key": "baseline", "label": "Baseline", "color": rgb(82, 137, 185),
         "linestyle": "--", "marker": "o", "linewidth": LINE_WIDTH, "zorder": 3},
        {"key": "sdkd", "label": "SDKD", "color": rgb(197, 57, 50),
         "linestyle": "-", "marker": "s", "linewidth": LINE_WIDTH, "zorder": 5},
    ]
    upper_values = []
    for config in plot_configs:
        name = config["key"]
        means = np.array([row[name + "_error_percent"] for row in rows])
        lows = np.array([row[name + "_error_ci95_low"] for row in rows])
        highs = np.array([row[name + "_error_ci95_high"] for row in rows])
        upper_values.extend(highs)
        ax.plot(
            x_positions, means, linestyle=config["linestyle"], marker=config["marker"],
            color=config["color"], linewidth=config["linewidth"], markersize=MARKER_SIZE,
            markerfacecolor="none", markeredgecolor=config["color"],
            markeredgewidth=MARKER_EDGE_WIDTH, label=config["label"], zorder=config["zorder"],
        )
        for index, x_position in enumerate(x_positions):
            yerr = np.maximum([[means[index] - lows[index]], [highs[index] - means[index]]], 0.0)
            ax.errorbar(
                x_position, means[index], yerr=yerr, fmt="none", ecolor=config["color"],
                elinewidth=ERROR_LINE_WIDTH, capsize=CAP_SIZE, capthick=ERROR_LINE_WIDTH,
                zorder=config["zorder"] + 1,
            )

    y_max = max(upper_values)
    y_margin = max(y_max * 0.15, 0.25)
    y_upper = min(100.0, y_max + y_margin)
    # Padding below zero keeps low-error points, caps, and connecting lines
    # inside the axes. Tick labels remain nonnegative; observations are unchanged.
    lower_padding = max(y_upper * 0.035, 0.25)
    ax.set_ylim(-lower_padding, y_upper)
    nonnegative_ticks = [tick for tick in ax.get_yticks() if 0 <= tick <= y_upper]
    ax.set_yticks(nonnegative_ticks)
    ax.set_xlabel(X_LABEL, fontsize=20, fontweight="bold")
    ax.set_ylabel(Y_LABEL, fontsize=20, fontweight="bold")
    tick_indices = (x_positions if len(rows) <= 20 else
                    x_positions[::max(1, len(rows) // 10)])
    ax.set_xticks(tick_indices)
    ax.set_xticklabels([bin_labels[index] for index in tick_indices])
    ax.tick_params(axis="both", labelsize=16)
    ax.yaxis.grid(True, linestyle="-", color="white", linewidth=1.2)
    ax.xaxis.grid(True, linestyle="-", color="white", linewidth=1.2)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    legend_order = ["Baseline", "SDKD"]
    ordered_handles = [handles[labels.index(label)] for label in legend_order]
    ax.legend(ordered_handles, legend_order, loc="best", fontsize=14,
              frameon=True, edgecolor="black")
    fig.tight_layout()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_FILENAME
    fig.savefig(str(output_path), format="pdf", dpi=FIG_DPI,
                bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print("[FIGURE] {} format=pdf dpi={}".format(output_path, FIG_DPI), flush=True)
    return output_path


def redraw_existing_results(args):
    if args.plot_from == "latest":
        project_root = resolve_project_root(args)
        candidates = list((project_root / "output").glob("difficulty_analysis_*/bins.csv"))
        if not candidates:
            raise FileNotFoundError("No completed bins.csv found below {}".format(project_root / "output"))
        csv_path = max(candidates, key=lambda path: (path.stat().st_mtime_ns, str(path)))
    else:
        csv_path = resolve_path(args.plot_from)
        if csv_path.is_dir():
            csv_path = csv_path / "bins.csv"
    if not csv_path.is_file():
        raise FileNotFoundError("Bin statistics not found: {}".format(csv_path))
    fields = ["bin", "samples"] + [name + suffix for name in ("baseline", "sdkd")
              for suffix in ("_error_percent", "_error_ci95_low", "_error_ci95_high")]
    with csv_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(fields) - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Missing fields in bins.csv: {}".format(sorted(missing)))
        rows = [{key: int(record[key]) if key in ("bin", "samples") else float(record[key])
                 for key in fields} for record in reader]
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else csv_path.parent)
    print("[REDRAW] {}".format(csv_path), flush=True)
    save_figure(rows, output_dir)


def main():
    args = parse_args()
    if args.plot_from is not None:
        redraw_existing_results(args)
        return
    start = time.perf_counter()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the training environment or use --device cpu")

    project_root = resolve_project_root(args)
    sys.path.insert(0, str(project_root))
    model_file = (resolve_path(args.model_file, (project_root,)) if args.model_file else
                  project_root / "models" / "resnet.py")
    if not model_file.is_file():
        raise FileNotFoundError("Model definition not found: {}; specify --model-file if needed".format(model_file))
    checkpoint_paths = {
        name: resolve_path(getattr(args, name + "_ckpt"), (project_root,))
        for name in ("teacher", "baseline", "sdkd")
    }
    data_root = resolve_path(args.data_root)
    output_dir = (Path(args.output_dir).expanduser().resolve() if args.output_dir else
                  project_root / "output" / ("difficulty_analysis_" + datetime.now().strftime("%Y%m%d_%H%M%S")))
    if output_dir.exists() and any((output_dir / filename).exists() for filename in
                                   ("summary.json", "samples.csv", "bins.csv", "logits.npz")):
        raise FileExistsError("Analysis outputs already exist in {}; choose a new --output-dir".format(output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    print("[PROJECT] {}".format(project_root), flush=True)
    print("[MODEL_FILE] {}".format(model_file), flush=True)
    for name, path in checkpoint_paths.items():
        print("[CHECKPOINT] {}={}".format(name, path), flush=True)
    print("[PREPROCESS] mean={} std={} input=3x32x32".format(args.mean, args.std), flush=True)
    print("[DEVICE] {} precision=fp32".format(device), flush=True)
    dataset = CIFAR100Test(data_root, args.mean, args.std)
    print("[DATA] file={} split=test samples={} classes=100".format(dataset.test_file, len(dataset)), flush=True)
    # All test pixels are already in memory; single-process loading avoids IPC
    # overhead and shared-memory requirements for this small evaluation split.
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False,
                        num_workers=0, pin_memory=(device.type == "cuda"))
    model_module = import_model_file(model_file)
    outputs, checkpoint_info = {}, {}
    labels, indices = None, None
    for name in ("teacher", "baseline", "sdkd"):
        architecture = args.teacher_arch if name == "teacher" else args.student_arch
        if not hasattr(model_module, architecture):
            raise AttributeError("{} is not defined in {}".format(architecture, model_file))
        model = getattr(model_module, architecture)(num_classes=100)
        checkpoint_info[name] = load_checkpoint_strict(model, checkpoint_paths[name], name)
        print("[LOAD] {} arch={} container={} prefix={!r} strict=True".format(
            name, architecture, checkpoint_info[name]["container"], checkpoint_info[name]["removed_prefix"]), flush=True)
        outputs[name], this_labels, this_indices = collect_logits(model, loader, device, name)
        if labels is not None and (not torch.equal(labels, this_labels) or not torch.equal(indices, this_indices)):
            raise RuntimeError("Labels or example IDs differ between models")
        labels, indices = this_labels, this_indices
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    columns, groups = make_sample_records(outputs, labels, indices, args.bins)
    rows = summarize_bins(columns, groups)
    correlations = {}
    for name in ("baseline", "sdkd"):
        print("[STATISTICS] score_vs_{}_ce bootstrap={}".format(name, args.bootstrap), flush=True)
        stat = spearman_summary(columns["teacher_score"], columns[name + "_ce"], args.bootstrap, args.seed)
        correlations["teacher_score_vs_" + name + "_ce"] = stat
        print("[SPEARMAN] {} rho={:.4f} ci95={} p={:.6g}".format(
            name, stat["rho"], stat["ci95_paired_example_bootstrap"], stat["p_value_asymptotic_two_sided"]), flush=True)

    summary = {
        "dataset": "CIFAR-100 test", "samples": len(dataset), "classes": 100,
        "project_root": str(project_root), "model_file": str(model_file),
        "data_file": str(dataset.test_file), "checkpoints": checkpoint_info,
        "teacher_arch": args.teacher_arch, "student_arch": args.student_arch,
        "preprocessing": {"mean": list(args.mean), "std": list(args.std),
                          "random_augmentation": False},
        "accuracy_percent": {name: float(100.0 * (1.0 - columns[name + "_error"].mean()))
                             for name in ("teacher", "baseline", "sdkd")},
        "score": {"formula": "(z_t[y] - mean(z_t)) / sqrt(mean((z_t - mean(z_t))**2))",
                  "axis": "within each example, all 100 classes", "std_ddof": 0,
                  "temperature_applied": False, "near_zero_std_rejected_at": 1e-12},
        "binning": {"bins": args.bins, "method": "stable ascending teacher score; equal example counts",
                    "counts": [len(group) for group in groups],
                    "exact_score_ties": "original example index breaks bin ties; Spearman uses average ranks"},
        "correlations": correlations, "seed": args.seed, "device": str(device),
        "versions": {"python": sys.version.split()[0], "pytorch": torch.__version__, "numpy": np.__version__},
        "interpretation": "Baseline provides the primary external validation; SDKD results are supplementary. "
                          "Negative sample-level rho supports the score as a relative-difficulty proxy. "
                          "Bin error bars are marginal 95% Wilson intervals and do not test method differences.",
        "bootstrap_method": "Resample paired examples with replacement and recompute Spearman; "
                            "percentile 95% confidence interval; fixed trained checkpoints.",
        "figure": {"filename": OUTPUT_FILENAME, "format": "pdf", "dpi": FIG_DPI,
                   "figsize_inches": list(FIG_SIZE), "x_label": X_LABEL, "y_label": Y_LABEL,
                   "error_bars": "95% Wilson confidence intervals"},
    }
    write_csv(output_dir / "samples.csv", ({key: value[i] for key, value in columns.items()}
                                         for i in range(len(dataset))), list(columns))
    write_csv(output_dir / "bins.csv", rows, list(rows[0]))
    np.savez_compressed(str(output_dir / "logits.npz"),
                        teacher_logits=outputs["teacher"].numpy(),
                        baseline_logits=outputs["baseline"].numpy(),
                        sdkd_logits=outputs["sdkd"].numpy(),
                        labels=labels.numpy(), sample_indices=indices.numpy())
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, allow_nan=False)
    save_figure(rows, output_dir)
    print("[DONE] elapsed_seconds={:.1f}".format(time.perf_counter() - start), flush=True)
    print("[OUTPUT] {}".format(output_dir), flush=True)
    print("[FILES] difficulty_analysis.pdf samples.csv bins.csv summary.json logits.npz", flush=True)


if __name__ == "__main__":
    main()
