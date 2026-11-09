from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset.cifar100 import get_cifar100_dataloaders
from models import model_dict


NUM_CLASSES = 100
MAX_DIFF = 4.0
TEACHER_MODEL = "resnet32x4"
STUDENT_MODEL = "resnet8x4"
TEACHER_CHECKPOINT = "/root/autodl-tmp/CTKD-main/save/resnet32x4-(79.42).pth"
OUTPUT_PATH = Path(__file__).with_name("correlation.png")

METHOD_CHECKPOINTS = (
    ("Vanilla", "/root/autodl-tmp/CTKD-main/save/vanilla/resnet8x4_last.pth"),
    ("KD", "/root/autodl-tmp/CTKD-main/save/kd/resnet8x4_last.pth"),
    ("DKD", "/root/autodl-tmp/CTKD-main/save/dkd/resnet8x4_last.pth"),
    ("TeKAP", "/root/autodl-tmp/CTKD-main/save/tekap/resnet8x4_last.pth"),
    ("FCKA", "/root/autodl-tmp/CTKD-main/save/fcka/resnet8x4_last.pth"),
    ("SDKD", "/root/autodl-tmp/CTKD-main/save/sdkd/resnet8x4-(76.63).pth"),
)


def load_model(model_name, checkpoint_path, device):
    model = model_dict[model_name](num_classes=NUM_CLASSES)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)
    model.eval()
    return model


def get_output_metric(model, val_loader, device, num_classes=NUM_CLASSES):
    model.eval()
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    counts = np.zeros((num_classes, 1), dtype=np.float64)

    with torch.no_grad():
        for data, labels in tqdm(val_loader):
            outputs = model(data.to(device))
            predictions = outputs.detach().cpu().numpy()
            batch_labels = labels.cpu().numpy()

            for prediction, label in zip(predictions, batch_labels):
                counts[label, 0] += 1
                matrix[label] += prediction

    matrix /= counts
    return matrix


def compute_difference(student_matrix, teacher_matrix, max_diff=MAX_DIFF):
    difference = np.abs(student_matrix - teacher_matrix) / max_diff
    np.fill_diagonal(difference, 0.0)
    return difference


def plot_correlation(difference, axis, title):
    seaborn.heatmap(
        difference,
        vmin=0,
        vmax=1.0,
        cmap="PuBuGn",
        ax=axis,
        xticklabels=False,
        yticklabels=False,
    )
    axis.set_title(title)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader = get_cifar100_dataloaders(batch_size=64, num_workers=8)

    teacher_model = load_model(TEACHER_MODEL, TEACHER_CHECKPOINT, device)
    teacher_matrix = get_output_metric(
        teacher_model,
        val_loader,
        device,
        NUM_CLASSES,
    )

    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    for index, (method_name, checkpoint_path) in enumerate(METHOD_CHECKPOINTS):
        student_model = load_model(STUDENT_MODEL, checkpoint_path, device)
        student_matrix = get_output_metric(
            student_model,
            val_loader,
            device,
            NUM_CLASSES,
        )
        difference = compute_difference(student_matrix, teacher_matrix, MAX_DIFF)
        print(
            "[{}] max(diff): {:.6f}, mean(diff): {:.6f}".format(
                method_name,
                difference.max(),
                difference.mean(),
            )
        )
        title = "({}) Student: {}".format(
            chr(ord("a") + index),
            method_name,
        )
        plot_correlation(difference, axes.flat[index], title)

    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
