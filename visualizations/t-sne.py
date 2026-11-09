from pathlib import Path

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
    "font.size": 7,
    "axes.titlesize": 12,
})
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
import torch
from tqdm import tqdm
from dataset.cifar100 import get_cifar100_dataloaders
from models import model_dict


OUTPUT_PATH = Path(__file__).with_name('t-sne.png')
PNG_OUTPUT_DIR = Path(__file__).with_name("t-sne-png")
PNG_DPI = 500


# visualize t-SNE
def get_tsne(stu, mpath, method_name):
    num_classes = 100
    train_loader, val_loader = get_cifar100_dataloaders(batch_size=64, num_workers=8)
    model = model_dict[stu](num_classes=num_classes)
    model.load_state_dict(torch.load(mpath, map_location='cpu')['model'])
    all_features, all_labels = [], []
    correct, total = 0, 0
    model.eval()
    model = model.cuda()
    with torch.no_grad():
        for i, (data, labels) in tqdm(enumerate(val_loader)):
            data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data, is_feat=True)
            all_features.append(features[-1].data.cpu().numpy())
            all_labels.append(labels.data.cpu().numpy())
            correct += outputs.argmax(dim=1).eq(labels).sum().item()
            total += labels.size(0)
    all_features = np.concatenate(all_features, 0)
    all_labels = np.concatenate(all_labels, 0)
    top1 = 100.0 * correct / total
    print("[{}] test Acc@1: {:.2f}%".format(method_name, top1))

    tsne = TSNE()
    all_features = tsne.fit_transform(all_features)
    return all_features, all_labels, top1


def get_class_colors(num_classes):
    return ['C' + str(index) for index in range(num_classes)]


def plot_features(features, labels, num_classes, axis, title, colors):
    for label in range(num_classes):
        axis.scatter(
            features[labels == label, 0],
            features[labels == label, 1],
            color=colors[label], s=1, alpha=0.4, rasterized=True)
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])


def save_method_png(features, labels, num_classes, colors, method_name):
    PNG_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6, 6))
    plot_features(features, labels, num_classes, axis, "", colors)
    figure.tight_layout(pad=0)
    output_path = PNG_OUTPUT_DIR / "{}.png".format(method_name.lower())
    figure.savefig(
        output_path,
        format="png",
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0,
    )
    plt.close(figure)


METHOD_CHECKPOINTS = (
    ("Vanilla", "/root/autodl-tmp/CTKD-main/save/vanilla/resnet8x4_last.pth"),
    ("KD", "/root/autodl-tmp/CTKD-main/save/kd/resnet8x4_last.pth"),
    ("DKD", "/root/autodl-tmp/CTKD-main/save/dkd/resnet8x4_last.pth"),
    ("TeKAP", "/root/autodl-tmp/CTKD-main/save/tekap/resnet8x4_last.pth"),
    ("FCKA", "/root/autodl-tmp/CTKD-main/save/fcka/resnet8x4_last.pth"),
    ("SDKD", "/root/autodl-tmp/CTKD-main/save/sdkd/resnet8x4-(76.63).pth"),
)


def main():
    figure, axes = plt.subplots(2, 3, figsize=(12, 8))
    colors = get_class_colors(100)
    for index, (method_name, checkpoint_path) in enumerate(METHOD_CHECKPOINTS):
        features, labels, _ = get_tsne("resnet8x4", checkpoint_path, method_name)
        panel_title = "({}) Student: {}".format(
            chr(ord("a") + index), method_name
        )
        plot_features(features, labels, 100, axes.flat[index], panel_title, colors)
        save_method_png(features, labels, 100, colors, method_name)
    figure.tight_layout()
    figure.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(figure)


if __name__ == "__main__":
    main()
