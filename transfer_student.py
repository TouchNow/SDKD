from __future__ import print_function

import argparse
import json
import os
import random
import socket
from bisect import bisect_right

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import model_dict


SUPPORTED_ARCHITECTURE = "resnet8x4"
TRANSFER_SEED = 0
TRANSFER_EPOCHS = 100
TRANSFER_BATCH_SIZE = 64
TRANSFER_MILESTONES = (30, 60, 90)
CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


def normalize_student_state(checkpoint):
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError('student checkpoint must contain a "model" state dictionary')
    state = checkpoint["model"]
    if not isinstance(state, dict):
        raise ValueError('checkpoint["model"] must be a state dictionary')

    normalized = {}
    for name, value in state.items():
        normalized[name[7:] if name.startswith("module.") else name] = value
    return normalized


def build_source_eval_model(architecture, checkpoint):
    if architecture != SUPPORTED_ARCHITECTURE:
        raise ValueError(
            "unsupported student architecture: {}; expected {}".format(
                architecture, SUPPORTED_ARCHITECTURE
            )
        )
    state = normalize_student_state(checkpoint)
    model = model_dict[architecture](num_classes=100)
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    if missing:
        raise ValueError("missing source keys: {}".format(", ".join(missing)))
    if unexpected:
        raise ValueError("unexpected source keys: {}".format(", ".join(unexpected)))
    for name in expected:
        if expected[name].shape != state[name].shape:
            raise ValueError(
                "source shape mismatch for {}: checkpoint {} != model {}".format(
                    name, tuple(state[name].shape), tuple(expected[name].shape)
                )
            )
    model.load_state_dict(state, strict=True)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def build_cifar100_test_loader(data_root, num_workers):
    expected_directory = os.path.join(data_root, "cifar-100-python")
    if not os.path.isdir(expected_directory):
        raise FileNotFoundError(
            "offline CIFAR-100 data not found: {}".format(expected_directory)
        )
    test_set = datasets.CIFAR100(
        root=data_root,
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR100_MEAN, CIFAR100_STD),
        ]),
    )
    return DataLoader(
        test_set,
        batch_size=TRANSFER_BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def validate_source_checkpoint(model, loader, device):
    model = model.to(device)
    model.eval()
    top1_correct = 0
    top5_correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.float().to(device)
            targets = targets.to(device)
            predictions = model(inputs).topk(5, dim=1).indices
            matches = predictions.eq(targets.view(-1, 1))
            top1_correct += matches[:, :1].sum().item()
            top5_correct += matches.sum().item()
            total += targets.size(0)
    if total == 0:
        raise ValueError("CIFAR-100 test loader is empty")
    return 100.0 * top1_correct / total, 100.0 * top5_correct / total


def build_linear_eval_model(architecture, checkpoint, num_classes):
    if architecture != SUPPORTED_ARCHITECTURE:
        raise ValueError(
            "unsupported student architecture: {}; expected {}".format(
                architecture, SUPPORTED_ARCHITECTURE
            )
        )

    state = normalize_student_state(checkpoint)
    model = model_dict[architecture](num_classes=num_classes)
    target_state = model.state_dict()
    backbone_keys = {name for name in target_state if not name.startswith("fc.")}
    checkpoint_backbone_keys = {
        name for name in state if not name.startswith("fc.")
    }
    missing = sorted(backbone_keys - checkpoint_backbone_keys)
    unexpected = sorted(checkpoint_backbone_keys - backbone_keys)
    if missing:
        raise ValueError("missing backbone keys: {}".format(", ".join(missing)))
    if unexpected:
        raise ValueError("unexpected backbone keys: {}".format(", ".join(unexpected)))

    for name in sorted(backbone_keys):
        if target_state[name].shape != state[name].shape:
            raise ValueError(
                "backbone shape mismatch for {}: checkpoint {} != model {}".format(
                    name, tuple(state[name].shape), tuple(target_state[name].shape)
                )
            )
        target_state[name] = state[name]
    model.load_state_dict(target_state)

    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("fc.")
    model.eval()
    model.fc.train()
    return model


def _set_transfer_seed(seed=TRANSFER_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_linear_train_mode(model):
    model.eval()
    model.fc.train()


def _transfer_learning_rate(epoch):
    decay_steps = bisect_right(TRANSFER_MILESTONES, epoch - 1)
    return 0.1 * (0.1 ** decay_steps)


def train_linear_epoch(model, loader, criterion, optimizer, device):
    _set_linear_train_mode(model)
    for inputs, targets in loader:
        inputs = inputs.float().to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()


def validate_linear(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.float().to(device)
            targets = targets.to(device)
            predictions = model(inputs).argmax(dim=1)
            correct += predictions.eq(targets).sum().item()
            total += targets.size(0)
    if total == 0:
        raise ValueError("transfer validation loader is empty")
    return 100.0 * correct / total


def _write_transfer_results(path, results):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as result_file:
        json.dump(results, result_file, indent=2)


def run_source_verification(
    checkpoint, architecture, data_root, num_workers, device,
    result_path, all_results
):
    model = build_source_eval_model(architecture, checkpoint)
    loader = build_cifar100_test_loader(data_root, num_workers)
    verified_acc1, verified_acc5 = validate_source_checkpoint(
        model, loader, device
    )
    recorded_acc1 = checkpoint.get("best_acc")
    difference = (
        verified_acc1 - float(recorded_acc1)
        if recorded_acc1 is not None else None
    )
    result = {
        "recorded_acc1": (
            float(recorded_acc1) if recorded_acc1 is not None else None
        ),
        "verified_acc1": verified_acc1,
        "verified_acc5": verified_acc5,
        "acc1_difference": difference,
    }
    all_results["CIFAR-100"] = result
    if recorded_acc1 is not None:
        print(
            "CIFAR-100 checkpoint: Recorded Acc@1 {:.2f}".format(
                float(recorded_acc1)
            )
        )
    print(
        "CIFAR-100 verified: Acc@1 {:.2f} Acc@5 {:.2f}".format(
            verified_acc1, verified_acc5
        )
    )
    if difference is not None and abs(difference) > 1e-6:
        print(
            "WARNING: verified CIFAR-100 Acc@1 differs from checkpoint "
            "by {:+.2f}".format(difference)
        )
    _write_transfer_results(result_path, all_results)
    return result


def run_linear_evaluation(
    dataset_name, model, train_loader, val_loader, device, result_path,
    all_results, epochs=TRANSFER_EPOCHS
):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(
        model.fc.parameters(), lr=0.1, momentum=0.9,
        weight_decay=0.0, nesterov=True
    )
    result = {"best_acc": -1.0, "best_epoch": 0, "history": []}
    all_results[dataset_name] = result

    for epoch in range(1, epochs + 1):
        learning_rate = _transfer_learning_rate(epoch)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        train_linear_epoch(model, train_loader, criterion, optimizer, device)
        current_acc = validate_linear(model, val_loader, device)
        if current_acc > result["best_acc"]:
            result["best_acc"] = current_acc
            result["best_epoch"] = epoch
        result["history"].append({
            "epoch": epoch,
            "acc": current_acc,
            "best_acc": result["best_acc"],
            "learning_rate": learning_rate,
        })
        print(
            "{} Epoch {}/{} Acc@1 {:.2f} Best Acc@1 {:.2f}".format(
                dataset_name, epoch, epochs, current_acc, result["best_acc"]
            )
        )
        _write_transfer_results(result_path, all_results)
    return result


def get_transfer_data_root(hostname=None):
    hostname = socket.gethostname() if hostname is None else hostname
    if hostname.startswith("TouchNow"):
        return "/home/touchnow/Documents/datasets"
    return "/mnt/home/chendingyao/datasets"


def build_transfer_loaders(dataset_name, data_root, num_workers):
    pin_memory = torch.cuda.is_available()
    if dataset_name == "STL-10":
        root = os.path.join(data_root, "STL-10")
        normalize = transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        )
        train_set = datasets.STL10(
            root=root,
            split="train",
            download=True,
            transform=transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]),
        )
        val_set = datasets.STL10(
            root=root,
            split="test",
            download=True,
            transform=transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                normalize,
            ]),
        )
    else:
        raise ValueError("unsupported transfer dataset: {}".format(dataset_name))

    return (
        DataLoader(
            train_set,
            batch_size=TRANSFER_BATCH_SIZE,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
        DataLoader(
            val_set,
            batch_size=TRANSFER_BATCH_SIZE,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False,
        ),
    )


def run_transfer_evaluations(
    checkpoint_path, architecture, data_root, output_dir,
    num_workers=8, device=None
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            "student checkpoint not found: {}".format(checkpoint_path)
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    device = device or torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    result_path = os.path.join(output_dir, "transfer_results.json")
    all_results = {}

    run_source_verification(
        checkpoint, architecture, data_root, num_workers, device,
        result_path, all_results
    )

    for dataset_name, num_classes in (
        ("STL-10", 10),
    ):
        _set_transfer_seed()
        train_loader, val_loader = build_transfer_loaders(
            dataset_name, data_root, num_workers
        )
        model = build_linear_eval_model(
            architecture, checkpoint, num_classes
        )
        run_linear_evaluation(
            dataset_name,
            model,
            train_loader,
            val_loader,
            device,
            result_path,
            all_results,
        )
    return all_results


def parse_option():
    parser = argparse.ArgumentParser(
        "linear evaluation for a student checkpoint"
    )
    parser.add_argument("--student_checkpoint", required=True)
    parser.add_argument(
        "--student_arch",
        default=SUPPORTED_ARCHITECTURE,
        choices=[SUPPORTED_ARCHITECTURE],
    )
    parser.add_argument("--data_root", default=get_transfer_data_root())
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    return parser.parse_args()


def main():
    args = parse_option()
    run_transfer_evaluations(
        args.student_checkpoint,
        args.student_arch,
        args.data_root,
        args.output_dir,
        args.num_workers,
    )


if __name__ == "__main__":
    main()
