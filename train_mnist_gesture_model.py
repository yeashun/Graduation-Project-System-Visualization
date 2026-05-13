"""
Train a small CNN on MNIST as a first 0-9 recognition baseline.

This is a proxy model for the GUI demo pipeline. MNIST contains handwritten
digits, not real hand gestures, so it is useful for testing training/export
and inference wiring before replacing the dataset with gesture images.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple MNIST 0-9 CNN.")
    parser.add_argument("--data-dir", default="data", help="Dataset download/cache directory.")
    parser.add_argument("--output-dir", default="models", help="Directory for trained model files.")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=128, help="Training batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
    parser.add_argument("--no-download", action="store_true", help="Do not download MNIST.")
    parser.add_argument("--export-onnx", action="store_true", help="Also export an ONNX model.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    return parser.parse_args()


def require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install torch and torchvision first, for example:\n"
            "  pip install torch torchvision\n"
        ) from exc
    return torch, nn, F, DataLoader, datasets, transforms


torch, nn, F, DataLoader, datasets, transforms = require_torch()


class SmallDigitCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)


def build_loaders(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    data_dir = Path(args.data_dir)
    train_set = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=not args.no_download,
        transform=transform,
    )
    test_set = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=not args.no_download,
        transform=transform,
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, test_loader


def train_one_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def evaluate(model, loader, device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = F.cross_entropy(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def export_model(model, output_dir: Path, metrics: dict, export_onnx: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pt_path = output_dir / "mnist_gesture_cnn.pt"
    meta_path = output_dir / "mnist_gesture_cnn.meta.json"

    torch.save(
        {
            "model_name": "SmallDigitCNN",
            "state_dict": model.state_dict(),
            "input_shape": [1, 1, 28, 28],
            "classes": list(range(10)),
            "normalization": {"mean": [0.1307], "std": [0.3081]},
            "note": "Trained on MNIST handwritten digits, not real gesture images.",
            "metrics": metrics,
        },
        pt_path,
    )
    meta_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    if export_onnx:
        onnx_path = output_dir / "mnist_gesture_cnn.onnx"
        dummy = torch.randn(1, 1, 28, 28)
        torch.onnx.export(
            model.cpu(),
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,
        )
        print(f"Saved ONNX model: {onnx_path}")

    print(f"Saved PyTorch model: {pt_path}")
    print(f"Saved metrics: {meta_path}")


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, test_loader = build_loaders(args)
    model = SmallDigitCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
            }
        )
        print(
            f"Epoch {epoch:02d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "dataset": "MNIST",
        "epochs": args.epochs,
        "best_test_acc": best_acc,
        "history": history,
    }
    export_model(model, Path(args.output_dir), metrics, args.export_onnx)


if __name__ == "__main__":
    main()
