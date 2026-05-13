"""
Train a simple CNN for 0-9 hand gesture recognition from image folders.

Recommended dataset:
  ASL-HG processed images, using only digit folders 0-9.

Supported directory layouts:
  1) Pre-split:
     dataset/train/0/*.jpg ... dataset/train/9/*.jpg
     dataset/test/0/*.jpg  ... dataset/test/9/*.jpg

  2) Single folder:
     dataset/0/*.jpg ... dataset/9/*.jpg
     The script will create a deterministic train/validation split in memory.
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIGIT_CLASSES = [str(index) for index in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN on 0-9 gesture image folders.")
    parser.add_argument(
        "--data-dir",
        default=".",
        help="Dataset root directory. Defaults to the current working directory.",
    )
    parser.add_argument("--output-dir", default="models", help="Directory for trained files.")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--image-size", type=int, default=64, help="Input image size.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation ratio for unsplit folders.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader worker count.")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument("--export-onnx", action="store_true", help="Also export ONNX.")
    return parser.parse_args()


def require_dependencies():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install training dependencies first:\n"
            "  pip install torch torchvision pillow\n"
        ) from exc
    return torch, nn, F, Image, DataLoader, Dataset, transforms


torch, nn, F, Image, DataLoader, Dataset, transforms = require_dependencies()


@dataclass(frozen=True)
class ImageSample:
    path: Path
    label: int


class GestureDigitDataset(Dataset):
    def __init__(self, samples: Sequence[ImageSample], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        image = Image.open(sample.path).convert("RGB")
        return self.transform(image), sample.label


class GestureCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def class_dir_candidates(root: Path, class_name: str) -> Iterable[Path]:
    yield root / class_name
    yield root / f"digit_{class_name}"
    yield root / f"Digit_{class_name}"


def collect_samples(root: Path) -> List[ImageSample]:
    samples: List[ImageSample] = []
    missing_classes = []

    for label, class_name in enumerate(DIGIT_CLASSES):
        class_dir = next((candidate for candidate in class_dir_candidates(root, class_name) if candidate.is_dir()), None)
        if class_dir is None:
            missing_classes.append(class_name)
            continue

        class_images = sorted(path for path in class_dir.rglob("*") if path.is_file() and is_image(path))
        samples.extend(ImageSample(path=image_path, label=label) for image_path in class_images)

    if missing_classes:
        print(f"Warning: missing class folders: {', '.join(missing_classes)}")
    if not samples:
        raise SystemExit(f"No digit images found under: {root}")
    return samples


def split_samples(samples: Sequence[ImageSample], val_ratio: float, seed: int) -> Tuple[List[ImageSample], List[ImageSample]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[ImageSample]] = {label: [] for label in range(10)}
    for sample in samples:
        by_class[sample.label].append(sample)

    train_samples: List[ImageSample] = []
    val_samples: List[ImageSample] = []
    for label, class_samples in by_class.items():
        if not class_samples:
            continue
        rng.shuffle(class_samples)
        val_count = max(1, int(len(class_samples) * val_ratio))
        val_samples.extend(class_samples[:val_count])
        train_samples.extend(class_samples[val_count:])

    return train_samples, val_samples


def find_split_dirs(data_dir: Path) -> Tuple[Path | None, Path | None]:
    train_names = ("train", "training", "Train", "Training")
    test_names = ("test", "val", "valid", "validation", "Test", "Val", "Validation")
    train_dir = next((data_dir / name for name in train_names if (data_dir / name).is_dir()), None)
    test_dir = next((data_dir / name for name in test_names if (data_dir / name).is_dir()), None)
    return train_dir, test_dir


def build_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.RandomAffine(degrees=12, translate=(0.08, 0.08), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.25),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    return train_transform, eval_transform


def build_loaders(args: argparse.Namespace):
    data_dir = Path(args.data_dir).resolve()
    train_dir, test_dir = find_split_dirs(data_dir)

    if train_dir and test_dir:
        train_samples = collect_samples(train_dir)
        val_samples = collect_samples(test_dir)
        split_mode = f"pre-split: train={train_dir.name}, val={test_dir.name}"
    else:
        all_samples = collect_samples(data_dir)
        train_samples, val_samples = split_samples(all_samples, args.val_ratio, args.seed)
        split_mode = f"in-memory stratified split: val_ratio={args.val_ratio}"

    train_transform, eval_transform = build_transforms(args.image_size)
    train_set = GestureDigitDataset(train_samples, train_transform)
    val_set = GestureDigitDataset(val_samples, eval_transform)

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, split_mode, len(train_samples), len(val_samples)


def accuracy_from_logits(logits, labels) -> int:
    return (logits.argmax(dim=1) == labels).sum().item()


def train_epoch(model, loader, optimizer, device) -> Tuple[float, float]:
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
        total_correct += accuracy_from_logits(logits, labels)
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
        total_correct += accuracy_from_logits(logits, labels)
        total_count += batch_size

    return total_loss / total_count, total_correct / total_count


def save_outputs(model, args: argparse.Namespace, metrics: dict) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "gesture_digit_cnn.pt"
    metrics_path = output_dir / "gesture_digit_cnn.meta.json"
    torch.save(
        {
            "model_name": "GestureCNN",
            "state_dict": model.state_dict(),
            "input_shape": [1, 1, args.image_size, args.image_size],
            "classes": DIGIT_CLASSES,
            "normalization": {"mean": [0.5], "std": [0.5]},
            "metrics": metrics,
        },
        model_path,
    )
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.export_onnx:
        onnx_path = output_dir / "gesture_digit_cnn.onnx"
        dummy = torch.randn(1, 1, args.image_size, args.image_size)
        torch.onnx.export(
            model.cpu(),
            dummy,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=12,
        )
        print(f"Saved ONNX: {onnx_path}")

    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, val_loader, split_mode, train_count, val_count = build_loaders(args)
    print(f"Dataset split: {split_mode}")
    print(f"Train samples: {train_count}, validation samples: {val_count}")

    model = GestureCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_acc = 0.0
    best_state = None
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": scheduler.get_last_lr()[0],
            }
        )
        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    metrics = {
        "dataset_root": str(Path(args.data_dir).resolve()),
        "split_mode": split_mode,
        "classes": DIGIT_CLASSES,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "best_val_acc": best_acc,
        "history": history,
    }
    save_outputs(model, args, metrics)


if __name__ == "__main__":
    main()
