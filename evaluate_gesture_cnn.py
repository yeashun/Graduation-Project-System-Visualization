"""
Evaluate a trained GestureCNN model on image folders.

Example:
  python evaluate_gesture_cnn.py --data-dir ".\\dataset\\test"
  python evaluate_gesture_cnn.py --data-dir ".\\dataset_capture\\train" --model ".\\gesture_digit_cnn.pt"
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DIGIT_CLASSES = [str(index) for index in range(10)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate gesture_digit_cnn.pt on 0-9 image folders.")
    parser.add_argument("--data-dir", required=True, help="Root folder containing class folders 0-9.")
    parser.add_argument("--model", default="gesture_digit_cnn.pt", help="Path to trained .pt model.")
    parser.add_argument("--output", default="gesture_eval_errors.csv", help="CSV path for wrong predictions.")
    parser.add_argument("--batch-size", type=int, default=128, help="Evaluation batch size.")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers.")
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
            "Missing dependency. Install first:\n"
            "  pip install torch torchvision pillow\n"
        ) from exc
    return torch, nn, F, Image, DataLoader, Dataset, transforms


torch, nn, F, Image, DataLoader, Dataset, transforms = require_dependencies()


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


class ImageFolderDigits(Dataset):
    def __init__(self, samples: Sequence[Tuple[Path, int]], transform) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label, str(path)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def class_dir_candidates(root: Path, class_name: str) -> Iterable[Path]:
    yield root / class_name
    yield root / f"digit_{class_name}"
    yield root / f"Digit_{class_name}"


def collect_samples(root: Path) -> List[Tuple[Path, int]]:
    samples: List[Tuple[Path, int]] = []
    missing = []
    for label, class_name in enumerate(DIGIT_CLASSES):
        class_dir = next((candidate for candidate in class_dir_candidates(root, class_name) if candidate.is_dir()), None)
        if class_dir is None:
            missing.append(class_name)
            continue
        images = sorted(path for path in class_dir.rglob("*") if path.is_file() and is_image(path))
        samples.extend((path, label) for path in images)

    if missing:
        print(f"Warning: missing class folders: {', '.join(missing)}")
    if not samples:
        raise SystemExit(f"No images found under: {root}")
    return samples


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = GestureCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    input_shape = checkpoint.get("input_shape", [1, 1, 64, 64])
    image_size = int(input_shape[-1])
    classes = [str(item) for item in checkpoint.get("classes", DIGIT_CLASSES)]
    return model, image_size, classes


@torch.inference_mode()
def evaluate(model, loader, classes: Sequence[str]) -> Tuple[dict, List[dict]]:
    total = 0
    correct = 0
    per_class_total: Dict[int, int] = {index: 0 for index in range(10)}
    per_class_correct: Dict[int, int] = {index: 0 for index in range(10)}
    errors: List[dict] = []

    for images, labels, paths in loader:
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        confs, preds = torch.max(probs, dim=1)

        for label, pred, conf, path in zip(labels.tolist(), preds.tolist(), confs.tolist(), paths):
            total += 1
            per_class_total[label] += 1
            if pred == label:
                correct += 1
                per_class_correct[label] += 1
            else:
                errors.append(
                    {
                        "path": path,
                        "label": classes[label] if label < len(classes) else str(label),
                        "prediction": classes[pred] if pred < len(classes) else str(pred),
                        "confidence": f"{conf * 100:.2f}",
                    }
                )

    accuracy = correct / total if total else 0.0
    per_class = {}
    for label in range(10):
        class_total = per_class_total[label]
        class_correct = per_class_correct[label]
        per_class[classes[label] if label < len(classes) else str(label)] = {
            "correct": class_correct,
            "total": class_total,
            "accuracy": class_correct / class_total if class_total else 0.0,
        }

    return {"total": total, "correct": correct, "accuracy": accuracy, "per_class": per_class}, errors


def write_errors(path: Path, errors: Sequence[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=["path", "label", "prediction", "confidence"])
        writer.writeheader()
        writer.writerows(errors)


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    model_path = Path(args.model).resolve()
    output_path = Path(args.output).resolve()

    model, image_size, classes = load_model(model_path)
    samples = collect_samples(data_dir)
    dataset = ImageFolderDigits(samples, build_transform(image_size))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    metrics, errors = evaluate(model, loader, classes)
    write_errors(output_path, errors)

    print(f"Model: {model_path}")
    print(f"Data: {data_dir}")
    print(f"Total: {metrics['total']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy'] * 100:.2f}%")
    print("\nPer-class accuracy:")
    for class_name, item in metrics["per_class"].items():
        print(f"  {class_name}: {item['correct']}/{item['total']} = {item['accuracy'] * 100:.2f}%")
    print(f"\nWrong predictions: {len(errors)}")
    print(f"Error CSV: {output_path}")


if __name__ == "__main__":
    main()
