"""
Predict 0-9 gesture digit from one or more image paths.

Example:
  python predict_gesture_image.py --image ".\\sample.png"
  python predict_gesture_image.py --image ".\\a.png" ".\\b.jpg" --model ".\\gesture_digit_cnn.pt"
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict gesture digit from image path(s).")
    parser.add_argument("--image", nargs="+", required=True, help="One or more image paths.")
    parser.add_argument("--model", default="gesture_digit_cnn.pt", help="Path to trained .pt model.")
    parser.add_argument("--topk", type=int, default=3, help="Number of candidates to print.")
    return parser.parse_args()


def require_dependencies():
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install first:\n"
            "  pip install torch torchvision pillow\n"
        ) from exc
    return torch, nn, Image, transforms


torch, nn, Image, transforms = require_dependencies()


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


def load_model(model_path: Path):
    checkpoint = torch.load(model_path, map_location="cpu")
    model = GestureCNN()
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    input_shape = checkpoint.get("input_shape", [1, 1, 64, 64])
    image_size = int(input_shape[-1])
    classes = [str(item) for item in checkpoint.get("classes", list(range(10)))]
    return model, image_size, classes


def build_transform(image_size: int):
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )


@torch.inference_mode()
def predict_image(model, transform, classes: Sequence[str], image_path: Path, topk: int):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)
    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    values, indices = torch.topk(probabilities, k=min(topk, probabilities.numel()))

    result = []
    for value, index in zip(values.tolist(), indices.tolist()):
        class_name = classes[index] if index < len(classes) else str(index)
        result.append((class_name, value * 100.0))
    return result


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    model, image_size, classes = load_model(model_path)
    transform = build_transform(image_size)

    print(f"Model: {model_path}")
    for raw_path in args.image:
        image_path = Path(raw_path).resolve()
        if not image_path.exists():
            print(f"\nImage: {image_path}")
            print("  Error: file not found")
            continue

        result = predict_image(model, transform, classes, image_path, args.topk)
        print(f"\nImage: {image_path}")
        if result:
            print(f"  Prediction: {result[0][0]} ({result[0][1]:.2f}%)")
            print("  Top candidates:")
            for rank, (class_name, confidence) in enumerate(result, 1):
                print(f"    #{rank}: {class_name}  {confidence:.2f}%")


if __name__ == "__main__":
    main()
