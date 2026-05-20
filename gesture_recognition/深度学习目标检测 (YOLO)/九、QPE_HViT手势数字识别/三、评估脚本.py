from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from qpe_hvit_model import QPE_HViT


SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_ROOT = SCRIPT_DIR / "dataset_cls"
CHECKPOINT_PATH = SCRIPT_DIR / "best_qpe_hvit.pth"


def main() -> None:
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"未找到模型权重: {CHECKPOINT_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    test_set = datasets.ImageFolder(DATASET_ROOT / "test", transform=transform)
    test_loader = DataLoader(test_set, batch_size=16, shuffle=False, num_workers=0)

    model = QPE_HViT(num_classes=10).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    acc = correct / total if total else 0.0
    print(f"test_acc={acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    main()
