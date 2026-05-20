"""
========================================
@FileName:    train_cnn
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 19:26
@Description:
========================================
"""
import argparse
import os
import random
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pad_to_square(img):
    is_pil = isinstance(img, Image.Image)
    if is_pil:
        img = np.array(img)

    h, w = img.shape[:2]
    if h == w:
        return Image.fromarray(img) if is_pil else img

    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    padded = cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return Image.fromarray(padded) if is_pil else padded


class GestureDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []
        for fname in sorted(os.listdir(data_dir)):
            if fname.endswith(".jpg"):
                label = int(fname.split("_")[0])
                path = os.path.join(data_dir, fname)
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, label


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def build_transforms():
    train_transform = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Lambda(pad_to_square),
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transform, val_transform


def build_dataloaders(data_dir, batch_size, val_ratio, seed):
    train_transform, val_transform = build_transforms()
    full_dataset = GestureDataset(data_dir, transform=None)
    counts = Counter(label for _, label in full_dataset.samples)

    print("每类样本数:")
    for label in sorted(counts):
        print(f"  类别 {label}: {counts[label]}")

    train_size = int((1 - val_ratio) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_dataset = TransformSubset(train_subset, train_transform)
    val_dataset = TransformSubset(val_subset, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def evaluate(model, data_loader, device, num_classes=10):
    model.eval()
    confusion = np.zeros((num_classes, num_classes), dtype=np.int32)
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predicted = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for true_label, pred_label in zip(labels.cpu().numpy(), predicted.cpu().numpy()):
                confusion[true_label, pred_label] += 1

    acc = 100.0 * correct / total if total else 0.0
    return acc, confusion


def save_confusion_matrix(confusion, save_path):
    plt.figure(figsize=(8, 6))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.colorbar()
    ticks = np.arange(confusion.shape[0])
    plt.xticks(ticks, ticks)
    plt.yticks(ticks, ticks)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = confusion.max() / 2 if confusion.size else 0
    for i in range(confusion.shape[0]):
        for j in range(confusion.shape[1]):
            color = "white" if confusion[i, j] > threshold else "black"
            plt.text(j, i, str(confusion[i, j]), ha="center", va="center", color=color, fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def print_per_class_accuracy(confusion):
    print("每类准确率:")
    for label in range(confusion.shape[0]):
        total = confusion[label].sum()
        correct = confusion[label, label]
        acc = 100.0 * correct / total if total else 0.0
        print(f"  类别 {label}: {correct}/{total} = {acc:.2f}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Train a CNN for hand gesture recognition.")
    parser.add_argument("--data-dir", default="new_gesture_data")
    parser.add_argument("--model-path", default="best_gesture_cnn.pth")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    train_loader, val_loader = build_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN(num_classes=args.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_acc = 0.0
    print(f"使用设备: {device}")
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100.0 * correct / total if total else 0.0
        val_acc, _ = evaluate(model, val_loader, device, num_classes=args.num_classes)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] "
            f"Loss: {running_loss / len(train_loader):.4f} "
            f"Train Acc: {train_acc:.2f}% "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), args.model_path)

        scheduler.step()

    print(f"最佳验证准确率: {best_acc:.2f}%")

    model.load_state_dict(torch.load(args.model_path, map_location=device))
    final_val_acc, confusion = evaluate(model, val_loader, device, num_classes=args.num_classes)
    print(f"最终最佳模型验证准确率: {final_val_acc:.2f}%")
    print_per_class_accuracy(confusion)
    save_confusion_matrix(confusion, "confusion_matrix.png")
    print("已保存混淆矩阵: confusion_matrix.png")


if __name__ == "__main__":
    main()
