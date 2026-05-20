"""
========================================
@FileName:    test
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 20:05
@Description:
========================================
"""
import argparse

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


def pad_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    return cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run offline prediction for one ROI image.")
    parser.add_argument("--image", default="debug_roi.jpg")
    parser.add_argument("--model-path", default="best_gesture_cnn.pth")
    parser.add_argument("--num-classes", type=int, default=10)
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cpu")
    model = SimpleCNN(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = cv2.imread(args.image)
    if img is None:
        print(f"图片加载失败: {args.image}")
        return

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = pad_to_square(img)
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        probs = torch.softmax(output, dim=1).squeeze()

    print(f"预测类别: {pred}")
    for idx, prob in enumerate(probs.tolist()):
        print(f"  类别 {idx}: {prob:.4f}")


if __name__ == "__main__":
    main()
