"""
========================================
@FileName:    八、清洗旧采集数据.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/20
@Description: 清洗旧采集图中的绿色文字和绿色框，尽量挽救已采集的 YOLO 数据
========================================
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="清洗旧采集图中的绿色覆盖层")
    parser.add_argument("--source", default="dataset_raw", help="原始数据根目录")
    parser.add_argument("--target", default="dataset_cleaned", help="清洗后数据根目录")
    return parser.parse_args()


def ensure_dirs(target_root: Path) -> None:
    for split in ("train", "val", "test"):
        (target_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clean_overlay(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 绿色文字和绿色框
    green_mask = cv2.inRange(hsv, (35, 80, 80), (90, 255, 255))

    # 红色关键点
    red_mask1 = cv2.inRange(hsv, (0, 100, 80), (10, 255, 255))
    red_mask2 = cv2.inRange(hsv, (160, 100, 80), (179, 255, 255))
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    mask = cv2.bitwise_or(green_mask, red_mask)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)

    # 只去掉明显覆盖层，尽量保留原画面
    cleaned = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    return cleaned


def copy_and_clean_split(source_root: Path, target_root: Path, split: str) -> tuple[int, int]:
    image_dir = source_root / "images" / split
    label_dir = source_root / "labels" / split
    target_image_dir = target_root / "images" / split
    target_label_dir = target_root / "labels" / split

    if not image_dir.exists():
        return 0, 0

    image_count = 0
    label_count = 0
    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTS:
            continue

        target_image_path = target_image_dir / image_path.name
        target_label_path = target_label_dir / f"{image_path.stem}.txt"
        if target_image_path.exists() and target_label_path.exists():
            image_count += 1
            label_count += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        cleaned = clean_overlay(image)
        cv2.imwrite(str(target_image_path), cleaned)
        image_count += 1

        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            shutil.copy2(label_path, target_label_path)
            label_count += 1

    return image_count, label_count


def main() -> None:
    args = parse_args()
    source_root = Path(args.source)
    target_root = Path(args.target)
    ensure_dirs(target_root)

    total_images = 0
    total_labels = 0
    for split in ("train", "val", "test"):
        images, labels = copy_and_clean_split(source_root, target_root, split)
        total_images += images
        total_labels += labels
        print(f"{split}: images={images}, labels={labels}")

    print("-" * 50)
    print(f"清洗完成: images={total_images}, labels={total_labels}")
    print(f"输出目录: {target_root.resolve()}")


if __name__ == "__main__":
    main()
