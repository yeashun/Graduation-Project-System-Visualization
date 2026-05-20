"""
========================================
@FileName:    一、标注数据检查脚本.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19
@Description: 检查 YOLO 数据集目录、标签文件和边界框是否合法
========================================
"""
from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="检查 YOLO 手势检测数据集")
    parser.add_argument("--dataset", default="dataset", help="数据集根目录")
    parser.add_argument("--num-classes", type=int, default=10, help="类别总数")
    return parser.parse_args()


def validate_label_file(label_path: Path, num_classes: int) -> list[str]:
    errors: list[str] = []
    lines = label_path.read_text(encoding="utf-8").splitlines()

    if not lines:
        errors.append(f"空标签文件: {label_path}")
        return errors

    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            errors.append(f"空白标注行: {label_path} 第 {line_no} 行")
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(f"字段数量错误: {label_path} 第 {line_no} 行 -> {line}")
            continue

        try:
            class_id = int(float(parts[0]))
            x_center, y_center, width, height = map(float, parts[1:])
        except ValueError:
            errors.append(f"存在非数字字段: {label_path} 第 {line_no} 行 -> {line}")
            continue

        if not 0 <= class_id < num_classes:
            errors.append(f"类别越界: {label_path} 第 {line_no} 行 -> class_id={class_id}")

        for name, value in (
            ("x_center", x_center),
            ("y_center", y_center),
            ("width", width),
            ("height", height),
        ):
            if not 0.0 <= value <= 1.0:
                errors.append(f"归一化坐标越界: {label_path} 第 {line_no} 行 -> {name}={value}")

        if width <= 0 or height <= 0:
            errors.append(f"框尺寸非法: {label_path} 第 {line_no} 行 -> w={width}, h={height}")

    return errors


def inspect_split(dataset_root: Path, split: str, num_classes: int) -> tuple[int, list[str]]:
    image_dir = dataset_root / "images" / split
    label_dir = dataset_root / "labels" / split
    errors: list[str] = []

    if not image_dir.exists():
        errors.append(f"缺少图片目录: {image_dir}")
        return 0, errors

    if not label_dir.exists():
        errors.append(f"缺少标签目录: {label_dir}")
        return 0, errors

    image_paths = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        errors.append(f"目录中没有图片: {image_dir}")
        return 0, errors

    valid_count = 0
    for image_path in image_paths:
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"缺少对应标签: {image_path.name}")
            continue

        file_errors = validate_label_file(label_path, num_classes)
        if file_errors:
            errors.extend(file_errors)
            continue

        valid_count += 1

    return valid_count, errors


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset)

    print(f"开始检查数据集: {dataset_root.resolve()}")
    total_valid = 0
    all_errors: list[str] = []

    for split in SPLITS:
        valid_count, errors = inspect_split(dataset_root, split, args.num_classes)
        total_valid += valid_count
        all_errors.extend(errors)
        print(f"[{split}] 有效样本数: {valid_count}")

    print("-" * 50)
    if all_errors:
        print("发现以下问题：")
        for error in all_errors:
            print(f"- {error}")
        print("-" * 50)
        print(f"检查完成，共发现 {len(all_errors)} 个问题。")
    else:
        print(f"检查通过，共发现 {total_valid} 个有效样本。")


if __name__ == "__main__":
    main()
