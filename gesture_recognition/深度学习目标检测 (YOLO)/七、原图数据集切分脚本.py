"""
========================================
@FileName:    七、原图数据集切分脚本.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/20
@Description: 将原图 YOLO 数据按类别分层切分为 train/val/test
========================================
"""
from __future__ import annotations

import argparse
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SAMPLE_NAME_RE = re.compile(r"^\d+_\d+$")
SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="切分原图 YOLO 数据集")
    parser.add_argument("--source", default="dataset_raw_v2", help="原始数据根目录")
    parser.add_argument("--target", default="dataset", help="切分后的数据根目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--clear-target", action="store_true", help="切分前清空目标数据目录")
    return parser.parse_args()


def resolve_root(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def validate_ratios(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("train/val/test 比例之和必须等于 1.0")


def ensure_dirs(root: Path) -> None:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def clear_target_dir(root: Path) -> None:
    for split in ("train", "val", "test"):
        for subdir in (root / "images" / split, root / "labels" / split):
            if not subdir.exists():
                continue
            for item in subdir.iterdir():
                if item.name == ".gitkeep":
                    continue
                if item.is_file():
                    item.unlink()


def parse_class_id(image_path: Path) -> int:
    return int(image_path.stem.split("_", 1)[0])


def is_valid_sample_name(image_path: Path) -> bool:
    return SAMPLE_NAME_RE.fullmatch(image_path.stem) is not None


def collect_samples(source_root: Path) -> dict[int, list[tuple[Path, Path]]]:
    image_root = source_root / "images"
    label_root = source_root / "labels"
    grouped: dict[int, list[tuple[Path, Path]]] = defaultdict(list)

    if not image_root.exists() or not label_root.exists():
        raise SystemExit(f"源目录结构不完整: {source_root}")

    for split_dir in image_root.iterdir():
        if not split_dir.is_dir():
            continue
        for image_path in split_dir.iterdir():
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            if not is_valid_sample_name(image_path):
                print(f"跳过非正式样本文件: {image_path.name}")
                continue
            label_path = label_root / split_dir.name / f"{image_path.stem}.txt"
            if not label_path.exists():
                raise SystemExit(f"缺少标签文件: {label_path}")
            class_id = parse_class_id(image_path)
            grouped[class_id].append((image_path, label_path))

    if not grouped:
        raise SystemExit(f"未在 {source_root} 中找到图片样本")

    return grouped


def split_count(total: int, train_ratio: float, val_ratio: float) -> tuple[int, int, int]:
    train_count = int(total * train_ratio)
    val_count = int(total * val_ratio)
    test_count = total - train_count - val_count
    return train_count, val_count, test_count


def copy_sample(image_path: Path, label_path: Path, target_root: Path, split: str) -> None:
    target_image = target_root / "images" / split / image_path.name
    target_label = target_root / "labels" / split / label_path.name
    shutil.copy2(image_path, target_image)
    shutil.copy2(label_path, target_label)


def main() -> None:
    args = parse_args()
    validate_ratios(args)

    source_root = resolve_root(args.source)
    target_root = resolve_root(args.target)
    ensure_dirs(target_root)

    if args.clear_target:
        clear_target_dir(target_root)

    grouped = collect_samples(source_root)
    random.seed(args.seed)

    summary_lines = []
    total_counts = {"train": 0, "val": 0, "test": 0}

    for class_id, samples in sorted(grouped.items()):
        random.shuffle(samples)
        train_count, val_count, test_count = split_count(len(samples), args.train_ratio, args.val_ratio)
        train_samples = samples[:train_count]
        val_samples = samples[train_count:train_count + val_count]
        test_samples = samples[train_count + val_count:]

        for image_path, label_path in train_samples:
            copy_sample(image_path, label_path, target_root, "train")
        for image_path, label_path in val_samples:
            copy_sample(image_path, label_path, target_root, "val")
        for image_path, label_path in test_samples:
            copy_sample(image_path, label_path, target_root, "test")

        total_counts["train"] += len(train_samples)
        total_counts["val"] += len(val_samples)
        total_counts["test"] += len(test_samples)

        line = f"类别 {class_id}: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
        summary_lines.append(line)
        print(line)

    report_lines = [
        f"源目录: {source_root}",
        f"目标目录: {target_root}",
        f"train={total_counts['train']}",
        f"val={total_counts['val']}",
        f"test={total_counts['test']}",
        "",
        "按类别统计:",
        *summary_lines,
    ]
    report_path = target_root / "split_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("-" * 50)
    print(f"切分完成: train={total_counts['train']}, val={total_counts['val']}, test={total_counts['test']}")
    print(f"报告已写入: {report_path}")


if __name__ == "__main__":
    main()
