from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SPLITS = ("train", "val", "test")
RATIOS = (0.8, 0.1, 0.1)
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分 QPE_HViT 分类数据集")
    parser.add_argument("--source", default="new_gesture_data", help="原始分类图片目录")
    parser.add_argument("--target", default="dataset_cls", help="划分后的输出目录")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def prepare_dirs(target_dir: Path) -> None:
    for split in SPLITS:
        for class_id in range(10):
            (target_dir / split / str(class_id)).mkdir(parents=True, exist_ok=True)


def clear_dirs(target_dir: Path) -> None:
    if not target_dir.exists():
        return
    for split in SPLITS:
        split_dir = target_dir / split
        if not split_dir.exists():
            continue
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            for file_path in class_dir.iterdir():
                if file_path.is_file():
                    file_path.unlink()


def split_items(items: list[Path]) -> tuple[list[Path], list[Path], list[Path]]:
    train_count = int(len(items) * RATIOS[0])
    val_count = int(len(items) * RATIOS[1])
    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]
    return train_items, val_items, test_items


def main() -> None:
    args = parse_args()
    source_dir = resolve_path(args.source)
    target_dir = resolve_path(args.target)

    if not source_dir.exists():
        raise SystemExit(f"未找到源数据目录: {source_dir}")

    clear_dirs(target_dir)
    prepare_dirs(target_dir)

    grouped: dict[int, list[Path]] = defaultdict(list)
    for image_path in source_dir.glob("*.jpg"):
        class_id = int(image_path.stem.split("_", 1)[0])
        grouped[class_id].append(image_path)

    random.seed(SEED)
    for class_id, items in sorted(grouped.items()):
        random.shuffle(items)
        train_items, val_items, test_items = split_items(items)

        for image_path in train_items:
            shutil.copy2(image_path, target_dir / "train" / str(class_id) / image_path.name)
        for image_path in val_items:
            shutil.copy2(image_path, target_dir / "val" / str(class_id) / image_path.name)
        for image_path in test_items:
            shutil.copy2(image_path, target_dir / "test" / str(class_id) / image_path.name)

        print(
            f"class {class_id}: train={len(train_items)}, "
            f"val={len(val_items)}, test={len(test_items)}"
        )

    print(f"数据划分完成: {target_dir}")


if __name__ == "__main__":
    main()
