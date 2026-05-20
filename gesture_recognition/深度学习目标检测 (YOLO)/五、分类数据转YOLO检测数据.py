"""
========================================
@FileName:    五、分类数据转YOLO检测数据.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19
@Description: 将单步分类手势数据自动转换为 YOLO 检测数据集
========================================
"""
from __future__ import annotations

import argparse
import random
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将分类图片自动转换成 YOLO 检测数据集")
    parser.add_argument(
        "--source-dir",
        default=r"D:\桌面\研0\毕业设计\可视化展示界面\gesture_recognition\深度学习单步分类（End-to-End CNN）\new_gesture_data",
        help="分类图片目录",
    )
    parser.add_argument("--target-dir", default="dataset", help="YOLO 数据集输出目录")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="验证集比例")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="测试集比例")
    parser.add_argument("--margin", type=int, default=30, help="边界框额外扩展像素")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--box-mode",
        choices=("auto", "mediapipe", "full-image"),
        default="auto",
        help="边界框生成方式：auto 为优先 MediaPipe，失败后退回整图框",
    )
    parser.add_argument(
        "--copy-failed-to",
        default="failed_samples",
        help="未检测到手的图片输出目录，设为空字符串则不复制",
    )
    return parser.parse_args()


def ensure_ratio_valid(args: argparse.Namespace) -> None:
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total - 1.0) > 1e-6:
        raise SystemExit("train/val/test 比例之和必须等于 1.0")


def parse_class_id(image_path: Path) -> int:
    stem = image_path.stem
    if "_" not in stem:
        raise ValueError(f"文件名不符合 '类别_编号' 规则: {image_path.name}")
    return int(stem.split("_", 1)[0])


def try_import_cv2():
    try:
        import cv2  # type: ignore
    except ImportError:
        return None
    return cv2


def try_import_mediapipe():
    try:
        import mediapipe as mp  # type: ignore
    except ImportError:
        return None
    except Exception:
        return None
    return mp


def full_image_box() -> tuple[float, float, float, float]:
    return 0.5, 0.5, 1.0, 1.0


def detect_hand_box(cv2, hands, image_path: Path, margin: int) -> tuple[float, float, float, float] | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    h, w = image.shape[:2]
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    x_min = max(0, int(min(xs)) - margin)
    y_min = max(0, int(min(ys)) - margin)
    x_max = min(w, int(max(xs)) + margin)
    y_max = min(h, int(max(ys)) + margin)

    if x_max <= x_min or y_max <= y_min:
        return None

    x_center = ((x_min + x_max) / 2) / w
    y_center = ((y_min + y_max) / 2) / h
    box_w = (x_max - x_min) / w
    box_h = (y_max - y_min) / h
    return x_center, y_center, box_w, box_h


def split_items(items: list[Path], train_ratio: float, val_ratio: float) -> dict[str, list[Path]]:
    train_count = int(len(items) * train_ratio)
    val_count = int(len(items) * val_ratio)
    train_items = items[:train_count]
    val_items = items[train_count:train_count + val_count]
    test_items = items[train_count + val_count:]
    return {"train": train_items, "val": val_items, "test": test_items}


def prepare_output_dirs(target_dir: Path) -> None:
    for split in ("train", "val", "test"):
        (target_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (target_dir / "labels" / split).mkdir(parents=True, exist_ok=True)


def write_label(label_path: Path, class_id: int, box: tuple[float, float, float, float]) -> None:
    x_center, y_center, box_w, box_h = box
    content = f"{class_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n"
    label_path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_ratio_valid(args)

    source_dir = Path(args.source_dir)
    target_dir = Path(args.target_dir)
    failed_dir = Path(args.copy_failed_to) if args.copy_failed_to else None

    if not source_dir.exists():
        raise SystemExit(f"未找到源数据目录: {source_dir}")

    prepare_output_dirs(target_dir)
    if failed_dir is not None:
        failed_dir.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(path for path in source_dir.iterdir() if path.suffix.lower() in IMAGE_EXTS)
    if not image_paths:
        raise SystemExit(f"源目录中没有图片: {source_dir}")

    cv2 = None
    mp = None
    hands_context = None
    can_use_mediapipe = False

    if args.box_mode in {"auto", "mediapipe"}:
        cv2 = try_import_cv2()
        mp = try_import_mediapipe()
        can_use_mediapipe = cv2 is not None and mp is not None
        if args.box_mode == "mediapipe" and not can_use_mediapipe:
            raise SystemExit("当前环境不可用 MediaPipe/OpenCV，请改用 --box-mode full-image 或 auto")

    grouped: dict[int, list[Path]] = defaultdict(list)
    invalid_names: list[Path] = []
    for image_path in image_paths:
        try:
            class_id = parse_class_id(image_path)
        except ValueError:
            invalid_names.append(image_path)
            continue
        grouped[class_id].append(image_path)

    if invalid_names:
        print("以下文件名不符合规则，已跳过：")
        for path in invalid_names:
            print(f"- {path.name}")

    random.seed(args.seed)
    split_mapping: dict[str, list[Path]] = {"train": [], "val": [], "test": []}
    for class_id, items in grouped.items():
        random.shuffle(items)
        split_result = split_items(items, args.train_ratio, args.val_ratio)
        for split, split_items_list in split_result.items():
            split_mapping[split].extend(split_items_list)
        print(
            f"类别 {class_id}: train={len(split_result['train'])}, "
            f"val={len(split_result['val'])}, test={len(split_result['test'])}"
        )

    converted = 0
    failed = 0
    failed_names: list[str] = []
    fallback_count = 0

    if can_use_mediapipe:
        hands_context = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5,
        )

    try:
        hands = hands_context.__enter__() if hands_context is not None else None
        for split, split_images in split_mapping.items():
            for image_path in split_images:
                class_id = parse_class_id(image_path)
                box = None

                if can_use_mediapipe and args.box_mode in {"auto", "mediapipe"}:
                    box = detect_hand_box(cv2, hands, image_path, args.margin)

                if box is None and args.box_mode in {"auto", "full-image"}:
                    box = full_image_box()
                    fallback_count += 1

                if box is None:
                    failed += 1
                    failed_names.append(image_path.name)
                    if failed_dir is not None:
                        shutil.copy2(image_path, failed_dir / image_path.name)
                    continue

                target_image = target_dir / "images" / split / image_path.name
                target_label = target_dir / "labels" / split / f"{image_path.stem}.txt"
                shutil.copy2(image_path, target_image)
                write_label(target_label, class_id, box)
                converted += 1
    finally:
        if hands_context is not None:
            hands_context.__exit__(None, None, None)

    report_lines = [
        f"源目录: {source_dir}",
        f"输出目录: {target_dir.resolve()}",
        f"框生成模式: {args.box_mode}",
        f"成功转换: {converted}",
        f"检测失败: {failed}",
        f"整图框回退次数: {fallback_count}",
    ]
    if failed_dir is not None:
        report_lines.append(f"失败样本目录: {failed_dir.resolve()}")
    if failed_names:
        report_lines.append("")
        report_lines.append("未检测到手的文件：")
        report_lines.extend(failed_names)

    report_path = target_dir / "conversion_report.txt"
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print("-" * 50)
    print(f"转换完成，成功生成 {converted} 个样本。")
    print(f"检测失败 {failed} 个样本。")
    print(f"报告已写入: {report_path.resolve()}")


if __name__ == "__main__":
    main()
