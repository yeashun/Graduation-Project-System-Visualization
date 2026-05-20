"""
========================================
@FileName:    三、训练脚本.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/20
@Description: 使用 YOLOv8 训练手势目标检测模型（默认针对 CPU 环境优化）
========================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("未检测到 ultralytics，请先运行: pip install ultralytics") from exc


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练 YOLOv8 手势检测模型")
    parser.add_argument("--data", default="二、数据集配置.yaml", help="数据集配置文件")
    parser.add_argument("--model", default="yolov8n.pt", help="预训练模型名称或权重路径")
    parser.add_argument("--epochs", type=int, default=30, help="训练轮数")
    parser.add_argument("--imgsz", type=int, default=320, help="输入图像尺寸")
    parser.add_argument("--batch", type=int, default=8, help="批大小")
    parser.add_argument("--workers", type=int, default=0, help="数据加载线程数，CPU 下建议 0")
    parser.add_argument("--patience", type=int, default=10, help="早停轮数")
    parser.add_argument("--cache", action="store_true", help="将图片缓存到内存，加快后续 epoch")
    parser.add_argument("--device", default=None, help="手动指定设备，如 cpu、0")
    parser.add_argument("--project", default="runs/train", help="训练输出目录")
    parser.add_argument("--name", default="gesture_yolo_clean", help="实验名称")
    return parser.parse_args()


def resolve_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def resolve_device(user_device: str | None) -> str:
    if user_device:
        return user_device
    return "0" if torch.cuda.is_available() else "cpu"


def cpu_friendly_defaults(device: str, batch: int, workers: int) -> tuple[int, int]:
    if device == "cpu":
        return min(batch, 8), 0
    return batch, workers


def build_absolute_data_yaml(data_yaml_path: Path) -> Path:
    dataset_root = (data_yaml_path.parent / "dataset").resolve()
    abs_yaml_path = data_yaml_path.with_name("dataset_abs.yaml")
    abs_yaml_path.write_text(
        "\n".join(
            [
                f"path: {dataset_root.as_posix()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "",
                "names:",
                '  0: "0"',
                '  1: "1"',
                '  2: "2"',
                '  3: "3"',
                '  4: "4"',
                '  5: "5"',
                '  6: "6"',
                '  7: "7"',
                '  8: "8"',
                '  9: "9"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    return abs_yaml_path


def main() -> None:
    args = parse_args()
    data_path = resolve_path(args.data)
    if not data_path.exists():
        raise SystemExit(f"未找到数据集配置文件: {data_path}")
    abs_data_path = build_absolute_data_yaml(data_path)

    project_path = resolve_path(args.project)
    device = resolve_device(args.device)
    batch, workers = cpu_friendly_defaults(device, args.batch, args.workers)

    print(f"使用设备: {device}")
    print(f"数据配置: {abs_data_path}")
    print(f"模型权重: {args.model}")
    print(
        f"训练参数: epochs={args.epochs}, imgsz={args.imgsz}, "
        f"batch={batch}, workers={workers}, patience={args.patience}, cache={args.cache}"
    )
    print(f"输出目录: {project_path}")

    model = YOLO(str(resolve_path(args.model)) if Path(args.model).suffix else args.model)
    model.train(
        data=str(abs_data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        workers=workers,
        patience=args.patience,
        cache=args.cache,
        amp=False if device == "cpu" else True,
        device=device,
        project=str(project_path),
        name=args.name,
        pretrained=True,
        verbose=True,
    )

    print("训练完成。")
    print(f"结果目录: {(project_path / args.name).resolve()}")


if __name__ == "__main__":
    main()
