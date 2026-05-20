"""
========================================
@FileName:    六、原图采集并自动标注YOLO.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/20
@Description: 采集原始摄像头画面，并用 MediaPipe 自动生成 YOLO 检测标签
========================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="采集原图并自动标注为 YOLO 数据")
    parser.add_argument("--output", default="dataset_raw_v2", help="输出目录，默认使用新的干净数据目录")
    parser.add_argument("--split", default="train", choices=("train", "val", "test"), help="保存到哪个数据集划分")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--label", type=int, default=0, help="当前采集的手势类别")
    parser.add_argument("--margin", type=int, default=40, help="边界框扩展像素")
    parser.add_argument("--start-index", type=int, default=0, help="起始编号")
    return parser.parse_args()


def resolve_root(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path.resolve()


def ensure_dirs(root: Path, split: str) -> tuple[Path, Path]:
    image_dir = root / "images" / split
    label_dir = root / "labels" / split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    return image_dir, label_dir


def hand_box_from_landmarks(frame, hand_landmarks, margin: int) -> tuple[int, int, int, int]:
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min = max(0, int(min(xs)) - margin)
    y_min = max(0, int(min(ys)) - margin)
    x_max = min(w, int(max(xs)) + margin)
    y_max = min(h, int(max(ys)) + margin)
    return x_min, y_min, x_max, y_max


def to_yolo(x_min: int, y_min: int, x_max: int, y_max: int, width: int, height: int) -> tuple[float, float, float, float]:
    x_center = ((x_min + x_max) / 2) / width
    y_center = ((y_min + y_max) / 2) / height
    box_w = (x_max - x_min) / width
    box_h = (y_max - y_min) / height
    return x_center, y_center, box_w, box_h


def next_index(image_dir: Path, label: int, start_index: int) -> int:
    max_index = start_index - 1
    for image_path in image_dir.glob(f"{label}_*.jpg"):
        try:
            index = int(image_path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        max_index = max(max_index, index)
    return max_index + 1


def save_image_unicode(path: Path, image: np.ndarray) -> bool:
    suffix = path.suffix if path.suffix else ".jpg"
    ok, buffer = cv2.imencode(suffix, image)
    if not ok:
        return False
    try:
        buffer.tofile(str(path))
    except OSError:
        return False
    return True


def main() -> None:
    args = parse_args()
    root = resolve_root(args.output)
    image_dir, label_dir = ensure_dirs(root, args.split)
    sample_index = next_index(image_dir, args.label, args.start_index)

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("摄像头打开失败，请检查设备编号。")

    current_label = args.label
    print(f"保存目录: {root}")
    print("按 s 保存当前原图和 YOLO 标签，按 n 切换到下一个类别，按 q 退出。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            raw_frame = cv2.flip(frame, 1)
            display_frame = raw_frame.copy()
            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)
            yolo_box = None

            if result.multi_hand_landmarks:
                hand_landmarks = result.multi_hand_landmarks[0]
                x_min, y_min, x_max, y_max = hand_box_from_landmarks(raw_frame, hand_landmarks, args.margin)
                yolo_box = to_yolo(x_min, y_min, x_max, y_max, raw_frame.shape[1], raw_frame.shape[0])
                cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            cv2.putText(
                display_frame,
                f"Label: {current_label}  Count: {sample_index}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )
            cv2.imshow("YOLO Raw Data Collection", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("s"):
                if yolo_box is None:
                    print("当前帧未检测到手，未保存。")
                    continue

                image_name = f"{current_label}_{sample_index}.jpg"
                label_name = f"{current_label}_{sample_index}.txt"
                image_path = image_dir / image_name
                label_path = label_dir / label_name

                ok = save_image_unicode(image_path, raw_frame)
                if not ok:
                    print(f"保存失败: {image_path}")
                    continue

                x_center, y_center, box_w, box_h = yolo_box
                label_path.write_text(
                    f"{current_label} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}\n",
                    encoding="utf-8",
                )
                sample_index += 1
                print(f"已保存: {image_path.name}")

            elif key == ord("n"):
                current_label = (current_label + 1) % 10
                sample_index = next_index(image_dir, current_label, 0)
                print(f"已切换到类别 {current_label}")

            elif key == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()
