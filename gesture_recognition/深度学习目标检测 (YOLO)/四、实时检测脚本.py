"""
========================================
@FileName:    四、实时检测脚本.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/20
@Description: 调用训练好的 YOLO 模型进行实时手势检测
========================================
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from ultralytics import YOLO
except ImportError as exc:
    raise SystemExit("未检测到 ultralytics，请先运行: pip install ultralytics") from exc


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 实时手势检测")
    parser.add_argument(
        "--weights",
        default=None,
        help="模型权重路径；不传时自动搜索当前项目下最新的 best.pt",
    )
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument(
        "--match-training-overlays",
        action="store_true",
        help="对实时画面先绘制 MediaPipe 关键点和手框，以匹配旧训练数据分布",
    )
    return parser.parse_args()


def resolve_weights(user_weights: str | None) -> Path:
    if user_weights:
        weights_path = Path(user_weights).expanduser()
        if not weights_path.is_absolute():
            weights_path = (SCRIPT_DIR / weights_path).resolve()
        if weights_path.exists():
            return weights_path
        raise SystemExit(f"未找到模型权重: {weights_path}")

    candidates = sorted(
        SCRIPT_DIR.glob("runs/**/weights/best.pt"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if candidates:
        return candidates[0]

    raise SystemExit("未找到任何 best.pt，请先完成训练，或手动通过 --weights 指定权重路径。")


def maybe_build_overlay_pipeline(enabled: bool):
    if not enabled:
        return None, None, None

    try:
        import mediapipe as mp  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "启用 --match-training-overlays 需要 mediapipe，请在当前环境安装后重试。"
        ) from exc

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp_hands, mp_draw, hands


def draw_training_style_overlay(frame, mp_hands, mp_draw, hands, margin: int = 40):
    display_frame = frame.copy()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        xs = [lm.x * w for lm in hand_landmarks.landmark]
        ys = [lm.y * h for lm in hand_landmarks.landmark]
        x_min = max(0, int(min(xs)) - margin)
        y_min = max(0, int(min(ys)) - margin)
        x_max = min(w, int(max(xs)) + margin)
        y_max = min(h, int(max(ys)) + margin)
        cv2.rectangle(display_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # 旧数据里左上角有固定调试文字，这里补一个接近的分布
    cv2.putText(
        display_frame,
        "Label: 0  Count: 0",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
    )
    return display_frame


def main() -> None:
    args = parse_args()
    weights_path = resolve_weights(args.weights)
    print(f"使用模型权重: {weights_path}")

    mp_hands, mp_draw, hands = maybe_build_overlay_pipeline(args.match_training_overlays)

    model = YOLO(str(weights_path))
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise SystemExit("摄像头打开失败，请检查设备编号。")

    print("按 q 退出实时检测。")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("摄像头读取失败，程序结束。")
                break

            frame = cv2.flip(frame, 1)
            infer_frame = frame
            if hands is not None:
                infer_frame = draw_training_style_overlay(frame, mp_hands, mp_draw, hands)

            results = model.predict(infer_frame, conf=args.conf, verbose=False)
            annotated = results[0].plot()

            cv2.imshow("YOLO Gesture Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
        if hands is not None:
            hands.close()


if __name__ == "__main__":
    main()
