from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import torch
from PIL import Image
from torchvision import transforms

from qpe_hvit_model import QPE_HViT


SCRIPT_DIR = Path(__file__).resolve().parent
CHECKPOINT_PATH = SCRIPT_DIR / "best_qpe_hvit.pth"
SNAPSHOT_DIR = SCRIPT_DIR / "snapshots"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QPE_HViT 拍照推理")
    parser.add_argument("--camera", type=int, default=0, help="摄像头编号")
    parser.add_argument("--use-mediapipe", action="store_true", help="尝试用 MediaPipe 裁手部 ROI")
    return parser.parse_args()


def build_roi_helper(enabled: bool):
    if not enabled:
        return None, None
    try:
        import mediapipe as mp  # type: ignore
    except Exception:
        return None, None

    hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp, hands


def crop_hand_roi(frame, mp, hands, margin: int = 40):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        return frame, None

    h, w, _ = frame.shape
    hand_landmarks = result.multi_hand_landmarks[0]
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min = max(0, int(min(xs)) - margin)
    y_min = max(0, int(min(ys)) - margin)
    x_max = min(w, int(max(xs)) + margin)
    y_max = min(h, int(max(ys)) + margin)
    roi = frame[y_min:y_max, x_min:x_max]
    if roi.size == 0:
        return frame, None
    return roi, (x_min, y_min, x_max, y_max)


def infer_single_image(model, device, transform, image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    tensor = transform(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        pred = logits.argmax(dim=1).item()
        prob = torch.softmax(logits, dim=1).max().item()
    return pred, prob


def main() -> None:
    args = parse_args()
    if not CHECKPOINT_PATH.exists():
        raise SystemExit(f"未找到模型权重: {CHECKPOINT_PATH}")

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = QPE_HViT(num_classes=10).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()
    model.enable_realtime_optimization()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    mp, hands = build_roi_helper(args.use_mediapipe)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("摄像头打开失败。")

    last_pred = "-"
    last_prob = 0.0
    shot_index = 0

    print("按空格拍照并推理，按 q 退出。")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        roi_box = None
        roi = frame
        if hands is not None:
            roi, roi_box = crop_hand_roi(frame, mp, hands)
            if roi_box is not None:
                x_min, y_min, x_max, y_max = roi_box
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

        cv2.putText(
            frame,
            f"Pred: {last_pred} ({last_prob:.2f})",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            frame,
            "SPACE: capture  Q: quit",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
        cv2.imshow("QPE_HViT Snapshot Inference", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32:
            capture = roi.copy()
            last_pred, last_prob = infer_single_image(model, device, transform, capture)
            shot_path = SNAPSHOT_DIR / f"shot_{shot_index}.jpg"
            cv2.imwrite(str(shot_path), capture)
            shot_index += 1
            print(f"预测结果: {last_pred} ({last_prob:.4f})  保存图像: {shot_path.name}")

    cap.release()
    cv2.destroyAllWindows()
    if hands is not None:
        hands.close()


if __name__ == "__main__":
    main()
