"""
========================================
@FileName:    自己采集数据
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 15:44
@Description: 
========================================
"""
import cv2
import mediapipe as mp
import os
import time

# 初始化MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def crop_hand_roi(frame, hand_landmarks, margin=30):
    h, w, _ = frame.shape
    # 获取所有关键点的坐标
    x_coords = [int(lm.x * w) for lm in hand_landmarks.landmark]
    y_coords = [int(lm.y * h) for lm in hand_landmarks.landmark]
    # 计算边界框，并添加边距
    x_min, x_max = max(0, min(x_coords) - margin), min(w, max(x_coords) + margin)
    y_min, y_max = max(0, min(y_coords) - margin), min(h, max(y_coords) + margin)
    # 裁剪ROI区域
    roi = frame[y_min:y_max, x_min:x_max].copy()
    return roi


def main():
    save_dir = "new_gesture_data"
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    label = 0  # 当前要采集的手势数字，请在此修改或通过按键切换
    sample_count = 0
    print("按 's' 保存当前手势，按 'n' 切换下一个数字，按 'q' 退出")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]
            # 裁剪手部ROI
            roi = crop_hand_roi(frame, hand_landmarks)
            if roi.size != 0:
                cv2.imshow("ROI", roi)
                # 绘制手部关键点
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label}  Count: {sample_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and result.multi_hand_landmarks:
            img_name = os.path.join(save_dir, f"{label}_{sample_count}.jpg")
            cv2.imwrite(img_name, roi)
            sample_count += 1
            print(f"已保存 {img_name}")
        elif key == ord('n'):
            label = (label + 1) % 10
            sample_count = 0
            print(f"已切换到数字 {label}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
