"""
========================================
@FileName:    一、数据采集脚本
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 15:23
@Description: 
========================================
"""
import cv2
import mediapipe as mp
import os
import time

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)


def crop_hand_roi(frame, hand_landmarks, margin=30):
    """根据手部关键点裁剪手部区域，返回ROI图像"""
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min, x_max = max(0, int(min(xs)) - margin), min(w, int(max(xs)) + margin)
    y_min, y_max = max(0, int(min(ys)) - margin), min(h, int(max(ys)) + margin)
    roi = frame[y_min:y_max, x_min:x_max]
    return roi


def main():
    save_dir = "gesture_data"
    os.makedirs(save_dir, exist_ok=True)
    cap = cv2.VideoCapture(0)
    label = 0
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
            roi = crop_hand_roi(frame, hand_landmarks)
            if roi.size != 0:
                cv2.imshow("ROI", roi)
                # 绘制关键点
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.putText(frame, f"Label: {label}  Count: {sample_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s') and result.multi_hand_landmarks:
            # 保存 ROI 图像，文件名：label_sample编号.jpg
            img_name = os.path.join(save_dir, f"{label}_{sample_count}.jpg")
            cv2.imwrite(img_name, roi)
            sample_count += 1
            print(f"Saved {img_name}")
        elif key == ord('n'):
            label = (label + 1) % 10
            sample_count = 0
            print(f"Switched to label {label}")
        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()