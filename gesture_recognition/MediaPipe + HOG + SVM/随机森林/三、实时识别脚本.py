"""
========================================
@FileName:    三、实时识别脚本
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 15:24
@Description: 
========================================
"""
import cv2
import mediapipe as mp
import numpy as np
import pickle
from skimage.feature import hog

# 加载模型（选择 SVM 或 随机森林）
with open("svm_gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)


def crop_hand_roi(frame, hand_landmarks, margin=30):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min, x_max = max(0, int(min(xs)) - margin), min(w, int(max(xs)) + margin)
    y_min, y_max = max(0, int(min(ys)) - margin), min(h, int(max(ys)) + margin)
    roi = frame[y_min:y_max, x_min:x_max]
    return roi


def preprocess_roi(roi):
    """预处理ROI：灰度化、缩放、HOG特征"""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (128, 128))
    feat = hog(resized, orientations=9, pixels_per_cell=(8, 8),
               cells_per_block=(2, 2), block_norm='L2-Hys', visualize=False)
    return feat


cap = cv2.VideoCapture(0)
while cap.isOpened():
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
            # 提取特征并预测
            feat = preprocess_roi(roi).reshape(1, -1)
            pred = model.predict(feat)[0]
            prob = model.predict_proba(feat).max() if hasattr(model, "predict_proba") else 0

            cv2.putText(frame, f"Gesture: {pred} ({prob:.2f})", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
            # 显示ROI小窗口
            cv2.imshow("ROI", roi)
        # 绘制手部关键点
        mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()