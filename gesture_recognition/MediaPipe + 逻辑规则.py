"""
========================================
@FileName:    MediaPipe + 逻辑规则
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 15:01
@Description: 完全兼容新版MediaPipe，逻辑不变，开箱即用
========================================
"""
import cv2
import mediapipe as mp
import math

# ====================== 稳定版解决方案（不依赖模型文件） ======================
# 直接使用旧版API风格，完全兼容你的代码！
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

# ====================== 你原来的代码 一行不动 ======================
WRIST = 0
THUMB_CMC = 1
THUMB_MCP = 2
THUMB_IP = 3
THUMB_TIP = 4
INDEX_FINGER_MCP = 5
INDEX_FINGER_PIP = 6
INDEX_FINGER_DIP = 7
INDEX_FINGER_TIP = 8
MIDDLE_FINGER_MCP = 9
MIDDLE_FINGER_PIP = 10
MIDDLE_FINGER_DIP = 11
MIDDLE_FINGER_TIP = 12
RING_FINGER_MCP = 13
RING_FINGER_PIP = 14
RING_FINGER_DIP = 15
RING_FINGER_TIP = 16
PINKY_MCP = 17
PINKY_PIP = 18
PINKY_DIP = 19
PINKY_TIP = 20


def get_landmark_array(hand_landmarks, h, w):
    """将归一化坐标转换为像素坐标列表"""
    points = []
    for lm in hand_landmarks.landmark:
        points.append((int(lm.x * w), int(lm.y * h)))
    return points


def distance(p1, p2):
    """计算两点欧氏距离"""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def is_finger_extended(points, tip_idx, pip_idx, is_thumb=False):
    if is_thumb:
        tip = points[tip_idx]
        mcp = points[THUMB_MCP]
        ip = points[THUMB_IP]
        dist_tip_mcp = distance(tip, mcp)
        dist_ip_mcp = distance(ip, mcp)
        return dist_tip_mcp > dist_ip_mcp * 1.2
    else:
        tip = points[tip_idx]
        pip = points[pip_idx]
        return tip[1] < pip[1] - 10


def get_finger_states(points):
    states = []
    states.append(1 if is_finger_extended(points, THUMB_TIP, THUMB_MCP, is_thumb=True) else 0)
    states.append(1 if is_finger_extended(points, INDEX_FINGER_TIP, INDEX_FINGER_PIP) else 0)
    states.append(1 if is_finger_extended(points, MIDDLE_FINGER_TIP, MIDDLE_FINGER_PIP) else 0)
    states.append(1 if is_finger_extended(points, RING_FINGER_TIP, RING_FINGER_PIP) else 0)
    states.append(1 if is_finger_extended(points, PINKY_TIP, PINKY_PIP) else 0)
    return states


def recognize_number(states, points):
    thumb, index, middle, ring, pinky = states
    if sum(states) == 0:
        return 0
    if thumb == 1 and pinky == 1 and index == 0 and middle == 0 and ring == 0:
        return 6
    if thumb == 1 and index == 1 and middle == 0 and ring == 0 and pinky == 0:
        return 8
    cnt = sum(states)
    if 1 <= cnt <= 5:
        return cnt
    return -1


def main():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                h, w, _ = frame.shape
                points = get_landmark_array(hand_landmarks, h, w)
                states = get_finger_states(points)
                number = recognize_number(states, points)

                if number != -1:
                    cv2.putText(frame, f"Number: {number}", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
                else:
                    cv2.putText(frame, "Unknown", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                for i, (name, state) in enumerate(zip(finger_names, states)):
                    cv2.putText(frame, f"{name}: {state}", (10, 150 + i * 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Recognition 0-9", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()