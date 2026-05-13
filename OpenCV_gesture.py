import cv2
import mediapipe as mp
import numpy as np
from collections import deque

# ===================== MediaPipe 初始化 =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ===================== 手势识别 =====================
class GestureRecognizer:
    def __init__(self):
        self.history = deque(maxlen=10)

    def finger_status(self, lm):
        fingers = []

        # 拇指（x方向判断）
        fingers.append(lm[4][0] > lm[3][0])

        # 其余四指（y方向）
        tips = [8, 12, 16, 20]
        dips = [6, 10, 14, 18]

        for tip, dip in zip(tips, dips):
            fingers.append(lm[tip][1] < lm[dip][1])

        return fingers  # [thumb, index, middle, ring, pinky]

    def recognize(self, lm):
        fingers = self.finger_status(lm)
        count = fingers.count(True)

        # ================= 手势规则 =================
        if count == 0:
            gesture = 0

        elif count == 1:
            gesture = 1

        elif count == 2:
            # ✌️
            if fingers[1] and fingers[2]:
                gesture = 2
            else:
                gesture = 2

        elif count == 3:
            gesture = 3

        elif count == 4:
            gesture = 4

        elif count == 5:
            gesture = 5

        else:
            gesture = -1

        # ================= 平滑 =================
        self.history.append(gesture)
        return int(np.median(self.history))


# ===================== 主程序 =====================
def main():
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    gr = GestureRecognizer()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        gesture = -1

        if result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                h, w, c = frame.shape

                lm_list = []
                for id, lm in enumerate(handLms.landmark):
                    lm_list.append((int(lm.x * w), int(lm.y * h)))

                gesture = gr.recognize(lm_list)

                # 画关键点
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        # ================= UI =================
        cv2.rectangle(frame, (0,0), (640,80), (30,30,30), -1)

        if gesture != -1:
            cv2.putText(frame, f"Number: {gesture}",
                        (20,60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.8, (0,255,0), 3)
        else:
            cv2.putText(frame, "Show hand gesture",
                        (20,60), cv2.FONT_HERSHEY_SIMPLEX,
                        1.2, (0,200,255), 2)

        cv2.imshow("MediaPipe Gesture Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()