"""
========================================
@FileName:    四、实时识别脚本（修正版）
@Author:      ye_shun
@Email:       2942613675@qq.com
@Description: 修正了缺少导入、颜色空间错误等问题
========================================
"""
import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# -------------------- 模型定义（必须与训练一致）--------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# -------------------- 加载模型 --------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleCNN(num_classes=10).to(device)
model.load_state_dict(torch.load("best_gesture_cnn.pth", map_location=device))
model.eval()

# -------------------- 预处理（与训练完全一致）--------------------
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -------------------- MediaPipe 初始化 --------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils


def pad_to_square(img):
    h, w = img.shape[:2]
    if h == w:
        return img

    size = max(h, w)
    pad_top = (size - h) // 2
    pad_bottom = size - h - pad_top
    pad_left = (size - w) // 2
    pad_right = size - w - pad_left
    return cv2.copyMakeBorder(
        img,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0)
    )

def crop_hand_roi(frame, hand_landmarks, margin=30):
    h, w, _ = frame.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]
    x_min, x_max = max(0, int(min(xs)) - margin), min(w, int(max(xs)) + margin)
    y_min, y_max = max(0, int(min(ys)) - margin), min(h, int(max(ys)) + margin)
    roi = frame[y_min:y_max, x_min:x_max].copy()
    return roi

# -------------------- 实时识别循环 --------------------
cap = cv2.VideoCapture(0)
print("按 'q' 退出，按 'd' 保存当前ROI调试")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)                     # 镜像
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # MediaPipe 需要 RGB
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        # 裁剪 ROI（从原始 BGR 图像中裁剪）
        roi = crop_hand_roi(frame, hand_landmarks, margin=30)
        if roi.size != 0:
            # 关键：将 ROI 从 BGR 转为 RGB，与训练数据一致
            roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            roi_rgb = pad_to_square(roi_rgb)
            input_tensor = preprocess(roi_rgb).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                pred = torch.argmax(output, dim=1).item()
                prob = torch.softmax(output, dim=1).max().item()

            # 显示识别结果
            cv2.putText(frame, f"Number: {pred} ({prob:.2f})", (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            cv2.imshow("ROI", roi)

            # 调试：按 'd' 保存当前 ROI 到磁盘
            key = cv2.waitKey(1) & 0xFF
            if key == ord('d'):
                cv2.imwrite("debug_roi.jpg", roi)
                print("已保存 debug_roi.jpg")
            elif key == ord('q'):
                break

        # 绘制手部关键点
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("CNN Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
