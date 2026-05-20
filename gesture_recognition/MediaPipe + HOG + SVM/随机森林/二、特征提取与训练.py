"""
========================================
@FileName:    二、特征提取与训练
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 15:23
@Description: 
========================================
"""
import cv2
import numpy as np
import os
from skimage.feature import hog
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

def load_data(data_dir):
    X = []
    y = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg"):
            # 文件名格式: label_count.jpg
            label = int(filename.split('_')[0])
            img_path = os.path.join(data_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            # 统一缩放为 128x128（HOG 对尺寸敏感，建议固定大小）
            img_resized = cv2.resize(img, (128, 128))
            # 提取 HOG 特征
            features = hog(img_resized, orientations=9, pixels_per_cell=(8,8),
                           cells_per_block=(2,2), block_norm='L2-Hys', visualize=False)
            X.append(features)
            y.append(label)
    return np.array(X), np.array(y)

# 加载数据
X, y = load_data("gesture_data")
print(f"Total samples: {len(X)}, Feature dimension: {X.shape[1]}")
# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 训练 SVM (RBF 核)
svm_model = svm.SVC(kernel='rbf', gamma='scale', C=1.0, probability=True)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)
acc_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM Accuracy: {acc_svm:.4f}")
print(classification_report(y_test, y_pred_svm))

# 训练随机森林
rf_model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf))

# 保存模型（两种都保存，任选其一）
with open("svm_gesture_model.pkl", "wb") as f:
    pickle.dump(svm_model, f)
with open("rf_gesture_model.pkl", "wb") as f:
    pickle.dump(rf_model, f)