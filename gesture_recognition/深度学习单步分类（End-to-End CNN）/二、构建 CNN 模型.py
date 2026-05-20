"""
========================================
@FileName:    二、构建 CNN 模型
@Author:      ye_shun
@Email:       2942613675@qq.com
@Created:     2026/5/19 15:39
@Description: 
========================================
"""
import torch.nn as nn
import torch.nn.functional as F


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

        # 计算全连接层输入维度: 128 * (128/8)^2 = 128 * 16 * 16 = 32768
        self.fc1 = nn.Linear(128 * 16 * 16, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 64x64
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 32x32
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 16x16
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 也可以使用预训练的 MobileNetV2（更强大但稍重）
# from torchvision import models
# model = models.mobilenet_v2(pretrained=True)
# model.classifier[1] = nn.Linear(model.last_channel, 10)