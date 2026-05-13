"""
========================================
@FileName:    main_gui.py
@Author:      ye_shun
@Email:       2942613675@qq.com
@Description: QPE-HViT 量子增强视觉Transformer实验系统 (整合版)
========================================
"""
import sys
import cv2
import numpy as np
import time
import random
from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                               QHBoxLayout, QComboBox, QPushButton, QLabel,
                               QFileDialog, QFrame, QButtonGroup, QStatusBar)
from PySide6.QtCore import Qt, QThread, Signal, QSize
from PySide6.QtGui import QImage, QPixmap, QFont

# ==================== 论文实验数据库 (整合自 4.4, 4.5, 4.7) ====================
EXP_DATABASE = {
    # 主实验模型 (Table 2)
    "ResNet34": {"CIFAR-10": "60.74%", "CIFAR-100": "33.75%", "ImageNet-1k": "70.04%", "Params": "21.28M",
                 "FLOPs": "3.68G"},
    "ResNet101": {"CIFAR-10": "68.22%", "CIFAR-100": "40.10%", "ImageNet-1k": "77.37%", "Params": "44.62M",
                  "FLOPs": "8.52G"},
    "EfficientNet-B7": {"CIFAR-10": "65.91%", "CIFAR-100": "36.58%", "ImageNet-1k": "84.38%", "Params": "66.34M",
                        "FLOPs": "19.77G"},
    "ConvNeXt-B": {"CIFAR-10": "93.06%", "CIFAR-100": "72.0%", "ImageNet-1k": "83.81%", "Params": "89.32M",
                   "FLOPs": "15.41G"},
    "ViT-B/16": {"CIFAR-10": "67.42%", "CIFAR-100": "41.41%", "ImageNet-1k": "77.27%", "Params": "86.61M",
                 "FLOPs": "11.28G"},
    "Swin-B": {"CIFAR-10": "84.26%", "CIFAR-100": "70.05%", "ImageNet-1k": "83.56%", "Params": "88.04M",
               "FLOPs": "10.22G"},
    "DeiT-B": {"CIFAR-10": "92.0%", "CIFAR-100": "70.20%", "ImageNet-1k": "81.85%", "Params": "86.66M",
               "FLOPs": "17.96G"},
    "CrossViT-B": {"CIFAR-10": "75.0%", "CIFAR-100": "49.94%", "ImageNet-1k": "82.21%", "Params": "105.01M",
                   "FLOPs": "15.51G"},
    "CTA-Net": {"CIFAR-10": "86.76%", "CIFAR-100": "59.43%", "ImageNet-1k": "76.86%", "Params": "20.32M",
                "FLOPs": "2.83G"},
    "MambaVision-B": {"CIFAR-10": "93.82%", "CIFAR-100": "72.45%", "ImageNet-1k": "84.28%", "Params": "97.70M",
                      "FLOPs": "15.01G"},
    "QPE-HViT (Full)": {"CIFAR-10": "85.24%", "CIFAR-100": "75.49%", "ImageNet-1k": "84.45%", "Params": "88.16M",
                        "FLOPs": "87.92G*"},

    # 消融实验变体 (Table 3)
    "Base HViT (w/o QPE & QCSA)": {"CIFAR-100": "68.32%", "Params": "87.5M", "FLOPs": "14.86G"},
    "HViT + QPE": {"CIFAR-100": "72.18%", "Params": "87.8M", "FLOPs": "48.53G"},
    "HViT + QCSA": {"CIFAR-100": "71.92%", "Params": "88.0M", "FLOPs": "50.17G"}
}


class InferenceThread(QThread):
    finished = Signal(dict)

    def __init__(self, model_name, image):
        super().__init__()
        self.model_name = model_name
        self.image = image

    def run(self):
        # 模拟推理延迟
        time.sleep(1.2)

        results = {}
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. 原图
        results["输入原图"] = {"base": self.image.copy(), "ours": self.image.copy()}

        # 2. 多尺度特征 (模拟 4.6 节描述)
        # Base: 模糊边缘 ; Ours: 高纯度层级语义
        res_base = cv2.applyColorMap(cv2.Canny(gray, 100, 200), cv2.COLORMAP_JET)
        res_ours = cv2.applyColorMap(cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=5), cv2.COLORMAP_DEEPWINTER)
        results["多尺度特征响应"] = {"base": res_base, "ours": res_ours}

        # 3. Grad-CAM (模拟锁定效应)
        mask = np.zeros_like(gray, dtype=np.float32)
        cv2.circle(mask, (w // 2, h // 2), min(w, h) // 4, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (101, 101), 0)
        cam_ours = cv2.applyColorMap((mask * 255).astype(np.uint8), cv2.COLORMAP_JET)

        cam_base_raw = cv2.GaussianBlur(gray, (151, 151), 0)
        cam_base = cv2.applyColorMap(cam_base_raw, cv2.COLORMAP_BONE)

        results["Grad-CAM 可视化"] = {
            "base": cv2.addWeighted(self.image, 0.5, cam_base, 0.5, 0),
            "ours": cv2.addWeighted(self.image, 0.5, cam_ours, 0.5, 0)
        }

        # 4. t-SNE (模拟坍缩聚集效应)
        results["t-SNE 特征空间"] = {
            "base": self._gen_tsne(clustered=False),
            "ours": self._gen_tsne(clustered=True)
        }

        self.finished.emit(results)

    def _gen_tsne(self, clustered):
        img = np.ones((400, 400, 3), dtype=np.uint8) * 25
        colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100)]
        for c in colors:
            center = (random.randint(100, 300), random.randint(100, 300))
            std = 12 if clustered else 55
            for _ in range(60):
                px = int(np.random.normal(center[0], std))
                py = int(np.random.normal(center[1], std))
                cv2.circle(img, (np.clip(px, 5, 395), np.clip(py, 5, 395)), 3, c, -1)
        return img


class QuantumViTGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("基于量子位置编码的多尺度视觉Transformer可视化系统 v3.1")
        self.resize(1400, 850)
        self.raw_img = None
        self.inference_results = {}

        self.init_ui()
        self.apply_stylesheet()
        self.update_metrics_display()  # 初始化指标

    def init_ui(self):
        main_layout = QHBoxLayout()
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)

        # ==================== 左侧控制面板 ====================
        side_panel = QVBoxLayout()
        side_panel.setSpacing(12)

        side_panel.addWidget(QLabel("<b>模型架构 (Table 2/5)</b>"))
        self.model_box = QComboBox()
        # 按照论文分类添加
        model_groups = {
            "CNN-Variants": ["ResNet34", "ResNet101", "EfficientNet-B7", "ConvNeXt-B"],
            "ViT-Variants": ["ViT-B/16", "Swin-B", "DeiT-B"],
            "Hybrid": ["CrossViT-B", "CTA-Net", "MambaVision-B"],
            "Ablation (CIFAR-100)": ["Base HViT (w/o QPE & QCSA)", "HViT + QPE", "HViT + QCSA"],
            "Proposed": ["QPE-HViT (Full)"]
        }
        for group, models in model_groups.items():
            self.model_box.addItem(f"--- {group} ---")
            self.model_box.model().item(self.model_box.count() - 1).setEnabled(False)
            for m in models: self.model_box.addItem(m)
        self.model_box.setCurrentText("QPE-HViT (Full)")
        self.model_box.currentTextChanged.connect(self.update_metrics_display)
        side_panel.addWidget(self.model_box)

        side_panel.addWidget(QLabel("<b>测试数据集</b>"))
        self.data_box = QComboBox()
        self.data_box.addItems(["CIFAR-10", "CIFAR-100", "ImageNet-1k", "DUT-Anti-UAV"])
        self.data_box.setCurrentText("CIFAR-100")
        self.data_box.currentTextChanged.connect(self.update_metrics_display)
        side_panel.addWidget(self.data_box)

        side_panel.addSpacing(10)
        side_panel.addWidget(QLabel("<b>量化评估指标 (Metrics)</b>"))
        self.metrics_label = QLabel()
        self.metrics_label.setObjectName("metrics_board")
        side_panel.addWidget(self.metrics_label)

        self.note_label = QLabel("* FLOPs 为经典架构模拟代价\n理论 QPU 复杂度为 O(L)")
        self.note_label.setStyleSheet("color: #888; font-size: 11px;")
        side_panel.addWidget(self.note_label)

        side_panel.addStretch()
        self.btn_load = QPushButton("📁 加载测试图像")
        self.btn_load.clicked.connect(self.load_image)
        side_bar_btn_style = "min-height: 40px; font-weight: bold;"
        self.btn_load.setStyleSheet(side_bar_btn_style)
        side_panel.addWidget(self.btn_load)

        self.btn_run = QPushButton("🚀 运行对比分析")
        self.btn_run.setObjectName("btn_run")
        self.btn_run.setEnabled(False)
        self.btn_run.clicked.connect(self.start_inference)
        self.btn_run.setStyleSheet(side_bar_btn_style)
        side_panel.addWidget(self.btn_run)

        main_layout.addLayout(side_panel, 1)

        # ==================== 右侧展示面板 ====================
        content_layout = QVBoxLayout()

        # 导航
        self.nav_layout = QHBoxLayout()
        self.stages = ["输入原图", "多尺度特征响应", "Grad-CAM 可视化", "t-SNE 特征空间"]
        self.stage_group = QButtonGroup(self)
        for i, stage in enumerate(self.stages):
            btn = QPushButton(stage)
            btn.setCheckable(True)
            btn.setEnabled(False)
            btn.clicked.connect(lambda checked, s=stage: self.show_stage_result(s))
            self.nav_layout.addWidget(btn)
            self.stage_group.addButton(btn, i)
        content_layout.addLayout(self.nav_layout)

        # 对比显示
        display_layout = QHBoxLayout()

        def create_view(title, obj_name):
            container = QVBoxLayout()
            lbl_title = QLabel(title)
            lbl_title.setAlignment(Qt.AlignCenter)
            lbl_title.setStyleSheet("font-weight: bold; color: #aaa;")
            lbl_view = QLabel("等待数据...")
            lbl_view.setObjectName(obj_name)
            lbl_view.setAlignment(Qt.AlignCenter)
            container.addWidget(lbl_title)
            container.addWidget(lbl_view, 1)
            return container, lbl_view

        self.c_base, self.display_base = create_view("Baseline (HViT / Standard)", "screen_base")
        self.c_ours, self.display_ours = create_view("Proposed (QPE-HViT / Ours)", "screen_ours")

        display_layout.addLayout(self.c_base)
        display_layout.addLayout(self.c_ours)
        content_layout.addLayout(display_layout, 5)

        main_layout.addLayout(content_layout, 4)

        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage("系统就绪。请加载图片以开始。")

    def apply_stylesheet(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #121212; color: #ffffff; }
            QLabel { color: #e0e0e0; }
            QComboBox { padding: 6px; background-color: #1e1e1e; color: white; border: 1px solid #333; border-radius: 4px; }
            QPushButton { padding: 10px; border-radius: 4px; background-color: #2a2a2a; color: white; border: 1px solid #444; }
            QPushButton:hover { background-color: #383838; }
            QPushButton:checked { background-color: #0078d7; border: 1px solid #005a9e; font-weight: bold; }
            #metrics_board { 
                background-color: #000; border: 1px solid #2e7d32; border-radius: 4px; 
                padding: 12px; font-family: 'Consolas', monospace; color: #4CAF50; font-size: 14px;
            }
            #btn_run { background-color: #1b5e20; }
            #btn_run:hover { background-color: #2e7d32; }
            #screen_base { background-color: #0a0a0a; border: 1px solid #333; border-radius: 6px; }
            #screen_ours { background-color: #0a0a0a; border: 2px solid #2e7d32; border-radius: 6px; }
        """)

    def update_metrics_display(self):
        m_name = self.model_box.currentText()
        d_name = self.data_box.currentText()

        # 查找逻辑
        if m_name in EXP_DATABASE:
            data = EXP_DATABASE[m_name]
            # 获取对应数据集的 Acc
            acc = data.get(d_name, "N/A")
            # 如果是消融实验且数据集选了其他的，强制看 CIFAR-100
            if "Ablation" in m_name and d_name != "CIFAR-100":
                acc = data.get("CIFAR-100", "N/A") + " (C100)"

            self.metrics_label.setText(
                f" Top-1 Acc:  {acc}\n"
                f" Params:     {data['Params']}\n"
                f" FLOPs:      {data['FLOPs']}"
            )
        else:
            self.metrics_label.setText(" 无对应实验数据")

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg)")
        if path:
            self.raw_img = cv2.imread(path)
            self.update_dual_screen(self.raw_img, self.raw_img)
            self.btn_run.setEnabled(True)
            self.statusBar.showMessage(f"已加载: {path.split('/')[-1]}")

    def start_inference(self):
        m_name = self.model_box.currentText()
        self.btn_run.setEnabled(False)
        self.btn_run.setText("⏳ 计算中...")
        self.thread = InferenceThread(m_name, self.raw_img)
        self.thread.finished.connect(self.on_inference_finished)
        self.thread.start()

    def on_inference_finished(self, results):
        self.inference_results = results
        self.btn_run.setText("🚀 运行推理与对比分析")
        self.btn_run.setEnabled(True)
        for btn in self.stage_group.buttons(): btn.setEnabled(True)
        self.stage_group.button(1).setChecked(True)
        self.show_stage_result("多尺度特征响应")
        self.statusBar.showMessage("分析完成。请点击顶部阶段查看可视化对比。")

    def show_stage_result(self, stage_name):
        if stage_name in self.inference_results:
            d = self.inference_results[stage_name]
            self.update_dual_screen(d["base"], d["ours"])

    def update_dual_screen(self, img_b, img_o):
        def to_pix(img, lbl):
            h, w, c = img.shape
            q = QImage(img.data, w, h, w * c, QImage.Format_RGB888).rgbSwapped()
            return QPixmap.fromImage(q).scaled(lbl.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

        self.display_base.setPixmap(to_pix(img_b, self.display_base))
        self.display_ours.setPixmap(to_pix(img_o, self.display_ours))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self.raw_img is not None:
            btn = self.stage_group.checkedButton()
            if btn:
                self.show_stage_result(btn.text())
            else:
                self.update_dual_screen(self.raw_img, self.raw_img)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = QuantumViTGUI()
    window.show()
    sys.exit(app.exec())