"""
QPE-HViT 毕设可视化展示系统主界面。
"""
from collections import deque
from functools import partial
import json
import math
from pathlib import Path
import sys

try:
    import cv2
    import numpy as np
except ImportError:
    cv2 = None
    np = None

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor, QFont, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from experiment_data import (
    ABLATION_DATA,
    APP_META,
    PERFORMANCE_DATA,
    THEORY_DATA,
    VISUALIZATION_DATA,
)


class AdaptiveImageLabel(QLabel):
    def __init__(self, placeholder="等待载入图片"):
        super().__init__(placeholder)
        self._placeholder = placeholder
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignCenter)
        self.setWordWrap(True)
        self.setObjectName("imageViewport")
        self.setMinimumHeight(240)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_placeholder(self, text):
        self._placeholder = text
        if self._source_pixmap.isNull():
            self.setPixmap(QPixmap())
            self.setText(text)

    def set_image(self, pixmap):
        self._source_pixmap = pixmap
        self._update_scaled_pixmap()

    def clear_image(self):
        self._source_pixmap = QPixmap()
        self.setPixmap(QPixmap())
        self.setText(self._placeholder)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._update_scaled_pixmap()

    def _update_scaled_pixmap(self):
        if self._source_pixmap.isNull():
            return
        target_size = self.contentsRect().size()
        if target_size.width() <= 8 or target_size.height() <= 8:
            return
        scaled = self._source_pixmap.scaled(
            target_size,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.setText("")
        self.setPixmap(scaled)


class ImageCard(QFrame):
    def __init__(self, title, accent=False):
        super().__init__()
        self.default_path = ""
        self.custom_path = ""
        self._full_frame_mode = False

        self.setObjectName("imageCardAccent" if accent else "imageCard")
        self.card_layout = QVBoxLayout(self)
        self.card_layout.setContentsMargins(18, 18, 18, 18)
        self.card_layout.setSpacing(12)

        self.title_label = QLabel(title)
        self.title_label.setObjectName("sectionCardTitle")
        self.title_label.setAlignment(Qt.AlignCenter)
        self.title_label.setWordWrap(True)
        self.card_layout.addWidget(self.title_label)

        self.viewport_frame = QFrame()
        self.viewport_frame.setObjectName("imageViewportFrame")
        self.viewport_layout = QVBoxLayout(self.viewport_frame)
        self.viewport_layout.setContentsMargins(12, 12, 12, 12)
        self.viewport_layout.setSpacing(0)

        self.image_label = AdaptiveImageLabel("等待载入图片")
        self.viewport_layout.addWidget(self.image_label)
        self.card_layout.addWidget(self.viewport_frame, 1)

        self.path_label = QLabel("默认路径：未设置")
        self.path_label.setWordWrap(True)
        self.path_label.setObjectName("mutedText")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.card_layout.addWidget(self.path_label)

        self.caption_label = QLabel("")
        self.caption_label.setWordWrap(True)
        self.caption_label.setObjectName("mutedText")
        self.card_layout.addWidget(self.caption_label)

        self.upload_button = QPushButton("选择演示图片")
        self.card_layout.addWidget(self.upload_button)

    def set_caption(self, text):
        self.caption_label.setText(text)

    def set_default_path(self, path):
        self.default_path = path or ""
        self._refresh_path_label()
        self.reload_image()

    def set_custom_path(self, path):
        self.custom_path = path or ""
        self._refresh_path_label()
        self.reload_image()

    def current_image_path(self):
        if self.custom_path and Path(self.custom_path).exists():
            return self.custom_path
        if self.default_path and Path(self.default_path).exists():
            return self.default_path
        return ""

    def reload_image(self):
        image_path = self.current_image_path()
        if not image_path:
            if self.default_path:
                self.image_label.set_placeholder(
                    "未找到图片\n\n"
                    f"请将图片放入：\n{self.default_path}\n\n"
                    "或点击下方按钮临时载入。"
                )
            else:
                self.image_label.set_placeholder("当前模块尚未设置默认图片路径。")
            self.image_label.clear_image()
            self._sync_full_frame_geometry()
            return

        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.set_placeholder(f"图片读取失败：\n{image_path}")
            self.image_label.clear_image()
            self._sync_full_frame_geometry()
            return
        self.image_label.set_image(pixmap)
        self._sync_full_frame_geometry()

    def _refresh_path_label(self):
        if self.custom_path:
            self.path_label.setText(
                f"当前图片：{self.custom_path}\n默认路径：{self.default_path or '未设置'}"
            )
        else:
            self.path_label.setText(f"默认路径：{self.default_path or '未设置'}")
 
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._sync_full_frame_geometry()

    def set_full_frame_mode(self):
        self._full_frame_mode = True
        self.card_layout.setContentsMargins(0, 0, 0, 0)
        self.card_layout.setSpacing(0)
        self.viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.viewport_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.title_label.hide()
        self.path_label.hide()
        self.caption_label.hide()
        self.upload_button.hide()
        self.image_label.setMinimumHeight(240)
        self.image_label.setStyleSheet("padding: 0px;")
        self._sync_full_frame_geometry()

    def set_titled_frame_mode(self):
        self._full_frame_mode = True
        self.card_layout.setContentsMargins(18, 18, 18, 18)
        self.card_layout.setSpacing(12)
        self.viewport_layout.setContentsMargins(0, 0, 0, 0)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.viewport_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.title_label.show()
        self.path_label.hide()
        self.caption_label.hide()
        self.upload_button.hide()
        self.image_label.setMinimumHeight(240)
        self.image_label.setStyleSheet("padding: 0px;")
        self._sync_full_frame_geometry()

    def _sync_full_frame_geometry(self):
        if not self._full_frame_mode:
            return

        pixmap = self.image_label._source_pixmap
        if pixmap.isNull() or pixmap.width() <= 0:
            self.viewport_frame.setMinimumHeight(240)
            self.viewport_frame.setMaximumHeight(16777215)
            self.image_label.setMinimumHeight(240)
            self.image_label.setMaximumHeight(16777215)
            return

        target_width = self.viewport_frame.contentsRect().width()
        if target_width <= 8:
            target_width = self.contentsRect().width()
        if target_width <= 8:
            return

        target_height = max(1, round(target_width * pixmap.height() / pixmap.width()))
        if self.viewport_frame.minimumHeight() != target_height or self.viewport_frame.maximumHeight() != target_height:
            self.viewport_frame.setFixedHeight(target_height)
        if self.image_label.minimumHeight() != target_height or self.image_label.maximumHeight() != target_height:
            self.image_label.setFixedHeight(target_height)
 
 
if torch is not None:
    class GestureCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(0.25),
                nn.Linear(128, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))
else:
    GestureCNN = None


class OpenCVGestureRecognition:
    def __init__(self):
        self.history = deque(maxlen=10)

    def reset(self):
        self.history.clear()

    def distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    def build_skin_mask(self, roi):
        blur = cv2.GaussianBlur(roi, (5, 5), 0)

        ycrcb = cv2.cvtColor(blur, cv2.COLOR_BGR2YCrCb)
        mask1 = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135))

        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        mask2 = cv2.inRange(hsv, (0, 30, 60), (20, 150, 255))

        mask = cv2.bitwise_and(mask1, mask2)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return cv2.medianBlur(mask, 5)

    def recognize(self, roi):
        mask = self.build_skin_mask(roi)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return -1, mask

        cnt = max(contours, key=cv2.contourArea)
        if cv2.contourArea(cnt) < 3000:
            return -1, mask

        hull = cv2.convexHull(cnt, returnPoints=False)
        if hull is None or len(hull) < 3:
            return 0, mask

        defects = cv2.convexityDefects(cnt, hull)
        if defects is None:
            return 0, mask

        finger_count = 0
        for item in defects[:, 0]:
            s, e, f, d = item
            start = tuple(cnt[s][0])
            end = tuple(cnt[e][0])
            far = tuple(cnt[f][0])

            a = self.distance(start, end)
            b = self.distance(start, far)
            c = self.distance(end, far)
            denominator = 2 * b * c + 1e-5
            cosine = max(-1.0, min(1.0, (b * b + c * c - a * a) / denominator))
            angle = math.degrees(math.acos(cosine))

            if angle < 70 and d > 10000:
                finger_count += 1

        finger_count = min(finger_count + 1, 5)
        self.history.append(finger_count)
        result = int(np.median(self.history))
        return result, mask


class QuantumViTGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_META["window_title"])
        self.setWindowFlags(
            Qt.Window
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(1180, 720)
        self.resize(1520, 940)

        self.nav_buttons = {}
        self.page_index = {}
        self.visual_image_cards = {}
        self.visual_custom_paths = {}
        self.home_module_button_keys = {}
        self.camera_capture = None
        self.camera_timer = QTimer(self)
        self.camera_timer.timeout.connect(self.update_gesture_camera_frame)
        self.latest_camera_frame = None
        self.gesture_frame_counter = 0
        self.gesture_model = None
        self.gesture_model_path = Path(__file__).resolve().parent / "gesture_digit_cnn.pt"
        self.gesture_model_input_size = 64
        self.gesture_model_classes = [str(index) for index in range(10)]
        self.gesture_prediction_history = deque(maxlen=3)
        self.opencv_gesture_recognizer = OpenCVGestureRecognition() if cv2 is not None and np is not None else None

        self.init_ui()
        self.apply_stylesheet()
        self.load_gesture_model()
        self.populate_home_page()
        self.populate_theory_page()
        self.populate_performance_page()
        self.populate_ablation_page()
        self.populate_visualization_page()
        self.switch_page("home")

    def init_ui(self):
        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)
        self.setCentralWidget(root)

        root_layout.addWidget(self.build_sidebar())

        self.stack = QStackedWidget()
        root_layout.addWidget(self.stack, 1)

        self.add_page("home", self.build_home_page())
        self.add_page("theory", self.build_theory_page())
        self.add_page("performance", self.build_performance_page())
        self.add_page("ablation", self.build_ablation_page())
        self.add_page("visualization", self.build_visualization_page())
        self.add_page("gesture", self.build_gesture_page())

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        status_bar.showMessage("系统已就绪，可直接进入理论讲解或实验结果展示。")

    def build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(300)

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(16)

        brand_block = QFrame()
        brand_block.setObjectName("brandBlock")
        brand_layout = QVBoxLayout(brand_block)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(10)

        badge = QLabel("DEFENSE MODE")
        badge.setObjectName("sideBadge")
        brand_layout.addWidget(badge, 0, Qt.AlignLeft)

        logo = QLabel("Q")
        logo.setObjectName("brandLogo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(52, 52)
        brand_layout.addWidget(logo, 0, Qt.AlignLeft)

        title = QLabel("QPE-HViT")
        title.setObjectName("brandTitle")
        brand_layout.addWidget(title)

        subtitle = QLabel("本科毕业设计可视化演示系统")
        subtitle.setObjectName("brandSubtitle")
        subtitle.setWordWrap(True)
        brand_layout.addWidget(subtitle)

        layout.addWidget(brand_block)

        divider = QFrame()
        divider.setFrameShape(QFrame.HLine)
        divider.setObjectName("divider")
        layout.addWidget(divider)

        nav_title = QLabel("模块导航")
        nav_title.setObjectName("navSectionLabel")
        layout.addWidget(nav_title)

        nav_items = [
            ("home", "系统首页"),
            ("theory", "理论方法"),
            ("performance", "第一组：性能对比"),
            ("ablation", "第二组：消融实验"),
            ("visualization", "第三组：可视化实验"),
            ("gesture", "手势数字识别"),
        ]
        for key, text in nav_items:
            button = QPushButton(text)
            button.setCheckable(True)
            button.setObjectName("navButton")
            button.clicked.connect(partial(self.switch_page, key))
            self.nav_buttons[key] = button
            layout.addWidget(button)

        layout.addStretch()

        footer_card = QFrame()
        footer_card.setObjectName("sidebarFooterCard")
        footer_layout = QVBoxLayout(footer_card)
        footer_layout.setContentsMargins(16, 16, 16, 16)
        footer_layout.setSpacing(8)

        footer_title = QLabel("素材说明")
        footer_title.setObjectName("sidebarFooterTitle")
        footer_layout.addWidget(footer_title)

        footer = QLabel(
            "数据统一维护在 experiment_data.py。\n"
            "实验图片放在 assets/visualizations，理论框图放在 assets/presentation/theory。"
        )
        footer.setObjectName("sidebarHint")
        footer.setWordWrap(True)
        footer_layout.addWidget(footer)
        layout.addWidget(footer_card)
        return sidebar

    def add_page(self, key, widget):
        self.page_index[key] = self.stack.addWidget(widget)

    def switch_page(self, key):
        self.stack.setCurrentIndex(self.page_index[key])
        for name, button in self.nav_buttons.items():
            button.setChecked(name == key)

        names = {
            "home": "系统首页",
            "theory": "理论方法",
            "performance": "性能对比实验组",
            "ablation": "消融实验组",
            "visualization": "可视化实验组",
            "gesture": "手势数字识别",
        }
        self.statusBar().showMessage(f"当前页面：{names[key]}")

    def build_scroll_page(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(30, 28, 30, 28)
        layout.setSpacing(18)
        scroll.setWidget(container)
        return scroll, layout

    def build_home_page(self):
        scroll, layout = self.build_scroll_page()
        layout.setContentsMargins(38, 34, 38, 22)
        layout.setSpacing(18)

        hero_layout = QHBoxLayout()
        hero_layout.setSpacing(24)

        hero_text = QFrame()
        hero_text.setObjectName("homeHeroText")
        hero_text_layout = QVBoxLayout(hero_text)
        hero_text_layout.setContentsMargins(0, 16, 0, 10)
        hero_text_layout.setSpacing(12)

        hero_kicker = QLabel("QPE-HVIT VISUAL SYSTEM")
        hero_kicker.setObjectName("homeHeroKicker")
        hero_text_layout.addWidget(hero_kicker)

        hero_title = QLabel(
            "基于量子位置编码的多尺度视觉 "
            "<span style='color:#2f75f6;'>Transformer</span> 可视化系统"
        )
        hero_title.setObjectName("homeHeroTitle")
        hero_title.setTextFormat(Qt.RichText)
        hero_title.setWordWrap(True)
        hero_text_layout.addWidget(hero_title)

        hero_subtitle = QLabel(
            "系统集成理论、实验与可视化展示，支持多种经典方法对比、性能分析与实验结果可视化，"
            "助力深入理解 QPE-HViT 模型设计与性能优势。"
        )
        hero_subtitle.setObjectName("homeHeroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_text_layout.addWidget(hero_subtitle)

        project_brief = QLabel(
            "项目简介：本系统围绕 QPE-HViT 模型的结构设计、实验验证与可解释性分析展开，"
            "将论文中的关键方法与实验结论组织为适合答辩展示的交互式界面。"
        )
        project_brief.setObjectName("homeProjectBrief")
        project_brief.setWordWrap(True)
        hero_text_layout.addWidget(project_brief)

        hero_meta = QHBoxLayout()
        hero_meta.setSpacing(10)
        self.home_hero_meta_labels = []
        for _ in range(3):
            label = QLabel()
            label.setObjectName("heroMetaPill")
            hero_meta.addWidget(label, 0, Qt.AlignLeft)
            self.home_hero_meta_labels.append(label)
        hero_meta.addStretch()
        hero_text_layout.addLayout(hero_meta)
        hero_layout.addWidget(hero_text, 1)

        hero_visual = QFrame()
        hero_visual.setObjectName("homeHeroArt")
        hero_visual_layout = QVBoxLayout(hero_visual)
        hero_visual_layout.setContentsMargins(8, 0, 0, 0)
        hero_visual_layout.setSpacing(8)

        hero_visual_tag = QLabel("QPE-HViT STRUCTURE")
        hero_visual_tag.setObjectName("homeVisualTag")
        hero_visual_layout.addWidget(hero_visual_tag, 0, Qt.AlignRight)

        hero_stack = QFrame()
        hero_stack.setObjectName("homeVisualStack")
        stack_layout = QVBoxLayout(hero_stack)
        stack_layout.setContentsMargins(30, 24, 30, 24)
        stack_layout.setSpacing(9)

        stack_top = QLabel("QPE")
        stack_top.setObjectName("homeStackTop")
        stack_top.setAlignment(Qt.AlignCenter)
        stack_top.setFixedHeight(34)
        stack_layout.addWidget(stack_top)

        stack_mid = QLabel("QCSA")
        stack_mid.setObjectName("homeStackMid")
        stack_mid.setAlignment(Qt.AlignCenter)
        stack_mid.setFixedHeight(42)
        stack_layout.addWidget(stack_mid)

        stack_base = QLabel("HViT")
        stack_base.setObjectName("homeStackBase")
        stack_base.setAlignment(Qt.AlignCenter)
        stack_base.setFixedHeight(46)
        stack_layout.addWidget(stack_base)

        hero_visual_layout.addWidget(hero_stack, 0, Qt.AlignRight)

        hero_visual_note = QLabel("理论方法、性能对比、消融实验与可解释性展示统一入口")
        hero_visual_note.setObjectName("homeVisualNote")
        hero_visual_note.setWordWrap(True)
        hero_visual_note.setAlignment(Qt.AlignRight)
        hero_visual_layout.addWidget(hero_visual_note, 0, Qt.AlignRight)
        hero_layout.addWidget(hero_visual, 0)

        layout.addLayout(hero_layout)

        stats_layout = QGridLayout()
        stats_layout.setHorizontalSpacing(16)
        stats_layout.setVerticalSpacing(16)
        self.home_stat_cards = []
        for index in range(4):
            metric = self.create_home_metric_card(index)
            self.home_stat_cards.append(metric)
            stats_layout.addWidget(metric["card"], 0, index)
        layout.addLayout(stats_layout)

        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(18)

        roadmap_card = QFrame()
        roadmap_card.setObjectName("contentCard")
        roadmap_layout = QVBoxLayout(roadmap_card)
        roadmap_layout.setContentsMargins(24, 22, 24, 22)
        roadmap_layout.setSpacing(12)

        roadmap_title = QLabel("答辩展示路径")
        roadmap_title.setObjectName("sectionCardTitle")
        roadmap_layout.addWidget(roadmap_title)

        roadmap_hint = QLabel("建议按方法来源、定量结果、模块贡献、可解释性证据的顺序展开讲解。")
        roadmap_hint.setObjectName("mutedText")
        roadmap_hint.setWordWrap(True)
        roadmap_layout.addWidget(roadmap_hint)

        self.home_timeline_items = []
        for index in range(5):
            row = self.create_home_timeline_item(index + 1)
            self.home_timeline_items.append(row)
            roadmap_layout.addWidget(row["card"])
        middle_layout.addWidget(roadmap_card, 5)

        highlights_card = QFrame()
        highlights_card.setObjectName("contentCard")
        highlights_layout = QVBoxLayout(highlights_card)
        highlights_layout.setContentsMargins(24, 22, 24, 22)
        highlights_layout.setSpacing(14)

        highlights_title = QLabel("核心亮点")
        highlights_title.setObjectName("sectionCardTitle")
        highlights_layout.addWidget(highlights_title)

        self.home_highlight_items = []
        for _ in range(4):
            item = self.create_home_highlight_item()
            highlights_layout.addWidget(item["card"])
            self.home_highlight_items.append(item)
        highlights_layout.addStretch()
        middle_layout.addWidget(highlights_card, 3)

        layout.addLayout(middle_layout)

        section = QLabel("模块导航")
        section.setObjectName("pageTitle")
        layout.addWidget(section)

        cards_layout = QGridLayout()
        cards_layout.setHorizontalSpacing(16)
        cards_layout.setVerticalSpacing(16)
        self.home_module_cards = []
        module_tones = ["Blue", "Green", "Indigo", "Amber"]
        for index in range(4):
            tone = module_tones[index % len(module_tones)]
            card = QFrame()
            card.setObjectName("homeModuleCard")
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(22, 22, 22, 22)
            card_layout.setSpacing(12)

            top_row = QHBoxLayout()
            top_row.setSpacing(12)

            icon_label = QLabel()
            icon_label.setObjectName(f"homeModuleIcon{tone}")
            icon_label.setAlignment(Qt.AlignCenter)
            icon_label.setFixedSize(56, 56)
            top_row.addWidget(icon_label, 0, Qt.AlignTop)

            title_box = QVBoxLayout()
            title_box.setSpacing(6)

            eyebrow = QLabel()
            eyebrow.setObjectName("homeModuleTag")
            title_box.addWidget(eyebrow, 0, Qt.AlignLeft)

            title_label = QLabel()
            title_label.setObjectName("homeModuleTitle")
            title_label.setWordWrap(True)
            title_box.addWidget(title_label)
            top_row.addLayout(title_box, 1)

            card_layout.addLayout(top_row)

            desc_label = QLabel()
            desc_label.setWordWrap(True)
            desc_label.setObjectName("bodyText")
            card_layout.addWidget(desc_label, 1)

            button = QPushButton("进入模块")
            button.setObjectName(f"homeModuleButton{tone}")
            card_layout.addWidget(button, 0, Qt.AlignLeft)

            self.home_module_cards.append((icon_label, eyebrow, title_label, desc_label, button))
            cards_layout.addWidget(card, 0, index)

        layout.addLayout(cards_layout)
        footer = QFrame()
        footer.setObjectName("homeFooter")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(2, 6, 2, 0)
        footer_layout.setSpacing(10)

        footer_left = QLabel("QPE-HViT 可视化系统")
        footer_left.setObjectName("homeFooterText")
        footer_mid = QLabel("2024 毕业设计演示系统  |  仅供学术交流使用")
        footer_mid.setObjectName("homeFooterText")
        footer_right = QLabel("希望本系统能为你的答辩与展示助力")
        footer_right.setObjectName("homeFooterText")

        footer_layout.addWidget(footer_left)
        footer_layout.addStretch()
        footer_layout.addWidget(footer_mid)
        footer_layout.addStretch()
        footer_layout.addWidget(footer_right)
        layout.addWidget(footer)
        layout.addStretch()
        return scroll

    def build_theory_page(self):
        scroll, layout = self.build_scroll_page()
        layout.setContentsMargins(38, 34, 38, 26)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("theoryHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 26, 28, 26)
        hero_layout.setSpacing(24)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(10)

        hero_tag = QLabel("THEORY METHOD")
        hero_tag.setObjectName("tagPill")
        hero_text.addWidget(hero_tag, 0, Qt.AlignLeft)

        hero_title = QLabel("理论方法与模型框图")
        hero_title.setObjectName("heroTitle")
        hero_title.setWordWrap(True)
        hero_text.addWidget(hero_title)

        hero_subtitle = QLabel(
            "本页用于答辩中讲清 QPE-HViT 的方法来源、核心模块与整体推理链路，"
            "先建立模型设计逻辑，再对应到论文框图与模块细节。"
        )
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_text.addWidget(hero_subtitle)

        hero_meta = QHBoxLayout()
        hero_meta.setSpacing(10)
        for text in ("分层视觉 Transformer", "量子位置先验", "通道-空间协同注意"):
            label = QLabel(text)
            label.setObjectName("theoryMetaPill")
            hero_meta.addWidget(label, 0, Qt.AlignLeft)
        hero_meta.addStretch()
        hero_text.addLayout(hero_meta)
        hero_layout.addLayout(hero_text, 1)

        hero_model = QFrame()
        hero_model.setObjectName("theoryHeroModel")
        model_layout = QVBoxLayout(hero_model)
        model_layout.setContentsMargins(22, 20, 22, 20)
        model_layout.setSpacing(10)
        for name, object_name in (
            ("Input / Patch", "theoryModelSoft"),
            ("HViT Encoder", "theoryModelPrimary"),
            ("QPE + QCSA", "theoryModelAccent"),
            ("Fusion Output", "theoryModelSoft"),
        ):
            node = QLabel(name)
            node.setObjectName(object_name)
            node.setAlignment(Qt.AlignCenter)
            node.setFixedHeight(36)
            model_layout.addWidget(node)
        hero_layout.addWidget(hero_model, 0, Qt.AlignRight)
        layout.addWidget(hero)

        summary_layout = QHBoxLayout()
        summary_layout.setSpacing(16)
        for index, item in enumerate(THEORY_DATA["innovation_cards"], start=1):
            summary_layout.addWidget(
                self.create_theory_innovation_card(
                    index,
                    item["title"],
                    item["description"],
                    accent=item.get("accent", False),
                ),
                1,
            )
        layout.addLayout(summary_layout)

        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(18)

        overview_card = QFrame()
        overview_card.setObjectName("contentCard")
        overview_layout = QVBoxLayout(overview_card)
        overview_layout.setContentsMargins(24, 22, 24, 22)
        overview_layout.setSpacing(12)

        overview_title = QLabel("方法概述")
        overview_title.setObjectName("sectionCardTitle")
        overview_layout.addWidget(overview_title)

        for index, line in enumerate(THEORY_DATA["overview"], start=1):
            overview_layout.addWidget(self.create_theory_point_card(index, line))
        middle_layout.addWidget(overview_card, 4)

        guide_card = QFrame()
        guide_card.setObjectName("contentCard")
        guide_layout = QVBoxLayout(guide_card)
        guide_layout.setContentsMargins(24, 22, 24, 22)
        guide_layout.setSpacing(12)

        guide_title = QLabel("答辩讲解顺序")
        guide_title.setObjectName("sectionCardTitle")
        guide_layout.addWidget(guide_title)
        for index, text in enumerate(THEORY_DATA["talk_track"], start=1):
            guide_layout.addWidget(self.create_theory_talk_item(index, text))
        guide_layout.addWidget(
            self.create_theory_talk_item(
                3,
                "推荐顺序：先说明层次化 HViT 主干，再强调 QPE 与 QCSA 的增强作用，最后落到实验结果与可解释性证据。",
            )
        )
        guide_layout.addStretch()
        middle_layout.addWidget(guide_card, 2)
        layout.addLayout(middle_layout)

        flow_card = QFrame()
        flow_card.setObjectName("contentCard")
        flow_layout = QVBoxLayout(flow_card)
        flow_layout.setContentsMargins(24, 22, 24, 22)
        flow_layout.setSpacing(14)

        flow_title = QLabel("整体推理流程")
        flow_title.setObjectName("sectionCardTitle")
        flow_layout.addWidget(flow_title)

        flow_subtitle = QLabel("从输入图像到分类预测，QPE 与 QCSA 在编码阶段注入位置先验与通道-空间协同注意。")
        flow_subtitle.setObjectName("mutedText")
        flow_subtitle.setWordWrap(True)
        flow_layout.addWidget(flow_subtitle)

        pipeline_layout = QHBoxLayout()
        pipeline_layout.setSpacing(10)
        for index, item in enumerate(THEORY_DATA["pipeline"], start=1):
            pipeline_layout.addWidget(self.create_theory_pipeline_step(index, item), 1)
            if index < len(THEORY_DATA["pipeline"]):
                arrow = QLabel("→")
                arrow.setObjectName("theoryFlowArrow")
                arrow.setAlignment(Qt.AlignCenter)
                pipeline_layout.addWidget(arrow, 0, Qt.AlignVCenter)
        flow_layout.addLayout(pipeline_layout)
        layout.addWidget(flow_card)

        diagram_card = QFrame()
        diagram_card.setObjectName("contentCard")
        diagram_card_layout = QVBoxLayout(diagram_card)
        diagram_card_layout.setContentsMargins(24, 22, 24, 22)
        diagram_card_layout.setSpacing(14)

        diagram_header = QHBoxLayout()
        diagram_title = QLabel("论文图示与模块细节")
        diagram_title.setObjectName("sectionCardTitle")
        diagram_header.addWidget(diagram_title)
        diagram_header.addStretch()
        diagram_note = QLabel("默认读取 assets/presentation/theory，可在界面中临时替换主框图。")
        diagram_note.setObjectName("mutedText")
        diagram_note.setWordWrap(True)
        diagram_header.addWidget(diagram_note, 1)
        diagram_card_layout.addLayout(diagram_header)

        self.theory_diagram_cards = []
        diagram_specs = self.get_theory_diagram_specs()
        if diagram_specs:
            main_card = ImageCard(diagram_specs[0][0], accent=True)
            main_card.set_titled_frame_mode()
            diagram_card_layout.addWidget(main_card)
            self.theory_diagram_cards.append(main_card)

            detail_grid = QGridLayout()
            detail_grid.setHorizontalSpacing(14)
            detail_grid.setVerticalSpacing(14)
            for index, (title_text, _) in enumerate(diagram_specs[1:]):
                card = ImageCard(title_text, accent=False)
                card.set_titled_frame_mode()
                detail_grid.addWidget(card, 0, index)
                self.theory_diagram_cards.append(card)
            if len(diagram_specs) > 1:
                diagram_card_layout.addLayout(detail_grid)
        layout.addWidget(diagram_card)

        layout.addStretch()
        return scroll

    def build_framework_flow_card(self):
        card = QFrame()
        card.setObjectName("contentCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)

        title = QLabel("QPE-HViT 方法框图（内置示意）")
        title.setObjectName("sectionCardTitle")
        layout.addWidget(title)

        subtitle = QLabel("左侧示意图可直接用于答辩讲解；右侧可替换为论文中的正式框图。")
        subtitle.setObjectName("mutedText")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        flow_layout = QHBoxLayout()
        flow_layout.setSpacing(12)
        pipeline = THEORY_DATA["pipeline"]
        for index, item in enumerate(pipeline):
            flow_layout.addWidget(
                self.create_flow_node(
                    item["title"],
                    item["description"],
                    item.get("tone", "primary"),
                ),
                1,
            )
            if index < len(pipeline) - 1:
                arrow = QLabel("→")
                arrow.setObjectName("flowArrow")
                arrow.setAlignment(Qt.AlignCenter)
                flow_layout.addWidget(arrow)
        layout.addLayout(flow_layout)

        reinforce_title = QLabel("增强模块解释")
        reinforce_title.setObjectName("sectionCardTitle")
        layout.addWidget(reinforce_title)

        reinforce_layout = QHBoxLayout()
        reinforce_layout.setSpacing(12)
        for item in THEORY_DATA["innovation_cards"]:
            reinforce_layout.addWidget(
                self.create_feature_card(
                    item["title"],
                    item["description"],
                    accent=item.get("accent", False),
                ),
                1,
            )
        layout.addLayout(reinforce_layout)
        return card

    def build_performance_page(self):
        scroll, layout = self.build_scroll_page()
        layout.setContentsMargins(38, 34, 38, 26)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("performanceHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 24, 28, 24)
        hero_layout.setSpacing(22)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(10)
        hero_tag = QLabel("EXP 01 / PERFORMANCE")
        hero_tag.setObjectName("tagPill")
        hero_text.addWidget(hero_tag, 0, Qt.AlignLeft)

        hero_title = QLabel("第一组：性能对比实验")
        hero_title.setObjectName("heroTitle")
        hero_title.setWordWrap(True)
        hero_text.addWidget(hero_title)

        hero_subtitle = QLabel(
            "在统一训练配置、统一数据预处理与统一评估指标下，比较 QPE-HViT 与 CNN、ViT、"
            "层次化 Transformer 和混合模型的 Top-1 精度、参数量与计算量。"
        )
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_text.addWidget(hero_subtitle)

        hero_meta = QHBoxLayout()
        hero_meta.setSpacing(10)
        for text in ("Top-1 Acc", "Params", "FLOPs", "多数据集对比"):
            label = QLabel(text)
            label.setObjectName("performanceMetaPill")
            hero_meta.addWidget(label, 0, Qt.AlignLeft)
        hero_meta.addStretch()
        hero_text.addLayout(hero_meta)
        hero_layout.addLayout(hero_text, 1)

        hero_panel = QFrame()
        hero_panel.setObjectName("performanceHeroPanel")
        panel_layout = QVBoxLayout(hero_panel)
        panel_layout.setContentsMargins(20, 18, 20, 18)
        panel_layout.setSpacing(10)
        panel_title = QLabel("对比维度")
        panel_title.setObjectName("performancePanelTitle")
        panel_layout.addWidget(panel_title)
        for text in ("模型族谱：CNN / ViT / Hybrid / Proposed", "排序规则：按 Top-1 Acc 自动降序", "高亮规则：QPE-HViT 行重点标识"):
            item = QLabel(text)
            item.setObjectName("performancePanelText")
            item.setWordWrap(True)
            panel_layout.addWidget(item)
        hero_layout.addWidget(hero_panel, 0, Qt.AlignRight)
        layout.addWidget(hero)

        control_card = QFrame()
        control_card.setObjectName("performanceControlCard")
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(22, 18, 22, 18)
        control_layout.setSpacing(14)

        title = QLabel("选择数据集")
        title.setObjectName("performanceControlTitle")
        control_layout.addWidget(title)

        self.performance_dataset_box = QComboBox()
        self.performance_dataset_box.setMinimumWidth(240)
        self.performance_dataset_box.addItems(PERFORMANCE_DATA["datasets"])
        self.performance_dataset_box.currentTextChanged.connect(self.refresh_performance_table)
        control_layout.addWidget(self.performance_dataset_box)

        self.performance_dataset_button = QPushButton("▼")
        self.performance_dataset_button.setObjectName("comboTrigger")
        self.performance_dataset_button.clicked.connect(self.performance_dataset_box.showPopup)
        control_layout.addWidget(self.performance_dataset_button)

        control_layout.addStretch()

        desc = QLabel("表格会按 Top-1 Acc 自动排序，并高亮显示 QPE-HViT。")
        desc.setObjectName("performanceControlHint")
        desc.setWordWrap(True)
        control_layout.addWidget(desc, 2)
        layout.addWidget(control_card)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)

        left_col = QVBoxLayout()
        left_col.setSpacing(16)

        summary_layout = QGridLayout()
        summary_layout.setHorizontalSpacing(14)
        summary_layout.setVerticalSpacing(14)
        self.performance_summary_cards = []
        for index in range(3):
            metric = self.create_performance_metric_card(index)
            self.performance_summary_cards.append(metric)
            summary_layout.addWidget(metric["card"], 0, index)
        left_col.addLayout(summary_layout)

        table_card = QFrame()
        table_card.setObjectName("performanceTableCard")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(22, 22, 22, 22)
        table_layout.setSpacing(14)

        table_header = QHBoxLayout()
        table_title = QLabel("主实验对比结果")
        table_title.setObjectName("sectionCardTitle")
        table_header.addWidget(table_title)
        table_header.addStretch()
        table_badge = QLabel("QPE-HViT 行已高亮")
        table_badge.setObjectName("performanceTableBadge")
        table_header.addWidget(table_badge)
        table_layout.addLayout(table_header)

        self.performance_table = QTableWidget()
        self.performance_table.setColumnCount(6)
        self.performance_table.setHorizontalHeaderLabels(
            ["排名", "模型", "类别", "Top-1 Acc", "Params", "FLOPs"]
        )
        self.performance_table.verticalHeader().setVisible(False)
        self.performance_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.performance_table.setSelectionMode(QTableWidget.NoSelection)
        self.performance_table.setAlternatingRowColors(True)
        self.performance_table.verticalHeader().setDefaultSectionSize(46)
        self.performance_table.setMinimumHeight(430)

        header = self.performance_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)

        table_layout.addWidget(self.performance_table)

        note = QLabel(PERFORMANCE_DATA["note"])
        note.setObjectName("mutedText")
        note.setWordWrap(True)
        table_layout.addWidget(note)
        left_col.addWidget(table_card)
        body_layout.addLayout(left_col, 5)

        analysis_card = QFrame()
        analysis_card.setObjectName("performanceAnalysisCard")
        analysis_layout = QVBoxLayout(analysis_card)
        analysis_layout.setContentsMargins(24, 22, 24, 22)
        analysis_layout.setSpacing(12)

        analysis_title = QLabel("实验解读")
        analysis_title.setObjectName("sectionCardTitle")
        analysis_layout.addWidget(analysis_title)

        for index, insight in enumerate(PERFORMANCE_DATA.get("insights", []), start=1):
            analysis_layout.addWidget(
                self.create_performance_insight_item(
                    index,
                    insight["title"],
                    insight["text"],
                )
            )

        analysis_layout.addStretch()
        body_layout.addWidget(analysis_card, 2)
        layout.addLayout(body_layout)
        layout.addStretch()
        return scroll

    def build_ablation_page(self):
        scroll, layout = self.build_scroll_page()
        layout.setContentsMargins(38, 34, 38, 26)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("ablationHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 24, 28, 24)
        hero_layout.setSpacing(22)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(10)
        hero_tag = QLabel("EXP 02 / ABLATION")
        hero_tag.setObjectName("tagPill")
        hero_text.addWidget(hero_tag, 0, Qt.AlignLeft)

        hero_title = QLabel("第二组：消融实验")
        hero_title.setObjectName("heroTitle")
        hero_title.setWordWrap(True)
        hero_text.addWidget(hero_title)

        hero_subtitle = QLabel(
            "通过基础 HViT、仅 QPE、仅 QCSA 与完整 QPE-HViT 四类变体，验证量子位置先验与通道-空间协同注意的独立贡献和协同增益。"
        )
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_text.addWidget(hero_subtitle)

        hero_meta = QHBoxLayout()
        hero_meta.setSpacing(10)
        for text in ("CIFAR-100", "模块贡献", "协同增益", "量子超参数"):
            label = QLabel(text)
            label.setObjectName("ablationMetaPill")
            hero_meta.addWidget(label, 0, Qt.AlignLeft)
        hero_meta.addStretch()
        hero_text.addLayout(hero_meta)
        hero_layout.addLayout(hero_text, 1)

        hero_panel = QFrame()
        hero_panel.setObjectName("ablationHeroPanel")
        panel_layout = QVBoxLayout(hero_panel)
        panel_layout.setContentsMargins(20, 18, 20, 18)
        panel_layout.setSpacing(10)
        panel_title = QLabel("验证路径")
        panel_title.setObjectName("ablationPanelTitle")
        panel_layout.addWidget(panel_title)
        for text in ("先看单模块收益：QPE 与 QCSA 是否各自有效", "再看完整模型：两个模块是否产生协同提升", "最后看 qubits 设置：精度与模拟代价是否平衡"):
            item = QLabel(text)
            item.setObjectName("ablationPanelText")
            item.setWordWrap(True)
            panel_layout.addWidget(item)
        hero_layout.addWidget(hero_panel, 0, Qt.AlignRight)
        layout.addWidget(hero)

        summary_layout = QGridLayout()
        summary_layout.setHorizontalSpacing(14)
        summary_layout.setVerticalSpacing(14)
        self.ablation_summary_cards = []
        for index in range(3):
            metric = self.create_ablation_metric_card(index)
            self.ablation_summary_cards.append(metric)
            summary_layout.addWidget(metric["card"], 0, index)
        layout.addLayout(summary_layout)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)

        table_card = QFrame()
        table_card.setObjectName("ablationTableCard")
        table_layout = QVBoxLayout(table_card)
        table_layout.setContentsMargins(22, 22, 22, 22)
        table_layout.setSpacing(14)

        table_header = QHBoxLayout()
        table_title = QLabel("模块消融结果")
        table_title.setObjectName("sectionCardTitle")
        table_header.addWidget(table_title)
        table_header.addStretch()
        table_badge = QLabel("完整模型已高亮")
        table_badge.setObjectName("ablationTableBadge")
        table_header.addWidget(table_badge)
        table_layout.addLayout(table_header)

        self.ablation_table = QTableWidget()
        self.ablation_table.setColumnCount(7)
        self.ablation_table.setHorizontalHeaderLabels(
            ["模型变体", "QPE", "QCSA", "Top-1 Acc", "Params", "FLOPs", "相对基础增益"]
        )
        self.ablation_table.verticalHeader().setVisible(False)
        self.ablation_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ablation_table.setSelectionMode(QTableWidget.NoSelection)
        self.ablation_table.setAlternatingRowColors(True)
        self.ablation_table.verticalHeader().setDefaultSectionSize(48)
        self.ablation_table.setMinimumHeight(300)

        header = self.ablation_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeToContents)

        table_layout.addWidget(self.ablation_table)
        body_layout.addWidget(table_card, 5)

        insight_card = QFrame()
        insight_card.setObjectName("ablationInsightCard")
        insight_layout = QVBoxLayout(insight_card)
        insight_layout.setContentsMargins(24, 22, 24, 22)
        insight_layout.setSpacing(12)

        title = QLabel("模块贡献分析")
        title.setObjectName("sectionCardTitle")
        insight_layout.addWidget(title)

        self.ablation_module_insight_label = QLabel()
        self.ablation_module_insight_label.setObjectName("ablationInsightText")
        self.ablation_module_insight_label.setWordWrap(True)
        insight_layout.addWidget(self.ablation_module_insight_label)

        self.ablation_gain_items = []
        for index in range(3):
            item = self.create_ablation_gain_item(index)
            insight_layout.addWidget(item["card"])
            self.ablation_gain_items.append(item)
        insight_layout.addStretch()
        body_layout.addWidget(insight_card, 2)
        layout.addLayout(body_layout)

        hyper_table_card = QFrame()
        hyper_table_card.setObjectName("ablationHyperCard")
        hyper_table_layout = QVBoxLayout(hyper_table_card)
        hyper_table_layout.setContentsMargins(22, 22, 22, 22)
        hyper_table_layout.setSpacing(14)

        hyper_data = ABLATION_DATA["hyperparameter_study"]

        hyper_title = QLabel(hyper_data["title"])
        hyper_title.setObjectName("sectionCardTitle")
        hyper_table_layout.addWidget(hyper_title)

        hyper_note = QLabel(hyper_data["note"])
        hyper_note.setObjectName("mutedText")
        hyper_note.setWordWrap(True)
        hyper_table_layout.addWidget(hyper_note)

        self.ablation_hyperparameter_table = QTableWidget()
        self.ablation_hyperparameter_table.setColumnCount(len(hyper_data["columns"]))
        self.ablation_hyperparameter_table.setHorizontalHeaderLabels(hyper_data["columns"])
        self.ablation_hyperparameter_table.verticalHeader().setVisible(False)
        self.ablation_hyperparameter_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.ablation_hyperparameter_table.setSelectionMode(QTableWidget.NoSelection)
        self.ablation_hyperparameter_table.setAlternatingRowColors(True)
        self.ablation_hyperparameter_table.verticalHeader().setDefaultSectionSize(46)
        self.ablation_hyperparameter_table.setMinimumHeight(190)

        hyper_header = self.ablation_hyperparameter_table.horizontalHeader()
        hyper_header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hyper_header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hyper_header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hyper_header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        hyper_header.setSectionResizeMode(4, QHeaderView.Stretch)

        hyper_table_layout.addWidget(self.ablation_hyperparameter_table)

        self.ablation_hyperparameter_insight_label = QLabel()
        self.ablation_hyperparameter_insight_label.setObjectName("ablationInsightText")
        self.ablation_hyperparameter_insight_label.setWordWrap(True)
        hyper_table_layout.addWidget(self.ablation_hyperparameter_insight_label)

        layout.addWidget(hyper_table_card)
        layout.addStretch()
        return scroll

    def build_visualization_page(self):
        scroll, layout = self.build_scroll_page()
        layout.setContentsMargins(38, 34, 38, 26)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("visualHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 24, 28, 24)
        hero_layout.setSpacing(22)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(10)
        hero_tag = QLabel("EXP 03 / VISUAL EXPLANATION")
        hero_tag.setObjectName("tagPill")
        hero_text.addWidget(hero_tag, 0, Qt.AlignLeft)

        hero_title = QLabel("第三组：可视化实验")
        hero_title.setObjectName("heroTitle")
        hero_title.setWordWrap(True)
        hero_text.addWidget(hero_title)

        hero_subtitle = QLabel(
            "通过 Grad-CAM、t-SNE 与多尺度特征响应三类结果，从注意区域、特征流形和层级响应三个角度解释 QPE-HViT 的性能来源。"
        )
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_text.addWidget(hero_subtitle)

        hero_meta = QHBoxLayout()
        hero_meta.setSpacing(10)
        for text in ("Grad-CAM", "t-SNE", "多尺度响应", "可解释性证据"):
            label = QLabel(text)
            label.setObjectName("visualMetaPill")
            hero_meta.addWidget(label, 0, Qt.AlignLeft)
        hero_meta.addStretch()
        hero_text.addLayout(hero_meta)
        hero_layout.addLayout(hero_text, 1)

        hero_panel = QFrame()
        hero_panel.setObjectName("visualHeroPanel")
        panel_layout = QVBoxLayout(hero_panel)
        panel_layout.setContentsMargins(20, 18, 20, 18)
        panel_layout.setSpacing(10)
        panel_title = QLabel("解释维度")
        panel_title.setObjectName("visualPanelTitle")
        panel_layout.addWidget(panel_title)
        for text in ("空间注意：目标区域是否更集中", "特征流形：类间边界是否更清晰", "层级响应：局部到全局是否有序演化"):
            item = QLabel(text)
            item.setObjectName("visualPanelText")
            item.setWordWrap(True)
            panel_layout.addWidget(item)
        hero_layout.addWidget(hero_panel, 0, Qt.AlignRight)
        layout.addWidget(hero)

        control_card = QFrame()
        control_card.setObjectName("visualControlCard")
        control_layout = QHBoxLayout(control_card)
        control_layout.setContentsMargins(22, 18, 22, 18)
        control_layout.setSpacing(14)

        title = QLabel("可视化方法")
        title.setObjectName("visualControlTitle")
        control_layout.addWidget(title)

        self.visual_method_box = QComboBox()
        self.visual_method_box.setMinimumWidth(320)
        self.visual_method_box.addItems([item["name"] for item in VISUALIZATION_DATA])
        self.visual_method_box.currentIndexChanged.connect(self.refresh_visualization_page)
        control_layout.addWidget(self.visual_method_box)

        self.visual_method_button = QPushButton("▼")
        self.visual_method_button.setObjectName("comboTrigger")
        self.visual_method_button.clicked.connect(self.visual_method_box.showPopup)
        control_layout.addWidget(self.visual_method_button)

        control_layout.addStretch()

        self.visual_method_desc = QLabel()
        self.visual_method_desc.setObjectName("visualControlHint")
        self.visual_method_desc.setWordWrap(True)
        control_layout.addWidget(self.visual_method_desc, 2)
        layout.addWidget(control_card)

        comparison_layout = QHBoxLayout()
        comparison_layout.setSpacing(18)

        left_col = QVBoxLayout()
        left_col.setSpacing(16)

        compare_card = QFrame()
        compare_card.setObjectName("visualCompareCard")
        compare_layout = QVBoxLayout(compare_card)
        compare_layout.setContentsMargins(22, 22, 22, 22)
        compare_layout.setSpacing(14)

        compare_header = QHBoxLayout()
        compare_title = QLabel("模型可解释性对比")
        compare_title.setObjectName("sectionCardTitle")
        compare_header.addWidget(compare_title)
        compare_header.addStretch()
        self.visual_figure_label = QLabel()
        self.visual_figure_label.setObjectName("visualFigureBadge")
        self.visual_figure_label.setWordWrap(True)
        compare_header.addWidget(self.visual_figure_label, 1)
        compare_layout.addLayout(compare_header)

        image_grid = QHBoxLayout()
        image_grid.setSpacing(14)
        self.visual_baseline_card = ImageCard("基础模型", accent=False)
        self.visual_baseline_card.set_titled_frame_mode()
        self.visual_baseline_card.image_label.setMinimumHeight(260)
        self.visual_ours_card = ImageCard("QPE-HViT", accent=True)
        self.visual_ours_card.set_titled_frame_mode()
        self.visual_ours_card.image_label.setMinimumHeight(260)
        image_grid.addWidget(self.visual_baseline_card, 1)
        image_grid.addWidget(self.visual_ours_card, 1)
        compare_layout.addLayout(image_grid)
        left_col.addWidget(compare_card)

        self.visual_result_card = ImageCard("可视化结果", accent=True)
        self.visual_result_card.set_titled_frame_mode()
        self.visual_result_card.image_label.setMinimumHeight(360)
        self.visual_result_card.upload_button.setText("替换当前方法图片")
        self.visual_result_card.upload_button.clicked.connect(self.choose_visual_image)
        left_col.addWidget(self.visual_result_card)
        comparison_layout.addLayout(left_col, 5)

        insight_card = QFrame()
        insight_card.setObjectName("visualInsightCard")
        insight_layout = QVBoxLayout(insight_card)
        insight_layout.setContentsMargins(24, 22, 24, 22)
        insight_layout.setSpacing(12)

        insight_title = QLabel("解释结论")
        insight_title.setObjectName("sectionCardTitle")
        insight_layout.addWidget(insight_title)

        self.visual_method_tag = QLabel()
        self.visual_method_tag.setObjectName("visualMethodTag")
        insight_layout.addWidget(self.visual_method_tag, 0, Qt.AlignLeft)

        self.visual_insight_label = QLabel()
        self.visual_insight_label.setObjectName("visualInsightText")
        self.visual_insight_label.setWordWrap(True)
        insight_layout.addWidget(self.visual_insight_label)

        self.visual_result_caption_label = QLabel()
        self.visual_result_caption_label.setObjectName("visualCaptionText")
        self.visual_result_caption_label.setWordWrap(True)
        insight_layout.addWidget(self.visual_result_caption_label)

        self.visual_evidence_items = []
        for index in range(3):
            item = self.create_visual_evidence_item(index)
            insight_layout.addWidget(item["card"])
            self.visual_evidence_items.append(item)
        insight_layout.addStretch()
        comparison_layout.addWidget(insight_card, 2)
        layout.addLayout(comparison_layout)
        layout.addStretch()
        return scroll

    def build_gesture_page(self):
        scroll, layout = self.build_scroll_page()
        layout.setContentsMargins(38, 34, 38, 26)
        layout.setSpacing(18)

        hero = QFrame()
        hero.setObjectName("gestureHero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(28, 24, 28, 24)
        hero_layout.setSpacing(22)

        hero_text = QVBoxLayout()
        hero_text.setSpacing(10)
        hero_tag = QLabel("DEMO / HAND GESTURE")
        hero_tag.setObjectName("tagPill")
        hero_text.addWidget(hero_tag, 0, Qt.AlignLeft)

        hero_title = QLabel("手势数字识别")
        hero_title.setObjectName("heroTitle")
        hero_title.setWordWrap(True)
        hero_text.addWidget(hero_title)

        hero_subtitle = QLabel(
            "面向 0-9 手势数字的实时识别演示，支持摄像头推理与单张图片推理，用于答辩中展示系统交互能力和模型部署效果。"
        )
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_text.addWidget(hero_subtitle)

        hero_meta = QHBoxLayout()
        hero_meta.setSpacing(10)
        for text in ("实时摄像头", "模型输入预览", "Top-3 候选", "图片推理"):
            label = QLabel(text)
            label.setObjectName("gestureMetaPill")
            hero_meta.addWidget(label, 0, Qt.AlignLeft)
        hero_meta.addStretch()
        hero_text.addLayout(hero_meta)
        hero_layout.addLayout(hero_text, 1)

        hero_panel = QFrame()
        hero_panel.setObjectName("gestureHeroPanel")
        panel_layout = QVBoxLayout(hero_panel)
        panel_layout.setContentsMargins(20, 18, 20, 18)
        panel_layout.setSpacing(10)
        panel_title = QLabel("演示流程")
        panel_title.setObjectName("gesturePanelTitle")
        panel_layout.addWidget(panel_title)
        for text in ("选择识别模型与输入模式", "打开摄像头或选择本地图片", "观察预测数字、置信度与候选结果"):
            item = QLabel(text)
            item.setObjectName("gesturePanelText")
            item.setWordWrap(True)
            panel_layout.addWidget(item)
        hero_layout.addWidget(hero_panel, 0, Qt.AlignRight)
        layout.addWidget(hero)

        layout.addWidget(self.create_gesture_recognition_card())
        layout.addStretch()
        return scroll

    def create_gesture_recognition_card(self):
        card = QFrame()
        card.setObjectName("gestureConsoleCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(16)

        control_panel = QFrame()
        control_panel.setObjectName("gestureControlCard")
        header_layout = QHBoxLayout(control_panel)
        header_layout.setContentsMargins(18, 16, 18, 16)
        header_layout.setSpacing(14)

        title_group = QVBoxLayout()
        title_group.setSpacing(5)

        title = QLabel("识别控制台")
        title.setObjectName("sectionCardTitle")
        title_group.addWidget(title)

        subtitle = QLabel("摄像头模式会截取中央手势区域；图片模式会直接对选择图片进行一次推理。")
        subtitle.setObjectName("gestureControlHint")
        subtitle.setWordWrap(True)
        title_group.addWidget(subtitle)
        header_layout.addLayout(title_group, 1)

        model_label = QLabel("识别模型")
        model_label.setObjectName("metricLabel")
        header_layout.addWidget(model_label)

        self.gesture_model_box = QComboBox()
        self.gesture_model_box.setMinimumWidth(210)
        self.gesture_model_box.addItems(
            [
                "Gesture CNN（已训练）",
                "OpenCV 规则演示",
            ]
        )
        self.gesture_model_box.currentTextChanged.connect(self.clear_gesture_prediction_history)
        header_layout.addWidget(self.gesture_model_box)

        input_label = QLabel("推理输入")
        input_label.setObjectName("metricLabel")
        header_layout.addWidget(input_label)

        self.gesture_input_mode_box = QComboBox()
        self.gesture_input_mode_box.setMinimumWidth(180)
        self.gesture_input_mode_box.addItems(
            [
                "手部裁剪（推荐）",
                "中央灰度 ROI",
                "二值化 ROI",
            ]
        )
        self.gesture_input_mode_box.currentTextChanged.connect(self.on_gesture_input_mode_changed)
        header_layout.addWidget(self.gesture_input_mode_box)
        layout.addWidget(control_panel)

        body_layout = QHBoxLayout()
        body_layout.setSpacing(18)

        camera_panel = QFrame()
        camera_panel.setObjectName("cameraPanel")
        camera_layout = QVBoxLayout(camera_panel)
        camera_layout.setContentsMargins(16, 16, 16, 16)
        camera_layout.setSpacing(12)

        preview_layout = QHBoxLayout()
        preview_layout.setSpacing(12)

        live_preview = QFrame()
        live_preview.setObjectName("gesturePreviewCard")
        live_layout = QVBoxLayout(live_preview)
        live_layout.setContentsMargins(12, 12, 12, 12)
        live_layout.setSpacing(10)

        live_title = QLabel("实时画面")
        live_title.setObjectName("metricLabel")
        live_layout.addWidget(live_title)

        self.gesture_camera_view = QLabel("摄像头未开启")
        self.gesture_camera_view.setObjectName("cameraViewport")
        self.gesture_camera_view.setAlignment(Qt.AlignCenter)
        self.gesture_camera_view.setFixedHeight(360)
        self.gesture_camera_view.setMinimumWidth(560)
        self.gesture_camera_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        live_layout.addWidget(self.gesture_camera_view)
        preview_layout.addWidget(live_preview, 3)

        binary_preview = QFrame()
        binary_preview.setObjectName("gesturePreviewCard")
        binary_layout = QVBoxLayout(binary_preview)
        binary_layout.setContentsMargins(12, 12, 12, 12)
        binary_layout.setSpacing(10)

        self.gesture_model_input_title = QLabel("模型输入预览")
        self.gesture_model_input_title.setObjectName("metricLabel")
        binary_layout.addWidget(self.gesture_model_input_title)

        self.gesture_model_input_view = QLabel("等待摄像头画面")
        self.gesture_model_input_view.setObjectName("binaryViewport")
        self.gesture_model_input_view.setAlignment(Qt.AlignCenter)
        self.gesture_model_input_view.setFixedHeight(360)
        self.gesture_model_input_view.setMinimumWidth(260)
        self.gesture_model_input_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        binary_layout.addWidget(self.gesture_model_input_view)
        preview_layout.addWidget(binary_preview, 1)

        camera_layout.addLayout(preview_layout)

        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(10)
        self.open_camera_button = QPushButton("打开摄像头")
        self.open_camera_button.setObjectName("gesturePrimaryButton")
        self.open_camera_button.clicked.connect(self.start_gesture_camera)
        controls_layout.addWidget(self.open_camera_button)

        self.close_camera_button = QPushButton("关闭摄像头")
        self.close_camera_button.setObjectName("gestureSecondaryButton")
        self.close_camera_button.setEnabled(False)
        self.close_camera_button.clicked.connect(self.stop_gesture_camera)
        controls_layout.addWidget(self.close_camera_button)
        controls_layout.addStretch()
        camera_layout.addLayout(controls_layout)

        image_card = QFrame()
        image_card.setObjectName("capturePanel")
        image_layout = QVBoxLayout(image_card)
        image_layout.setContentsMargins(16, 14, 16, 14)
        image_layout.setSpacing(12)

        image_title = QLabel("图片识别")
        image_title.setObjectName("metricLabel")
        image_layout.addWidget(image_title)

        image_controls = QHBoxLayout()
        image_controls.setSpacing(10)

        self.choose_gesture_image_button = QPushButton("选择图片进行识别")
        self.choose_gesture_image_button.setObjectName("gesturePrimaryButton")
        self.choose_gesture_image_button.clicked.connect(self.choose_gesture_image_for_prediction)
        image_controls.addWidget(self.choose_gesture_image_button)
        image_controls.addStretch()
        image_layout.addLayout(image_controls)

        self.selected_gesture_image_view = QLabel("尚未选择图片")
        self.selected_gesture_image_view.setObjectName("selectedImageViewport")
        self.selected_gesture_image_view.setAlignment(Qt.AlignCenter)
        self.selected_gesture_image_view.setFixedHeight(200)
        self.selected_gesture_image_view.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Fixed)
        image_layout.addWidget(self.selected_gesture_image_view)

        self.selected_gesture_image_label = QLabel("图片路径：未选择")
        self.selected_gesture_image_label.setObjectName("mutedText")
        self.selected_gesture_image_label.setWordWrap(True)
        image_layout.addWidget(self.selected_gesture_image_label)
        camera_layout.addWidget(image_card)

        body_layout.addWidget(camera_panel, 3)

        result_panel = QFrame()
        result_panel.setObjectName("gestureResultPanel")
        result_panel.setMinimumWidth(330)
        result_panel.setMaximumWidth(390)
        result_layout = QVBoxLayout(result_panel)
        result_layout.setContentsMargins(20, 20, 20, 20)
        result_layout.setSpacing(14)

        result_title = QLabel("识别结果")
        result_title.setObjectName("sectionCardTitle")
        result_layout.addWidget(result_title)

        digit_card = QFrame()
        digit_card.setObjectName("gestureDigitCard")
        digit_layout = QVBoxLayout(digit_card)
        digit_layout.setContentsMargins(12, 14, 12, 14)
        digit_layout.setSpacing(4)

        self.gesture_digit_label = QLabel("--")
        self.gesture_digit_label.setObjectName("gestureDigit")
        self.gesture_digit_label.setAlignment(Qt.AlignCenter)
        digit_layout.addWidget(self.gesture_digit_label)

        self.gesture_confidence_label = QLabel("置信度：--")
        self.gesture_confidence_label.setObjectName("gestureConfidence")
        self.gesture_confidence_label.setAlignment(Qt.AlignCenter)
        digit_layout.addWidget(self.gesture_confidence_label)
        result_layout.addWidget(digit_card)

        topk_card = QFrame()
        topk_card.setObjectName("gestureInfoCard")
        topk_layout = QVBoxLayout(topk_card)
        topk_layout.setContentsMargins(14, 12, 14, 12)
        topk_layout.setSpacing(6)

        topk_title = QLabel("候选结果")
        topk_title.setObjectName("metricLabel")
        topk_layout.addWidget(topk_title)

        self.gesture_topk_label = QLabel("Top-3：等待识别")
        self.gesture_topk_label.setObjectName("gestureRankText")
        self.gesture_topk_label.setWordWrap(True)
        topk_layout.addWidget(self.gesture_topk_label)
        result_layout.addWidget(topk_card)

        status_card = QFrame()
        status_card.setObjectName("gestureStatusBox")
        status_layout = QVBoxLayout(status_card)
        status_layout.setContentsMargins(14, 12, 14, 12)
        status_layout.setSpacing(6)

        status_title = QLabel("运行状态")
        status_title.setObjectName("metricLabel")
        status_layout.addWidget(status_title)

        self.gesture_status_label = QLabel(
            "当前为 OpenCV 演示识别器，并非真实训练模型；打开摄像头后会自动刷新识别结果。"
        )
        self.gesture_status_label.setObjectName("mutedText")
        self.gesture_status_label.setWordWrap(True)
        status_layout.addWidget(self.gesture_status_label)
        result_layout.addWidget(status_card)
        result_layout.addStretch()

        body_layout.addWidget(result_panel, 1)
        layout.addLayout(body_layout)

        if cv2 is None:
            self.open_camera_button.setEnabled(False)
            self.gesture_status_label.setText("未检测到 OpenCV，摄像头识别功能不可用。")

        return card

    def create_page_header(self, title_text, subtitle_text, tag_text):
        frame = QFrame()
        frame.setObjectName("heroCard")
        layout = QVBoxLayout(frame)
        layout.setContentsMargins(28, 26, 28, 26)
        layout.setSpacing(10)

        tag = QLabel(tag_text)
        tag.setObjectName("tagPill")
        layout.addWidget(tag, 0, Qt.AlignLeft)

        title = QLabel(title_text)
        title.setObjectName("heroTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        subtitle = QLabel(subtitle_text)
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        return frame

    def create_metric_card(self):
        card = QFrame()
        card.setObjectName("metricCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title = QLabel("指标")
        title.setObjectName("metricLabel")
        layout.addWidget(title)

        value = QLabel("--")
        value.setObjectName("valueLabel")
        value.setWordWrap(True)
        layout.addWidget(value)

        caption = QLabel("")
        caption.setObjectName("metricCaption")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        return {"card": card, "title": title, "value": value, "caption": caption}

    def create_performance_metric_card(self, index=0):
        card_names = [
            "performanceMetricCardBlue",
            "performanceMetricCardGreen",
            "performanceMetricCardAmber",
        ]
        icon_names = ["DS", "TOP", "QPE"]
        card = QFrame()
        card.setObjectName(card_names[index % len(card_names)])
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        icon = QLabel(icon_names[index % len(icon_names)])
        icon.setObjectName("performanceMetricIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(42, 42)
        top_row.addWidget(icon, 0, Qt.AlignTop)

        title = QLabel("指标")
        title.setObjectName("performanceMetricTitle")
        title.setWordWrap(True)
        top_row.addWidget(title, 1, Qt.AlignVCenter)
        layout.addLayout(top_row)

        value = QLabel("--")
        value.setObjectName("performanceMetricValue")
        value.setWordWrap(True)
        layout.addWidget(value)

        caption = QLabel("")
        caption.setObjectName("performanceMetricCaption")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        return {"card": card, "title": title, "value": value, "caption": caption}

    def create_performance_insight_item(self, index, title_text, body_text):
        card = QFrame()
        card.setObjectName("performanceInsightItem")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(12)

        badge = QLabel(f"{index:02d}")
        badge.setObjectName("performanceInsightBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(34, 34)
        layout.addWidget(badge, 0, Qt.AlignTop)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(4)
        title = QLabel(title_text)
        title.setObjectName("performanceInsightTitle")
        title.setWordWrap(True)
        text_box.addWidget(title)

        body = QLabel(body_text)
        body.setObjectName("performanceInsightBody")
        body.setWordWrap(True)
        text_box.addWidget(body)
        layout.addLayout(text_box, 1)
        return card

    def create_ablation_metric_card(self, index=0):
        card_names = [
            "ablationMetricCardBlue",
            "ablationMetricCardGreen",
            "ablationMetricCardAmber",
        ]
        icon_names = ["SET", "QPE", "ALL"]
        card = QFrame()
        card.setObjectName(card_names[index % len(card_names)])
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(8)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)
        icon = QLabel(icon_names[index % len(icon_names)])
        icon.setObjectName("ablationMetricIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(42, 42)
        top_row.addWidget(icon, 0, Qt.AlignTop)

        title = QLabel("指标")
        title.setObjectName("ablationMetricTitle")
        title.setWordWrap(True)
        top_row.addWidget(title, 1, Qt.AlignVCenter)
        layout.addLayout(top_row)

        value = QLabel("--")
        value.setObjectName("ablationMetricValue")
        value.setWordWrap(True)
        layout.addWidget(value)

        caption = QLabel("")
        caption.setObjectName("ablationMetricCaption")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        return {"card": card, "title": title, "value": value, "caption": caption}

    def create_ablation_gain_item(self, index):
        card = QFrame()
        card.setObjectName("ablationGainItem")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(12)

        badge = QLabel(f"{index + 1:02d}")
        badge.setObjectName("ablationGainBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(34, 34)
        layout.addWidget(badge, 0, Qt.AlignTop)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(4)

        title = QLabel()
        title.setObjectName("ablationGainTitle")
        title.setWordWrap(True)
        text_box.addWidget(title)

        value = QLabel()
        value.setObjectName("ablationGainValue")
        value.setWordWrap(True)
        text_box.addWidget(value)
        layout.addLayout(text_box, 1)
        return {"card": card, "title": title, "value": value}

    def create_visual_evidence_item(self, index):
        card = QFrame()
        card.setObjectName("visualEvidenceItem")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 6, 0, 6)
        layout.setSpacing(12)

        badge = QLabel(f"{index + 1:02d}")
        badge.setObjectName("visualEvidenceBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(34, 34)
        layout.addWidget(badge, 0, Qt.AlignTop)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(4)

        title = QLabel()
        title.setObjectName("visualEvidenceTitle")
        title.setWordWrap(True)
        text_box.addWidget(title)

        body = QLabel()
        body.setObjectName("visualEvidenceBody")
        body.setWordWrap(True)
        text_box.addWidget(body)
        layout.addLayout(text_box, 1)
        return {"card": card, "title": title, "body": body}

    def create_home_metric_card(self, index=0):
        card = QFrame()
        metric_styles = [
            ("homeMetricCardBlue", "homeMetricIconBlue"),
            ("homeMetricCardGreen", "homeMetricIconGreen"),
            ("homeMetricCardIndigo", "homeMetricIconIndigo"),
            ("homeMetricCardAmber", "homeMetricIconAmber"),
        ]
        card_name, icon_name = metric_styles[index % len(metric_styles)]
        card.setObjectName(card_name)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(9)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)

        icon = QLabel()
        icon.setObjectName(icon_name)
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(48, 48)

        title = QLabel("指标")
        title.setObjectName("homeMetricTitle")
        title.setWordWrap(True)

        top_row.addWidget(icon, 0, Qt.AlignTop)
        top_row.addWidget(title, 1, Qt.AlignVCenter)
        layout.addLayout(top_row)

        value = QLabel("--")
        value.setObjectName("homeMetricValue")
        value.setWordWrap(True)
        layout.addWidget(value)

        caption = QLabel("")
        caption.setObjectName("homeMetricCaption")
        caption.setWordWrap(True)
        layout.addWidget(caption)
        return {"card": card, "icon": icon, "title": title, "value": value, "caption": caption}

    def set_metric_card(self, metric, title, value, caption):
        metric["title"].setText(title)
        metric["value"].setText(value)
        metric["caption"].setText(caption)

    def set_home_metric_card(self, metric, icon_text, title, value, caption):
        metric["icon"].setText(icon_text)
        metric["title"].setText(title)
        metric["value"].setText(value)
        metric["caption"].setText(caption)

    def create_theory_innovation_card(self, index, title_text, body_text, accent=False):
        card = QFrame()
        card.setObjectName("theoryInnovationAccent" if accent else "theoryInnovationCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(9)

        badge = QLabel(f"{index:02d}")
        badge.setObjectName("theoryInnovationBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(38, 28)
        layout.addWidget(badge, 0, Qt.AlignLeft)

        title = QLabel(title_text)
        title.setObjectName("theoryInnovationTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        body = QLabel(body_text)
        body.setObjectName("theoryInnovationBody")
        body.setWordWrap(True)
        layout.addWidget(body)
        return card

    def create_theory_point_card(self, index, text):
        card = QFrame()
        card.setObjectName("theoryPointRow")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)

        badge = QLabel(f"{index:02d}")
        badge.setObjectName("theoryPointBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(34, 34)
        layout.addWidget(badge, 0, Qt.AlignTop)

        body = QLabel(text)
        body.setObjectName("theoryPointText")
        body.setWordWrap(True)
        layout.addWidget(body, 1)
        return card

    def create_theory_talk_item(self, index, text):
        card = QFrame()
        card.setObjectName("theoryTalkItem")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 5, 0, 5)
        layout.setSpacing(10)

        badge = QLabel(str(index))
        badge.setObjectName("theoryTalkBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(28, 28)
        layout.addWidget(badge, 0, Qt.AlignTop)

        body = QLabel(text)
        body.setObjectName("theoryTalkText")
        body.setWordWrap(True)
        layout.addWidget(body, 1)
        return card

    def create_theory_pipeline_step(self, index, item):
        card = QFrame()
        card.setObjectName("theoryPipelineStepAccent" if item.get("tone") == "accent" else "theoryPipelineStep")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        badge = QLabel(f"STEP {index}")
        badge.setObjectName("theoryPipelineBadge")
        layout.addWidget(badge, 0, Qt.AlignLeft)

        title = QLabel(item["title"])
        title.setObjectName("theoryPipelineTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        desc = QLabel(item["description"])
        desc.setObjectName("theoryPipelineDesc")
        desc.setWordWrap(True)
        layout.addWidget(desc)
        return card

    def create_feature_card(self, title_text, body_text, accent=False):
        card = QFrame()
        card.setObjectName("accentSoftCard" if accent else "softCard")
        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setObjectName("sectionCardTitle")
        title.setWordWrap(True)
        layout.addWidget(title)

        body = QLabel(body_text)
        body.setObjectName("bodyText")
        body.setWordWrap(True)
        layout.addWidget(body)
        return card

    def create_flow_node(self, title_text, body_text, tone):
        object_name = {
            "primary": "flowNodePrimary",
            "accent": "flowNodeAccent",
            "soft": "flowNodeSoft",
        }.get(tone, "flowNodePrimary")

        title_color = "#f8fbfc" if tone == "primary" else "#17303a"
        body_color = "#dce7ea" if tone == "primary" else "#44565c"

        card = QFrame()
        card.setObjectName(object_name)
        card.setMinimumHeight(170)

        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(8)

        title = QLabel(title_text)
        title.setWordWrap(True)
        title.setStyleSheet(f"font-size: 16px; font-weight: 700; color: {title_color};")
        layout.addWidget(title)

        body = QLabel(body_text)
        body.setWordWrap(True)
        body.setStyleSheet(f"font-size: 13px; color: {body_color};")
        layout.addWidget(body)
        return card

    def create_bullet_label(self, text):
        label = QLabel(f"• {text}")
        label.setWordWrap(True)
        label.setObjectName("bodyText")
        return label

    def create_home_timeline_item(self, step_number):
        card = QFrame()
        card.setObjectName("timelineRow")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(14)

        badge = QLabel(f"{step_number:02d}")
        badge.setObjectName("timelineBadge")
        badge.setAlignment(Qt.AlignCenter)
        badge.setFixedSize(38, 38)
        layout.addWidget(badge, 0, Qt.AlignTop)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(4)

        title = QLabel()
        title.setObjectName("timelineTitle")
        title.setWordWrap(True)
        text_box.addWidget(title)

        desc = QLabel()
        desc.setObjectName("timelineDesc")
        desc.setWordWrap(True)
        text_box.addWidget(desc)
        layout.addLayout(text_box, 1)

        return {"card": card, "badge": badge, "title": title, "desc": desc}

    def create_home_highlight_item(self):
        card = QFrame()
        card.setObjectName("highlightItem")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(12)

        icon = QLabel()
        icon.setObjectName("highlightIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedSize(36, 36)
        layout.addWidget(icon, 0, Qt.AlignTop)

        text_box = QVBoxLayout()
        text_box.setContentsMargins(0, 0, 0, 0)
        text_box.setSpacing(3)

        title = QLabel()
        title.setObjectName("highlightTitle")
        title.setWordWrap(True)
        text_box.addWidget(title)

        desc = QLabel()
        desc.setObjectName("highlightDesc")
        desc.setWordWrap(True)
        text_box.addWidget(desc)
        layout.addLayout(text_box, 1)

        return {"card": card, "icon": icon, "title": title, "desc": desc}

    def populate_home_page(self):
        self.home_hero_meta_labels[0].setText("理论方法 + 三组实验 + 手势识别")
        self.home_hero_meta_labels[1].setText(f"{len(PERFORMANCE_DATA['datasets'])} 个公开数据集")
        self.home_hero_meta_labels[2].setText(f"{len(PERFORMANCE_DATA['models'])} 个对比模型")

        self.set_home_metric_card(
            self.home_stat_cards[0],
            "UI",
            "展示界面",
            f"{self.stack.count()} 个",
            "首页、理论方法、三组实验与识别演示统一入口",
        )
        self.set_home_metric_card(
            self.home_stat_cards[1],
            "DS",
            "公开数据集",
            f"{len(PERFORMANCE_DATA['datasets'])} 个",
            "可在性能对比页一键切换",
        )
        self.set_home_metric_card(
            self.home_stat_cards[2],
            "ML",
            "对比模型",
            f"{len(PERFORMANCE_DATA['models'])} 个",
            "覆盖 CNN / ViT / Hybrid / Proposed",
        )
        self.set_home_metric_card(
            self.home_stat_cards[3],
            "VIS",
            "可视化方法",
            f"{len(VISUALIZATION_DATA)} 类",
            "Grad-CAM、t-SNE、多尺度响应",
        )

        timeline = [
            ("理论方法页面", "讲解 QPE-HViT 的设计动机、核心模块与整体推理流程，先建立方法来源。"),
            ("性能对比页面", "在统一训练配置下比较 QPE-HViT 与主流视觉模型的精度、参数量与 FLOPs。"),
            ("消融实验页面", "分析基础模型、仅 QPE、仅 QCSA 与完整模型的模块贡献差异。"),
            ("可视化实验页面", "利用 Grad-CAM、t-SNE 和多尺度响应图解释模型性能来源。"),
            ("手势识别演示", "展示系统交互能力，为答辩收尾提供直观演示模块。"),
        ]
        for widgets, (title, desc) in zip(self.home_timeline_items, timeline):
            widgets["title"].setText(title)
            widgets["desc"].setText(desc)

        highlights = [
            ("QPE", "量子位置编码（QPE）", "引入量子相位先验，增强位置感知与长程依赖建模能力。"),
            ("QC", "量子通道注意力（QCSA）", "联合通道与空间信息，提升关键特征选择与表达能力。"),
            ("MS", "多尺度特征融合", "兼顾局部纹理与全局语义，增强模型鲁棒性。"),
            ("VA", "多维可视化分析", "从注意区域、特征空间与层级响应支持答辩讲解。"),
        ]
        for widgets, (icon, title, desc) in zip(self.home_highlight_items, highlights):
            widgets["icon"].setText(icon)
            widgets["title"].setText(title)
            widgets["desc"].setText(desc)

        cards = [
            ("TH", "THEORY", "理论方法", "讲解模型动机、QPE 与 QCSA 两个核心模块，以及整体方法框图。", "theory"),
            ("P1", "EXP 01", "性能对比实验", "比较 QPE-HViT 与 ViT、Swin Transformer、MViT 等主流模型的精度、参数量与 FLOPs。", "performance"),
            ("P2", "EXP 02", "消融实验", "展示基础模型、仅 QPE、仅 QCSA 与完整模型四类变体，解释模块独立贡献与协同增益。", "ablation"),
            ("P3", "EXP 03", "可视化实验", "通过 Grad-CAM、t-SNE、多尺度特征响应三种视图，对比基础模型与 QPE-HViT 的内部特征差异。", "visualization"),
        ]

        for (icon_text, tag, title, desc, page_key), widgets in zip(cards, self.home_module_cards):
            icon_label, eyebrow, title_label, desc_label, button = widgets
            icon_label.setText(icon_text)
            eyebrow.setText(tag)
            title_label.setText(title)
            desc_label.setText(desc)
            if self.home_module_button_keys.get(button) != page_key:
                button.clicked.connect(partial(self.switch_page, page_key))
                self.home_module_button_keys[button] = page_key

    def get_theory_diagram_specs(self):
        base_dir = Path(__file__).resolve().parent / "assets" / "presentation" / "theory"
        return [
            ("QPE-HViT 整体方法框图", str((base_dir / "framework.png").resolve())),
            ("量子位置编码（QPE）层的详细架构", str((base_dir / "qpe_detail.png").resolve())),
            ("基于量子引导权重修正的 Transformer 块", str((base_dir / "quantum_guided_transformer_block.png").resolve())),
        ]

    def populate_theory_page(self):
        for card, (title_text, image_path) in zip(self.theory_diagram_cards, self.get_theory_diagram_specs()):
            card.title_label.setText(title_text)
            card.set_default_path(image_path)

    def populate_performance_page(self):
        self.refresh_performance_table(self.performance_dataset_box.currentText())

    def refresh_performance_table(self, dataset_name):
        rows = []
        for item in PERFORMANCE_DATA["models"]:
            acc = item["accuracy"].get(dataset_name)
            if acc is None:
                continue
            rows.append(
                {
                    "name": item["name"],
                    "family": item["family"],
                    "acc": float(acc),
                    "params": item["params"],
                    "flops": item["flops"],
                    "highlight": item.get("highlight", False),
                }
            )

        rows.sort(key=lambda entry: entry["acc"], reverse=True)
        self.performance_table.setRowCount(len(rows))

        for row_index, entry in enumerate(rows):
            values = [
                str(row_index + 1),
                entry["name"],
                entry["family"],
                f"{entry['acc']:.2f}%",
                entry["params"],
                entry["flops"],
            ]
            for col_index, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if entry["highlight"]:
                    item.setBackground(QColor("#dce9ff"))
                    item.setForeground(QColor("#142b52"))
                self.performance_table.setItem(row_index, col_index, item)

        best = rows[0] if rows else None
        ours = next((entry for entry in rows if entry["highlight"]), None)
        runner_up = rows[1] if ours and best and ours["name"] == best["name"] and len(rows) > 1 else best
        lead = ours["acc"] - runner_up["acc"] if ours and runner_up and ours["name"] == best["name"] else 0.0

        self.set_metric_card(
            self.performance_summary_cards[0],
            "当前数据集",
            dataset_name,
            "统一训练与评估设置",
        )
        self.set_metric_card(
            self.performance_summary_cards[1],
            "最高精度模型",
            f"{best['name']} ({best['acc']:.2f}%)" if best else "--",
            "表格已按 Top-1 Acc 自动排序",
        )
        if ours and best and ours["name"] == best["name"] and runner_up:
            self.set_metric_card(
                self.performance_summary_cards[2],
                "QPE-HViT 领先幅度",
                f"+{lead:.2f}%",
                f"相对第二名 {runner_up['name']}",
            )
        else:
            self.set_metric_card(
                self.performance_summary_cards[2],
                "QPE-HViT 当前结果",
                f"{ours['acc']:.2f}%" if ours else "--",
                f"{ours['params']} | {ours['flops']}" if ours else "暂无结果",
            )

        self.performance_table.resizeRowsToContents()

    def populate_ablation_page(self):
        variants = ABLATION_DATA["variants"]
        hyperparameter_variants = ABLATION_DATA["hyperparameter_study"]["variants"]
        base_acc = variants[0]["acc"]
        self.ablation_table.setRowCount(len(variants))

        for row_index, variant in enumerate(variants):
            gain = variant["acc"] - base_acc
            values = [
                variant["name"],
                "是" if variant["qpe"] else "否",
                "是" if variant["qcsa"] else "否",
                f"{variant['acc']:.2f}%",
                variant["params"],
                variant["flops"],
                f"{gain:+.2f}%",
            ]
            for col_index, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if variant.get("highlight", False):
                    item.setBackground(QColor("#dce9ff"))
                    item.setForeground(QColor("#142b52"))
                self.ablation_table.setItem(row_index, col_index, item)

        self.ablation_hyperparameter_table.setRowCount(len(hyperparameter_variants))
        for row_index, variant in enumerate(hyperparameter_variants):
            values = [
                str(variant["qubits"]),
                str(variant["layers"]),
                f"{variant['acc']:.2f}",
                f"{variant['flops']:.2f}",
                variant["note"],
            ]
            for col_index, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setTextAlignment(Qt.AlignCenter)
                if variant.get("highlight", False):
                    item.setBackground(QColor("#dce9ff"))
                    item.setForeground(QColor("#142b52"))
                self.ablation_hyperparameter_table.setItem(row_index, col_index, item)

        qpe_gain = variants[1]["acc"] - base_acc
        qcsa_gain = variants[2]["acc"] - base_acc
        full_gain = variants[3]["acc"] - base_acc

        self.set_metric_card(
            self.ablation_summary_cards[0],
            "评估数据集",
            ABLATION_DATA["dataset"],
            "与论文消融设置保持一致",
        )
        self.set_metric_card(
            self.ablation_summary_cards[1],
            "仅加入 QPE",
            f"+{qpe_gain:.2f}%",
            "单模块独立收益",
        )
        self.set_metric_card(
            self.ablation_summary_cards[2],
            "完整模型提升",
            f"+{full_gain:.2f}%",
            "QPE 与 QCSA 协同增益",
        )

        self.ablation_module_insight_label.setText(
            "基础 HViT 作为对照组；仅加入 QPE 与仅加入 QCSA 均能带来独立收益。"
            f"其中 QPE 带来 {qpe_gain:.2f}% 的提升，QCSA 带来 {qcsa_gain:.2f}% 的提升；"
            f"完整 QPE-HViT 相对基础模型累计提升 {full_gain:.2f}%，说明两个模块具有协同优化作用。"
        )
        gain_items = [
            ("仅加入 QPE", f"+{qpe_gain:.2f}%  |  位置先验带来的独立收益"),
            ("仅加入 QCSA", f"+{qcsa_gain:.2f}%  |  通道-空间注意带来的独立收益"),
            ("完整 QPE-HViT", f"+{full_gain:.2f}%  |  两个模块共同作用后的总体提升"),
        ]
        for widgets, (title, value) in zip(self.ablation_gain_items, gain_items):
            widgets["title"].setText(title)
            widgets["value"].setText(value)

        self.ablation_hyperparameter_insight_label.setText(
            ABLATION_DATA["hyperparameter_study"]["insight"]
        )
        self.ablation_table.resizeRowsToContents()
        self.ablation_hyperparameter_table.resizeRowsToContents()

    def populate_visualization_page(self):
        self.refresh_visualization_page(0)

    def refresh_visualization_page(self, index):
        if index < 0 or index >= len(VISUALIZATION_DATA):
            return

        data = VISUALIZATION_DATA[index]
        self.visual_method_desc.setText(data["description"])
        self.visual_figure_label.setText(data.get("figure_label", ""))
        self.visual_method_tag.setText(data["name"])
        self.visual_insight_label.setText(data.get("insight", ""))
        self.visual_result_caption_label.setText(data.get("result_caption", ""))

        self.visual_baseline_card.title_label.setText(data.get("baseline_title", "基础模型"))
        self.visual_baseline_card.set_caption(data.get("baseline_caption", ""))
        self.visual_baseline_card.set_default_path(data.get("baseline_image", ""))

        self.visual_ours_card.title_label.setText(data.get("ours_title", "QPE-HViT"))
        self.visual_ours_card.set_caption(data.get("ours_caption", ""))
        self.visual_ours_card.set_default_path(data.get("ours_image", ""))

        default_path = self.get_visualization_default_path(index)
        self.visual_result_card.title_label.setText(data["name"])
        self.visual_result_card.set_caption(data.get("result_caption", ""))
        self.visual_result_card.set_default_path(default_path)
        self.visual_result_card.set_custom_path(self.visual_custom_paths.get(index, ""))

        evidence_sets = [
            [
                ("关注区域", "观察热力图是否集中于目标主体，判断背景抑制与主体定位能力。"),
                ("响应强度", "比较关键区域激活是否更稳定，辅助说明判别证据来源。"),
                ("答辩结论", "QPE-HViT 若呈现更集中的响应，可作为量子先验改善空间注意的可视化支撑。"),
            ],
            [
                ("类内聚集", "观察同类样本在二维映射中的紧凑程度，评估特征一致性。"),
                ("类间分离", "比较不同类别之间的边界清晰度，辅助说明表征质量。"),
                ("答辩结论", "若 QPE-HViT 的类簇更紧凑且边界更清晰，可支持其高层特征表达更稳定。"),
            ],
            [
                ("浅层响应", "Stage 1 主要观察边缘、纹理和局部结构响应。"),
                ("深层语义", "Stage 2/3 关注响应是否逐步聚焦到主体与全局判别区域。"),
                ("答辩结论", "从局部纹理到全局语义的有序演化，可用于说明多尺度融合和 QCSA 的作用。"),
            ],
        ]
        for widgets, (title, body) in zip(self.visual_evidence_items, evidence_sets[index]):
            widgets["title"].setText(title)
            widgets["body"].setText(body)

    def choose_theory_image(self):
        path = self.choose_image_path("选择论文方法框图")
        if path and self.theory_diagram_cards:
            self.theory_diagram_cards[0].set_custom_path(path)
            self.statusBar().showMessage(f"已临时载入理论框图：{Path(path).name}")

    def choose_visual_image(self):
        path = self.choose_image_path("选择演示图片")
        if not path:
            return

        method_index = self.visual_method_box.currentIndex()
        self.visual_custom_paths[method_index] = path
        self.visual_result_card.set_custom_path(path)
        self.statusBar().showMessage(f"已临时载入图片：{Path(path).name}")

    def get_visualization_default_path(self, index):
        folder_names = ["gradcam", "tsne", "multiscale"]
        if index < 0 or index >= len(folder_names):
            return ""

        folder = Path(__file__).resolve().parent / "assets" / "visualizations" / folder_names[index]
        if not folder.exists():
            return ""

        for pattern in ("*.png", "*.jpg", "*.jpeg", "*.bmp"):
            matches = sorted(folder.glob(pattern))
            if matches:
                return str(matches[0])
        return ""

    def choose_image_path(self, title):
        path, _ = QFileDialog.getOpenFileName(
            self,
            title,
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)",
        )
        return path

    def clear_gesture_prediction_history(self):
        self.gesture_prediction_history.clear()
        if self.opencv_gesture_recognizer is not None:
            self.opencv_gesture_recognizer.reset()

    def on_gesture_input_mode_changed(self):
        self.clear_gesture_prediction_history()
        if hasattr(self, "gesture_model_input_title"):
            self.gesture_model_input_title.setText(f"模型输入预览：{self.gesture_input_mode_box.currentText()}")
        if self.latest_camera_frame is not None:
            self.show_model_input_frame(self.latest_camera_frame)

    def choose_gesture_image_for_prediction(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择手势图片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.webp)",
        )
        if not path:
            return

        image = self.read_image_with_unicode_path(path)
        if image is None:
            self.gesture_status_label.setText(f"图片读取失败：{path}")
            return

        self.selected_gesture_image_label.setText(f"图片路径：{path}")
        self.show_selected_gesture_image(image)
        self.predict_selected_gesture_image(image, Path(path).name)

    def show_selected_gesture_image(self, image):
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.selected_gesture_image_view.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.selected_gesture_image_view.setText("")
        self.selected_gesture_image_view.setPixmap(pixmap)

    def predict_selected_gesture_image(self, image, image_name):
        self.clear_gesture_prediction_history()
        model_name = self.gesture_model_box.currentText()
        digit, confidence, topk, message = self.predict_gesture_digit(image, model_name)
        self.gesture_digit_label.setText(str(digit) if digit is not None else "--")
        self.gesture_confidence_label.setText(f"置信度：{confidence:.1f}%" if digit is not None else "置信度：--")
        self.gesture_topk_label.setText(
            "\n".join(f"#{rank}  数字 {item_digit}    {score:.1f}%" for rank, (item_digit, score) in enumerate(topk, 1))
            if topk
            else "Top-3：等待识别"
        )
        self.gesture_status_label.setText(f"图片 {image_name} 识别完成。{message}")
        self.show_model_input_frame(image)
        self.statusBar().showMessage(f"图片识别结果：{self.gesture_digit_label.text()}")

    def read_image_with_unicode_path(self, path):
        if cv2 is None or np is None:
            return None
        try:
            data = np.fromfile(path, dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except Exception:
            return None

    def load_gesture_model(self):
        if not hasattr(self, "gesture_status_label"):
            return

        if torch is None or GestureCNN is None:
            self.gesture_status_label.setText("未检测到 PyTorch，已切换为 OpenCV 规则演示。")
            self.gesture_model_box.setCurrentText("OpenCV 规则演示")
            return

        if not self.gesture_model_path.exists():
            self.gesture_status_label.setText(f"未找到模型文件：{self.gesture_model_path.name}，已切换为 OpenCV 规则演示。")
            self.gesture_model_box.setCurrentText("OpenCV 规则演示")
            return

        try:
            checkpoint = torch.load(self.gesture_model_path, map_location="cpu")
            model = GestureCNN()
            model.load_state_dict(checkpoint["state_dict"])
            model.eval()
            self.gesture_model = model
            self.gesture_model_input_size = int(checkpoint.get("input_shape", [1, 1, 64, 64])[-1])
            self.gesture_model_classes = [str(item) for item in checkpoint.get("classes", list(range(10)))]

            meta_path = self.gesture_model_path.with_suffix(".meta.json")
            acc_text = ""
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                best_acc = meta.get("best_val_acc")
                if best_acc is not None:
                    acc_text = f"验证准确率：{best_acc * 100:.2f}%。"

            self.gesture_model_box.setCurrentText("Gesture CNN（已训练）")
            self.gesture_status_label.setText(f"已加载真实 CNN 模型：{self.gesture_model_path.name}。{acc_text}")
        except Exception as exc:
            self.gesture_model = None
            self.gesture_model_box.setCurrentText("OpenCV 规则演示")
            self.gesture_status_label.setText(f"模型加载失败：{exc}。已切换为 OpenCV 规则演示。")

    def start_gesture_camera(self):
        if cv2 is None:
            self.gesture_status_label.setText("未检测到 OpenCV，无法打开摄像头。")
            return

        if self.camera_capture is not None and self.camera_capture.isOpened():
            return

        capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not capture.isOpened():
            capture.release()
            capture = cv2.VideoCapture(0)

        if not capture.isOpened():
            self.gesture_status_label.setText("摄像头打开失败，请检查权限或设备占用状态。")
            self.statusBar().showMessage("手势识别摄像头打开失败。")
            return

        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        self.camera_capture = capture
        self.gesture_frame_counter = 0
        self.gesture_prediction_history.clear()
        self.camera_timer.start(33)
        self.open_camera_button.setEnabled(False)
        self.close_camera_button.setEnabled(True)
        self.gesture_status_label.setText("摄像头已开启，系统会自动识别中央框内的手势。")
        self.statusBar().showMessage("手势数字识别摄像头已开启。")

    def stop_gesture_camera(self):
        self.camera_timer.stop()
        if self.camera_capture is not None:
            self.camera_capture.release()
            self.camera_capture = None
        self.latest_camera_frame = None
        self.gesture_prediction_history.clear()
        self.gesture_camera_view.setPixmap(QPixmap())
        self.gesture_camera_view.setText("摄像头未开启")
        self.gesture_model_input_view.setPixmap(QPixmap())
        self.gesture_model_input_view.setText("等待摄像头画面")
        self.open_camera_button.setEnabled(cv2 is not None)
        self.close_camera_button.setEnabled(False)
        self.gesture_status_label.setText("摄像头已关闭。")
        self.statusBar().showMessage("手势数字识别摄像头已关闭。")

    def update_gesture_camera_frame(self):
        if self.camera_capture is None or not self.camera_capture.isOpened():
            self.stop_gesture_camera()
            return

        ok, frame = self.camera_capture.read()
        if not ok:
            self.gesture_status_label.setText("摄像头帧读取失败。")
            return

        frame = cv2.flip(frame, 1)
        self.latest_camera_frame = frame
        self.gesture_frame_counter += 1

        display_frame = frame.copy()
        x1, y1, x2, y2 = self.get_gesture_roi(display_frame)
        self.draw_gesture_roi(display_frame, x1, y1, x2, y2)
        if self.gesture_digit_label.text() != "--":
            cv2.putText(
                display_frame,
                f"Digit {self.gesture_digit_label.text()}",
                (x1 + 14, y2 - 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (216, 176, 107),
                2,
                cv2.LINE_AA,
            )

        self.show_camera_frame(display_frame)
        self.show_model_input_frame(frame)
        if self.gesture_frame_counter % 5 == 0:
            self.recognize_current_gesture(silent=True)

    def recognize_current_gesture(self, silent=False):
        if self.latest_camera_frame is None:
            if not silent:
                self.gesture_status_label.setText("请先打开摄像头并等待画面稳定。")
            return

        model_name = self.gesture_model_box.currentText()
        digit, confidence, topk, message = self.predict_gesture_digit(self.latest_camera_frame, model_name)
        self.gesture_digit_label.setText(str(digit) if digit is not None else "--")
        self.gesture_confidence_label.setText(f"置信度：{confidence:.1f}%" if digit is not None else "置信度：--")
        self.gesture_topk_label.setText(
            "\n".join(f"#{rank}  数字 {item_digit}    {score:.1f}%" for rank, (item_digit, score) in enumerate(topk, 1))
            if topk
            else "Top-3：等待识别"
        )
        self.gesture_status_label.setText(message)
        if not silent:
            self.statusBar().showMessage(f"当前手势识别结果：{self.gesture_digit_label.text()}")

    def get_gesture_roi(self, frame):
        height, width = frame.shape[:2]
        side = int(min(width, height) * 0.62)
        center_x = width // 2
        center_y = height // 2
        x1 = max(0, center_x - side // 2)
        y1 = max(0, center_y - side // 2)
        x2 = min(width, x1 + side)
        y2 = min(height, y1 + side)
        return x1, y1, x2, y2

    def draw_gesture_roi(self, frame, x1, y1, x2, y2):
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], y1), (12, 22, 25), -1)
        cv2.rectangle(overlay, (0, y2), (frame.shape[1], frame.shape[0]), (12, 22, 25), -1)
        cv2.rectangle(overlay, (0, y1), (x1, y2), (12, 22, 25), -1)
        cv2.rectangle(overlay, (x2, y1), (frame.shape[1], y2), (12, 22, 25), -1)
        cv2.addWeighted(overlay, 0.34, frame, 0.66, 0, frame)

        box_color = (216, 176, 107)
        soft_color = (236, 214, 174)
        line_len = max(34, int((x2 - x1) * 0.13))
        thickness = 3
        corners = [
            ((x1, y1), (x1 + line_len, y1), (x1, y1 + line_len)),
            ((x2, y1), (x2 - line_len, y1), (x2, y1 + line_len)),
            ((x1, y2), (x1 + line_len, y2), (x1, y2 - line_len)),
            ((x2, y2), (x2 - line_len, y2), (x2, y2 - line_len)),
        ]
        for origin, horizontal, vertical in corners:
            cv2.line(frame, origin, horizontal, box_color, thickness, cv2.LINE_AA)
            cv2.line(frame, origin, vertical, box_color, thickness, cv2.LINE_AA)

        cv2.rectangle(frame, (x1, y1), (x2, y2), soft_color, 1, cv2.LINE_AA)
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        cv2.drawMarker(frame, center, soft_color, cv2.MARKER_CROSS, 18, 1, cv2.LINE_AA)
        label = "Gesture ROI"
        cv2.rectangle(frame, (x1, max(0, y1 - 34)), (x1 + 142, y1), box_color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 10, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 33, 38),
            2,
            cv2.LINE_AA,
        )

    def show_camera_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.gesture_camera_view.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.gesture_camera_view.setText("")
        self.gesture_camera_view.setPixmap(pixmap)

    def show_model_input_frame(self, frame):
        if self.gesture_model_box.currentText() == "OpenCV 规则演示" and self.opencv_gesture_recognizer is not None:
            x1, y1, x2, y2 = self.get_gesture_roi(frame)
            roi = frame[y1:y2, x1:x2]
            if roi.size == 0:
                self.gesture_model_input_view.setPixmap(QPixmap())
                self.gesture_model_input_view.setText("无有效输入")
                return
            model_input = self.opencv_gesture_recognizer.build_skin_mask(roi)
            if hasattr(self, "gesture_model_input_title"):
                self.gesture_model_input_title.setText("OpenCV 肤色 Mask")
        else:
            model_input = self.build_gesture_model_input(frame)
            if hasattr(self, "gesture_model_input_title"):
                self.gesture_model_input_title.setText(f"模型输入预览：{self.gesture_input_mode_box.currentText()}")

        if model_input is None:
            self.gesture_model_input_view.setPixmap(QPixmap())
            self.gesture_model_input_view.setText("无有效输入")
            return

        bordered = cv2.copyMakeBorder(model_input, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=24)
        height, width = bordered.shape
        qimage = QImage(bordered.data, width, height, width, QImage.Format_Grayscale8).copy()
        pixmap = QPixmap.fromImage(qimage).scaled(
            self.gesture_model_input_view.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.gesture_model_input_view.setText("")
        self.gesture_model_input_view.setPixmap(pixmap)

    def build_gesture_binary_mask(self, frame):
        x1, y1, x2, y2 = self.get_gesture_roi(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    def predict_gesture_digit(self, frame, model_name):
        if model_name == "Gesture CNN（已训练）":
            if self.gesture_model is None:
                return None, 0.0, [], "真实 CNN 模型尚未加载，无法进行模型推理。"
            return self.predict_gesture_digit_with_cnn(frame)

        if self.opencv_gesture_recognizer is None:
            return None, 0.0, [], "OpenCV / NumPy 不可用，无法运行 OpenCV 规则识别。"

        x1, y1, x2, y2 = self.get_gesture_roi(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None, 0.0, [], "未获得有效手势区域。"

        result, _ = self.opencv_gesture_recognizer.recognize(roi)
        if result == -1:
            return None, 0.0, [], "未检测到明显手部，请把手放在中央框内。"

        topk = [(str(result), 100.0)]
        message = "OpenCV 肤色分割 + 凸包缺陷识别已完成；该模式识别的是伸出手指数，通常范围为 0-5。"
        return str(result), 100.0, topk, message

    def predict_gesture_digit_with_cnn(self, frame):
        model_input = self.build_gesture_model_input(frame)
        if model_input is None:
            return None, 0.0, [], "未定位到有效手势区域，无法进行 CNN 推理。"

        image = model_input.astype("float32") / 255.0
        image = (image - 0.5) / 0.5
        tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)

        with torch.inference_mode():
            logits = self.gesture_model(tensor)
            probabilities = torch.softmax(logits, dim=1)[0]
            self.gesture_prediction_history.append(probabilities.detach().cpu())
            smoothed_probabilities = torch.stack(list(self.gesture_prediction_history), dim=0).mean(dim=0)
            values, indices = torch.topk(smoothed_probabilities, k=min(3, smoothed_probabilities.numel()))

        topk = []
        for value, index in zip(values.tolist(), indices.tolist()):
            class_name = self.gesture_model_classes[index] if index < len(self.gesture_model_classes) else str(index)
            topk.append((class_name, value * 100.0))

        best_digit = topk[0][0] if topk else None
        confidence = topk[0][1] if topk else 0.0
        input_mode = self.gesture_input_mode_box.currentText()
        message = f"真实 CNN 模型已完成推理：{self.gesture_model_path.name}。当前输入：{input_mode}；最近 {len(self.gesture_prediction_history)} 帧平滑。"
        return best_digit, confidence, topk, message

    def build_gesture_model_input(self, frame):
        mode = self.gesture_input_mode_box.currentText()
        x1, y1, x2, y2 = self.get_gesture_roi(frame)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None

        if mode == "中央灰度 ROI":
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            return self.prepare_gesture_input_square(gray_roi, apply_equalize=False)

        mask = self.build_gesture_binary_mask(frame)
        if mask is None:
            return None

        if mode == "二值化 ROI":
            return self.prepare_gesture_input_square(mask, apply_equalize=False)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        if area / max(mask.shape[0] * mask.shape[1], 1) < 0.025:
            return None

        x, y, w, h = cv2.boundingRect(contour)
        pad = int(max(w, h) * 0.22)
        crop_x1 = max(0, x - pad)
        crop_y1 = max(0, y - pad)
        crop_x2 = min(roi.shape[1], x + w + pad)
        crop_y2 = min(roi.shape[0], y + h + pad)

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        crop = gray_roi[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            return None
        return self.prepare_gesture_input_square(crop, apply_equalize=False)

    def prepare_gesture_input_square(self, image, apply_equalize=False):
        if image is None or image.size == 0:
            return None

        side = max(image.shape[:2])
        canvas_value = int(float(image.mean()))
        square = cv2.copyMakeBorder(
            image,
            (side - image.shape[0]) // 2,
            side - image.shape[0] - (side - image.shape[0]) // 2,
            (side - image.shape[1]) // 2,
            side - image.shape[1] - (side - image.shape[1]) // 2,
            cv2.BORDER_CONSTANT,
            value=canvas_value,
        )
        if apply_equalize:
            square = cv2.equalizeHist(square)
        return cv2.resize(
            square,
            (self.gesture_model_input_size, self.gesture_model_input_size),
            interpolation=cv2.INTER_AREA,
        )

    def closeEvent(self, event):
        self.stop_gesture_camera()
        super().closeEvent(event)

    def apply_stylesheet(self):
        self.setStyleSheet(
            """
            QMainWindow { background-color: #eef4ff; color: #173059; }
            #sidebar { background-color: qlineargradient(x1: 0, y1: 0, x2: 0, y2: 1, stop: 0 #0a1a4d, stop: 1 #08163b); border-right: 1px solid rgba(130, 176, 255, 0.10); }
            #brandBlock { background-color: transparent; }
            #sideBadge { color: #ffbf59; background-color: rgba(255, 191, 89, 0.16); border: 1px solid rgba(255, 191, 89, 0.22); border-radius: 14px; padding: 6px 12px; font-size: 11px; font-weight: 800; letter-spacing: 1px; }
            #brandLogo { color: #f8fbff; background-color: rgba(76, 142, 255, 0.18); border: 1px solid rgba(149, 195, 255, 0.28); border-radius: 16px; font-size: 26px; font-weight: 900; }
            #brandTitle { color: #f8fbff; font-size: 28px; font-weight: 900; letter-spacing: 0px; }
            #brandSubtitle { color: rgba(215, 229, 255, 0.82); font-size: 13px; line-height: 1.6; }
            #divider { color: rgba(179, 205, 255, 0.16); }
            #navSectionLabel { color: rgba(198, 220, 255, 0.72); font-size: 12px; font-weight: 700; letter-spacing: 1px; }
            #navButton { min-height: 54px; text-align: left; padding: 12px 18px; border-radius: 16px; border: 1px solid transparent; background-color: rgba(255, 255, 255, 0.03); color: #eef4ff; font-size: 15px; font-weight: 700; }
            #navButton:hover { background-color: rgba(255, 255, 255, 0.08); border-color: rgba(128, 176, 255, 0.20); }
            #navButton:checked { background-color: #2f75f6; border-color: #2f75f6; color: #ffffff; }
            #sidebarFooterCard { background-color: rgba(255, 255, 255, 0.05); border: 1px solid rgba(143, 180, 255, 0.12); border-radius: 18px; }
            #sidebarFooterTitle { color: #f4f8ff; font-size: 13px; font-weight: 800; }
            #sidebarHint { color: rgba(199, 216, 245, 0.84); font-size: 12px; line-height: 1.65; }
            #heroCard, #contentCard, #metricCard, #imageCard, #imageCardAccent, #softCard, #accentSoftCard, #theoryHero, #theoryHeroModel, #theoryInnovationCard, #theoryInnovationAccent, #theoryPipelineStep, #theoryPipelineStepAccent, #performanceHero, #performanceHeroPanel, #performanceControlCard, #performanceTableCard, #performanceAnalysisCard, #performanceMetricCardBlue, #performanceMetricCardGreen, #performanceMetricCardAmber, #ablationHero, #ablationHeroPanel, #ablationTableCard, #ablationInsightCard, #ablationHyperCard, #ablationMetricCardBlue, #ablationMetricCardGreen, #ablationMetricCardAmber, #visualHero, #visualHeroPanel, #visualControlCard, #visualCompareCard, #visualInsightCard, #gestureHero, #gestureHeroPanel, #gestureConsoleCard, #gestureControlCard, #gesturePreviewCard, #homeModuleCard, #homeMetricCardBlue, #homeMetricCardGreen, #homeMetricCardIndigo, #homeMetricCardAmber { background-color: rgba(255, 255, 255, 0.98); border-radius: 18px; border: 1px solid rgba(214, 225, 243, 0.90); }
            #heroCard { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #eef5ff); }
            #homeHeroText, #homeHeroArt { background-color: transparent; border: none; }
            #homeHeroKicker { color: #3677e9; font-size: 11px; font-weight: 900; letter-spacing: 2px; }
            #homeHeroTitle { color: #142746; font-size: 35px; font-weight: 900; line-height: 1.18; }
            #homeHeroSubtitle { color: #60728b; font-size: 14px; line-height: 1.8; font-weight: 700; }
            #homeProjectBrief { color: #4f6684; background-color: rgba(255, 255, 255, 0.62); border: 1px solid rgba(214, 225, 243, 0.82); border-radius: 14px; padding: 10px 12px; font-size: 12px; line-height: 1.7; font-weight: 700; }
            #heroMetaPill { color: #3f5f91; background-color: rgba(73, 128, 238, 0.08); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 13px; padding: 7px 11px; font-size: 12px; font-weight: 800; }
            #homeVisualTag { color: #8a9bb1; font-size: 11px; font-weight: 900; letter-spacing: 1px; }
            #homeVisualStack { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #eef5ff); border: 1px solid rgba(204, 219, 244, 0.95); border-radius: 22px; min-width: 184px; }
            #homeStackTop { color: #ffffff; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #65adff, stop: 1 #2f75f6); border-radius: 13px; font-size: 15px; font-weight: 900; }
            #homeStackMid { color: #ffffff; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #7287ff, stop: 1 #5b58d7); border-radius: 13px; font-size: 15px; font-weight: 900; }
            #homeStackBase { color: #142746; background-color: #e8f0ff; border: 1px solid rgba(152, 183, 240, 0.70); border-radius: 13px; font-size: 15px; font-weight: 900; }
            #homeVisualNote { color: #74859b; font-size: 12px; line-height: 1.65; font-weight: 700; max-width: 210px; }
            #tagPill, #homeModuleTag { color: #5f80b7; background-color: rgba(84, 133, 232, 0.10); border: 1px solid rgba(84, 133, 232, 0.14); border-radius: 10px; padding: 4px 10px; font-size: 11px; font-weight: 800; }
            #heroTitle { color: #173059; font-size: 30px; font-weight: 800; }
            #pageTitle { color: #173059; font-size: 24px; font-weight: 900; }
            #heroSubtitle { color: #6f7f96; font-size: 14px; line-height: 1.7; }
            #theoryHero { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #theoryMetaPill { color: #3f5f91; background-color: rgba(73, 128, 238, 0.08); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 13px; padding: 7px 11px; font-size: 12px; font-weight: 800; }
            #theoryHeroModel { min-width: 250px; background-color: rgba(255, 255, 255, 0.74); }
            #theoryModelPrimary { color: #ffffff; background-color: #2f75f6; border-radius: 12px; font-size: 13px; font-weight: 900; }
            #theoryModelAccent { color: #ffffff; background-color: #f38a24; border-radius: 12px; font-size: 13px; font-weight: 900; }
            #theoryModelSoft { color: #173059; background-color: #e8f0ff; border: 1px solid rgba(152, 183, 240, 0.70); border-radius: 12px; font-size: 13px; font-weight: 900; }
            #theoryInnovationCard { min-height: 166px; background-color: #ffffff; }
            #theoryInnovationAccent { min-height: 166px; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #fff6e9); }
            #theoryInnovationBadge { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border-radius: 10px; font-size: 11px; font-weight: 900; }
            #theoryInnovationTitle { color: #173059; font-size: 16px; font-weight: 900; }
            #theoryInnovationBody { color: #60738f; font-size: 13px; line-height: 1.65; font-weight: 700; }
            #theoryPointRow, #theoryTalkItem { background-color: transparent; border: none; }
            #theoryPointBadge { color: #ffffff; background-color: #2f75f6; border-radius: 17px; font-size: 11px; font-weight: 900; }
            #theoryPointText { color: #425b7a; font-size: 13px; line-height: 1.72; font-weight: 700; }
            #theoryTalkBadge { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border-radius: 14px; font-size: 11px; font-weight: 900; }
            #theoryTalkText { color: #526985; font-size: 12px; line-height: 1.7; font-weight: 700; }
            #theoryPipelineStep { min-height: 150px; background-color: #f8fbff; }
            #theoryPipelineStepAccent { min-height: 150px; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #fffaf3, stop: 1 #fff1dd); border: 1px solid rgba(243, 138, 36, 0.30); }
            #theoryPipelineBadge { color: #5f80b7; background-color: rgba(73, 128, 238, 0.08); border-radius: 9px; padding: 4px 8px; font-size: 10px; font-weight: 900; }
            #theoryPipelineTitle { color: #173059; font-size: 15px; font-weight: 900; }
            #theoryPipelineDesc { color: #60738f; font-size: 12px; line-height: 1.62; font-weight: 700; }
            #theoryFlowArrow { color: #2f75f6; font-size: 22px; font-weight: 900; }
            #performanceHero { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #performanceMetaPill { color: #3f5f91; background-color: rgba(73, 128, 238, 0.08); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 13px; padding: 7px 11px; font-size: 12px; font-weight: 800; }
            #performanceHeroPanel { min-width: 330px; background-color: rgba(255, 255, 255, 0.76); }
            #performancePanelTitle { color: #173059; font-size: 15px; font-weight: 900; }
            #performancePanelText { color: #526985; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #performanceControlCard { background-color: #ffffff; }
            #performanceControlTitle { color: #173059; font-size: 18px; font-weight: 900; }
            #performanceControlHint { color: #60738f; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #performanceMetricCardBlue, #performanceMetricCardGreen, #performanceMetricCardAmber { min-height: 142px; }
            #performanceMetricCardBlue { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #performanceMetricCardGreen { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #effbf5); }
            #performanceMetricCardAmber { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #fff6e9); }
            #performanceMetricIcon { color: #ffffff; background-color: #2f75f6; border-radius: 14px; font-size: 11px; font-weight: 900; }
            #performanceMetricTitle { color: #586b86; font-size: 13px; font-weight: 900; }
            #performanceMetricValue { color: #142b52; font-size: 24px; font-weight: 900; }
            #performanceMetricCaption { color: #6f819a; font-size: 12px; line-height: 1.6; font-weight: 700; }
            #performanceTableBadge { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 11px; padding: 5px 10px; font-size: 11px; font-weight: 900; }
            #performanceInsightItem { background-color: transparent; border: none; }
            #performanceInsightBadge { color: #ffffff; background-color: #2f75f6; border-radius: 17px; font-size: 11px; font-weight: 900; }
            #performanceInsightTitle { color: #173059; font-size: 14px; font-weight: 900; }
            #performanceInsightBody { color: #60738f; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #ablationHero { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #ablationMetaPill { color: #3f5f91; background-color: rgba(73, 128, 238, 0.08); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 13px; padding: 7px 11px; font-size: 12px; font-weight: 800; }
            #ablationHeroPanel { min-width: 330px; background-color: rgba(255, 255, 255, 0.76); }
            #ablationPanelTitle { color: #173059; font-size: 15px; font-weight: 900; }
            #ablationPanelText { color: #526985; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #ablationMetricCardBlue, #ablationMetricCardGreen, #ablationMetricCardAmber { min-height: 142px; }
            #ablationMetricCardBlue { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #ablationMetricCardGreen { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #effbf5); }
            #ablationMetricCardAmber { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #fff6e9); }
            #ablationMetricIcon { color: #ffffff; background-color: #2f75f6; border-radius: 14px; font-size: 11px; font-weight: 900; }
            #ablationMetricTitle { color: #586b86; font-size: 13px; font-weight: 900; }
            #ablationMetricValue { color: #142b52; font-size: 24px; font-weight: 900; }
            #ablationMetricCaption { color: #6f819a; font-size: 12px; line-height: 1.6; font-weight: 700; }
            #ablationTableBadge { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 11px; padding: 5px 10px; font-size: 11px; font-weight: 900; }
            #ablationInsightText { color: #526985; font-size: 12px; line-height: 1.72; font-weight: 700; }
            #ablationGainItem { background-color: transparent; border: none; }
            #ablationGainBadge { color: #ffffff; background-color: #2f75f6; border-radius: 17px; font-size: 11px; font-weight: 900; }
            #ablationGainTitle { color: #173059; font-size: 14px; font-weight: 900; }
            #ablationGainValue { color: #60738f; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #visualHero { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #visualMetaPill { color: #3f5f91; background-color: rgba(73, 128, 238, 0.08); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 13px; padding: 7px 11px; font-size: 12px; font-weight: 800; }
            #visualHeroPanel { min-width: 330px; background-color: rgba(255, 255, 255, 0.76); }
            #visualPanelTitle { color: #173059; font-size: 15px; font-weight: 900; }
            #visualPanelText { color: #526985; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #visualControlCard { background-color: #ffffff; }
            #visualControlTitle { color: #173059; font-size: 18px; font-weight: 900; }
            #visualControlHint { color: #60738f; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #visualFigureBadge { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 11px; padding: 5px 10px; font-size: 11px; font-weight: 900; }
            #visualMethodTag { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 11px; padding: 5px 10px; font-size: 11px; font-weight: 900; }
            #visualInsightText { color: #526985; font-size: 12px; line-height: 1.72; font-weight: 700; }
            #visualCaptionText { color: #6f819a; background-color: #f8fbff; border: 1px solid rgba(214, 225, 243, 0.90); border-radius: 12px; padding: 10px 12px; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #visualEvidenceItem { background-color: transparent; border: none; }
            #visualEvidenceBadge { color: #ffffff; background-color: #2f75f6; border-radius: 17px; font-size: 11px; font-weight: 900; }
            #visualEvidenceTitle { color: #173059; font-size: 14px; font-weight: 900; }
            #visualEvidenceBody { color: #60738f; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #gestureHero { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #gestureMetaPill { color: #3f5f91; background-color: rgba(73, 128, 238, 0.08); border: 1px solid rgba(73, 128, 238, 0.14); border-radius: 13px; padding: 7px 11px; font-size: 12px; font-weight: 800; }
            #gestureHeroPanel { min-width: 330px; background-color: rgba(255, 255, 255, 0.76); }
            #gesturePanelTitle { color: #173059; font-size: 15px; font-weight: 900; }
            #gesturePanelText { color: #526985; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #gestureConsoleCard { background-color: #ffffff; }
            #gestureControlCard { background-color: #f8fbff; }
            #gestureControlHint { color: #60738f; font-size: 12px; line-height: 1.65; font-weight: 700; }
            #gesturePreviewCard { background-color: #ffffff; }
            #gesturePrimaryButton { min-height: 42px; padding: 10px 18px; border-radius: 10px; background-color: #2f75f6; border: 1px solid #2f75f6; color: #ffffff; font-weight: 800; }
            #gesturePrimaryButton:hover { background-color: #1f63e1; border-color: #1f63e1; }
            #gestureSecondaryButton { min-height: 42px; padding: 10px 18px; border-radius: 10px; background-color: #eef4ff; border: 1px solid #cbdcf7; color: #2f5f9f; font-weight: 800; }
            #gestureSecondaryButton:hover { background-color: #e1edff; border-color: #b9d0f4; }
            #accentSoftCard { background-color: #f7fbff; border: 1px solid rgba(112, 160, 245, 0.42); }
            #imageCardAccent { border: 2px solid rgba(90, 142, 240, 0.58); }
            #metricCard { min-height: 136px; }
            #homeMetricCardBlue, #homeMetricCardGreen, #homeMetricCardIndigo, #homeMetricCardAmber { min-height: 152px; }
            #homeMetricCardBlue { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f0f6ff); }
            #homeMetricCardGreen { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #effbf5); }
            #homeMetricCardIndigo { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #f3f2ff); }
            #homeMetricCardAmber { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffffff, stop: 1 #fff6e9); }
            #homeMetricIconBlue, #homeMetricIconGreen, #homeMetricIconIndigo, #homeMetricIconAmber { color: #ffffff; border-radius: 15px; font-size: 13px; font-weight: 900; }
            #homeMetricIconBlue { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #5a9cff, stop: 1 #2f75f6); }
            #homeMetricIconGreen { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #62cc95, stop: 1 #2fa873); }
            #homeMetricIconIndigo { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #7b74f2, stop: 1 #5650cf); }
            #homeMetricIconAmber { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffb64f, stop: 1 #f38a24); }
            #homeMetricTitle, #metricLabel { color: #586b86; font-size: 14px; font-weight: 900; }
            #homeMetricValue { color: #142b52; font-size: 30px; font-weight: 900; }
            #homeMetricCaption, #metricCaption { color: #6f819a; font-size: 12px; line-height: 1.6; }
            #valueLabel { color: #245dd8; font-size: 24px; font-weight: 800; }
            #sectionCardTitle { color: #173059; font-size: 20px; font-weight: 900; }
            #homeModuleTitle { color: #173059; font-size: 18px; font-weight: 900; }
            #bodyText { color: #415877; font-size: 14px; line-height: 1.7; }
            #mutedText { color: #6f819a; font-size: 12px; line-height: 1.6; }
            #timelineRow { background-color: transparent; }
            #timelineBadge { color: #ffffff; background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #4f8cff, stop: 1 #2d6df6); border-radius: 19px; font-size: 12px; font-weight: 900; }
            #timelineTitle { color: #173059; font-size: 14px; font-weight: 800; }
            #timelineDesc { color: #60738f; font-size: 13px; line-height: 1.65; }
            #highlightItem { background-color: transparent; border: none; }
            #highlightIcon { color: #2f75f6; background-color: rgba(73, 128, 238, 0.10); border-radius: 18px; font-size: 11px; font-weight: 900; }
            #highlightTitle { color: #173059; font-size: 14px; font-weight: 900; }
            #highlightDesc { color: #60738f; font-size: 12px; line-height: 1.6; font-weight: 700; }
            #homeModuleCard { min-height: 238px; background-color: #ffffff; }
            #homeModuleIconBlue, #homeModuleIconGreen, #homeModuleIconIndigo, #homeModuleIconAmber { color: #ffffff; border-radius: 18px; font-size: 13px; font-weight: 900; }
            #homeModuleIconBlue { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #5a9cff, stop: 1 #2f75f6); }
            #homeModuleIconGreen { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #62cc95, stop: 1 #2fa873); }
            #homeModuleIconIndigo { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #7b74f2, stop: 1 #5650cf); }
            #homeModuleIconAmber { background-color: qlineargradient(x1: 0, y1: 0, x2: 1, y2: 1, stop: 0 #ffb64f, stop: 1 #f38a24); }
            #ghostButton, QPushButton, #homeModuleButtonBlue, #homeModuleButtonGreen, #homeModuleButtonIndigo, #homeModuleButtonAmber { min-height: 42px; padding: 10px 18px; border-radius: 10px; color: #ffffff; font-weight: 800; }
            #ghostButton, QPushButton, #homeModuleButtonBlue, #homeModuleButtonIndigo { background-color: #2f75f6; border: 1px solid #2f75f6; }
            #homeModuleButtonGreen { background-color: #26a96f; border: 1px solid #26a96f; }
            #homeModuleButtonAmber { background-color: #f38a24; border: 1px solid #f38a24; }
            #ghostButton:hover, QPushButton:hover, #homeModuleButtonBlue:hover, #homeModuleButtonIndigo:hover { background-color: #1f63e1; border-color: #1f63e1; }
            #homeModuleButtonGreen:hover { background-color: #1f925f; border-color: #1f925f; }
            #homeModuleButtonAmber:hover { background-color: #df7818; border-color: #df7818; }
            #homeFooter { background-color: transparent; }
            #homeFooterText { color: #8190a8; font-size: 12px; font-weight: 700; }
            #comboTrigger { min-width: 42px; max-width: 42px; min-height: 42px; padding: 0px; font-size: 14px; }
            QPushButton:checked { background-color: #ffab3d; border-color: #ffab3d; color: #ffffff; }
            QComboBox { min-height: 40px; padding: 6px 10px; border-radius: 10px; background-color: #ffffff; border: 1px solid #d3def1; color: #173059; }
            QComboBox::drop-down { border: none; width: 28px; }
            QComboBox QAbstractItemView { background-color: #ffffff; color: #173059; selection-background-color: #2f75f6; selection-color: #ffffff; }
            QTableWidget { background-color: #ffffff; border: 1px solid #dce6f6; border-radius: 16px; gridline-color: #edf3fb; alternate-background-color: #f7faff; font-size: 13px; padding: 4px; }
            QHeaderView::section { background-color: #eef4ff; color: #173059; padding: 10px; border: none; border-right: 1px solid #dde7f7; font-weight: 800; }
            #imageViewportFrame { background-color: #f4f8ff; border-radius: 16px; border: 1px dashed #c7d9f3; }
            #imageViewport { background-color: transparent; color: #6d7f95; padding: 10px; }
            #cameraPanel { background-color: #f8fbff; border-radius: 16px; border: 1px solid #dce7f8; }
            #cameraPreviewCard { background-color: #ffffff; border-radius: 14px; border: 1px solid #dce7f8; }
            #capturePanel { background-color: #ffffff; border-radius: 14px; border: 1px solid #dce7f8; }
            #cameraViewport { background-color: #121c30; color: #d7e2f6; border-radius: 12px; border: 1px solid #2b426f; font-size: 15px; font-weight: 700; }
            #binaryViewport { background-color: #10182a; color: #d7e2f6; border-radius: 12px; border: 1px solid #2b426f; font-size: 14px; font-weight: 700; }
            #selectedImageViewport { background-color: #f4f8ff; color: #6d7f95; border-radius: 12px; border: 1px dashed #c7d9f3; font-size: 14px; font-weight: 700; }
            #gestureResultPanel { background-color: #f8fbff; border-radius: 16px; border: 1px solid #dce7f8; }
            #gestureDigitCard { background-color: #ffffff; border: 1px solid #dce7f8; border-radius: 16px; }
            #gestureInfoCard { background-color: #ffffff; border: 1px solid #dce7f8; border-radius: 14px; }
            #gestureStatusBox { background-color: #eef4ff; border: 1px solid #d8e4f7; border-radius: 14px; }
            #gestureDigit { color: #245dd8; font-size: 96px; font-weight: 900; }
            #gestureConfidence { color: #173059; font-size: 16px; font-weight: 800; }
            #gestureRankText { color: #415877; font-size: 14px; line-height: 1.7; font-weight: 600; }
            #flowNodePrimary { background-color: #173059; border: 1px solid #173059; border-radius: 18px; }
            #flowNodeAccent { background-color: #ffab3d; border: 1px solid #ffab3d; border-radius: 18px; }
            #flowNodeSoft { background-color: #eef4ff; border: 1px solid #d8e4f7; border-radius: 18px; }
            #flowArrow { color: #4f8cff; font-size: 28px; font-weight: 700; }
            QScrollArea { border: none; }
            QStatusBar { background-color: #0a1a4d; color: #eef4ff; }
            """
        )


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    window = QuantumViTGUI()
    window.show()
    sys.exit(app.exec())