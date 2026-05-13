"""
登录界面。
"""
import math
import sys
from pathlib import Path

from PySide6.QtCore import QPointF, QSettings, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QBrush,
    QColor,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)


BASE_DIR = Path(__file__).resolve().parent


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QPE-HViT 可视化演示系统 - 登录")
        self.setMinimumSize(980, 620)
        self.resize(1180, 720)

        self.background_pixmap = self._load_background()
        self._scaled_background = QPixmap()
        self._phase = 0.0

        self.init_ui()
        self._load_saved_preferences()
        self._prepare_background()

        self._animation_timer = QTimer(self)
        self._animation_timer.timeout.connect(self._advance_background)
        self._animation_timer.start(90)

    def init_ui(self):
        self.setObjectName("loginRoot")

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(56, 48, 56, 48)
        root_layout.setSpacing(36)

        left_panel = QFrame()
        left_panel.setObjectName("introPanel")
        left_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(4, 18, 34, 18)
        left_layout.setSpacing(18)

        brand = QFrame()
        brand.setObjectName("brandBlock")
        brand_layout = QHBoxLayout(brand)
        brand_layout.setContentsMargins(0, 0, 0, 0)
        brand_layout.setSpacing(14)

        logo = QLabel("YU")
        logo.setObjectName("universityLogo")
        logo.setAlignment(Qt.AlignCenter)
        logo.setFixedSize(58, 58)
        brand_layout.addWidget(logo)

        brand_text = QFrame()
        brand_text_layout = QVBoxLayout(brand_text)
        brand_text_layout.setContentsMargins(0, 0, 0, 0)
        brand_text_layout.setSpacing(2)

        university = QLabel("长江大学")
        university.setObjectName("universityName")
        brand_text_layout.addWidget(university)

        university_en = QLabel("YANGTZE UNIVERSITY")
        university_en.setObjectName("universityEn")
        brand_text_layout.addWidget(university_en)
        brand_layout.addWidget(brand_text)
        brand_layout.addStretch()
        left_layout.addWidget(brand)

        left_layout.addSpacing(36)

        badge = QLabel("UNDERGRADUATE THESIS DEMO")
        badge.setObjectName("badge")
        left_layout.addWidget(badge)

        title = QLabel("QPE-HViT\n可视化演示系统")
        title.setObjectName("heroTitle")
        title.setWordWrap(True)
        left_layout.addWidget(title)

        accent = QFrame()
        accent.setObjectName("accentLine")
        accent.setFixedSize(64, 4)
        left_layout.addWidget(accent)

        subtitle = QLabel(
            "围绕理论方法、性能对比、消融实验和可视化实验四个模块，"
            "为本科毕业设计答辩提供结构清晰、便于讲解与切换的数据展示界面。"
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        left_layout.addWidget(subtitle)

        stats = QFrame()
        stats.setObjectName("statsBar")
        stats_layout = QHBoxLayout(stats)
        stats_layout.setContentsMargins(0, 4, 0, 4)
        stats_layout.setSpacing(14)
        for value, label in [("04", "核心模块"), ("08", "图表页面"), ("0", "现场推理")]:
            stats_layout.addWidget(self._make_stat(value, label))
        stats_layout.addStretch()
        left_layout.addWidget(stats)

        left_layout.addSpacing(16)

        feature_items = [
            ("01", "支持理论框图、实验表格和可视化图片统一展示"),
            ("02", "支持静态展示实验指标，无需现场推理"),
            ("03", "支持后续替换论文表格数字、方法框图与可视化图片"),
            ("04", "页面结构对应答辩讲解顺序，现场切换更自然"),
        ]
        for index, text in feature_items:
            left_layout.addWidget(self._make_feature(index, text))

        left_layout.addStretch()
        root_layout.addWidget(left_panel, 6)

        right_panel = QFrame()
        right_panel.setObjectName("loginCard")
        right_panel.setFixedWidth(420)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(42, 42, 42, 34)
        right_layout.setSpacing(18)

        card_shadow = QGraphicsDropShadowEffect(self)
        card_shadow.setBlurRadius(34)
        card_shadow.setOffset(0, 18)
        card_shadow.setColor(QColor(34, 76, 136, 70))
        right_panel.setGraphicsEffect(card_shadow)

        header_label = QLabel("WELCOME")
        header_label.setObjectName("formKicker")
        header_label.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(header_label)

        login_title = QLabel("欢迎登录")
        login_title.setObjectName("loginTitle")
        login_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(login_title)

        hint = QLabel("请输入账号和密码，进入毕业设计演示界面。")
        hint.setObjectName("loginHint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        right_layout.addSpacing(12)

        username_block, self.username = self._make_input_block("账号", "请输入用户名", "ID")
        right_layout.addWidget(username_block)

        password_block, self.password = self._make_input_block("密码", "请输入密码", "PW", password=True)
        right_layout.addWidget(password_block)

        option_row = QFrame()
        option_row.setObjectName("optionRow")
        option_layout = QHBoxLayout(option_row)
        option_layout.setContentsMargins(0, 0, 0, 0)
        option_layout.setSpacing(8)

        self.remember_box = QCheckBox("记住账号")
        self.remember_box.setObjectName("rememberBox")
        option_layout.addWidget(self.remember_box)
        option_layout.addStretch()

        forgot_btn = QToolButton()
        forgot_btn.setObjectName("forgotButton")
        forgot_btn.setText("忘记密码？")
        forgot_btn.setCursor(Qt.PointingHandCursor)
        forgot_btn.clicked.connect(lambda: self.show_error("本地演示账号可在 main.py 中查看或修改。"))
        option_layout.addWidget(forgot_btn)
        right_layout.addWidget(option_row)

        self.error_label = QLabel("")
        self.error_label.setObjectName("errorText")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        right_layout.addWidget(self.error_label)

        self.login_btn = QPushButton("进入演示系统    >")
        self.login_btn.setObjectName("loginButton")
        self.login_btn.setCursor(Qt.PointingHandCursor)
        self.login_btn.setDefault(True)
        right_layout.addWidget(self.login_btn)

        note = QLabel("默认账号用于本地演示，可在 main.py 中自行修改。")
        note.setObjectName("loginNote")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(note)

        right_layout.addStretch()

        footer = QLabel("QPE-HViT VISUAL PRESENTATION")
        footer.setObjectName("cardFooter")
        footer.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(footer)

        root_layout.addWidget(right_panel, 4, Qt.AlignVCenter | Qt.AlignRight)

        self.username.returnPressed.connect(self.password.setFocus)
        self.password.returnPressed.connect(self.login_btn.click)
        self.username.textChanged.connect(self.clear_error)
        self.password.textChanged.connect(self.clear_error)
        self._apply_styles()

    def _load_background(self):
        for file_name in ("background2.png", "background.png", "background3.jpg"):
            pixmap = QPixmap(str(BASE_DIR / file_name))
            if not pixmap.isNull():
                return pixmap
        return QPixmap()

    def _make_stat(self, value, label):
        item = QFrame()
        item.setObjectName("statItem")
        layout = QVBoxLayout(item)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        value_label = QLabel(value)
        value_label.setObjectName("statValue")
        layout.addWidget(value_label)

        text_label = QLabel(label)
        text_label.setObjectName("statLabel")
        layout.addWidget(text_label)
        return item

    def _make_feature(self, index, text):
        row = QFrame()
        row.setObjectName("featureRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        marker = QLabel(index)
        marker.setObjectName("featureMarker")
        marker.setAlignment(Qt.AlignCenter)
        marker.setFixedSize(38, 38)
        layout.addWidget(marker)

        label = QLabel(text)
        label.setObjectName("introText")
        label.setWordWrap(True)
        layout.addWidget(label, 1)
        return row

    def _make_input_block(self, label_text, placeholder, icon_text, password=False):
        block = QFrame()
        block.setObjectName("inputBlock")
        layout = QVBoxLayout(block)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        label = QLabel(label_text)
        label.setObjectName("inputLabel")
        layout.addWidget(label)

        field = QFrame()
        field.setObjectName("fieldFrame")
        field_layout = QHBoxLayout(field)
        field_layout.setContentsMargins(16, 0, 14, 0)
        field_layout.setSpacing(10)

        icon = QLabel(icon_text)
        icon.setObjectName("fieldIcon")
        icon.setAlignment(Qt.AlignCenter)
        icon.setFixedWidth(32)
        field_layout.addWidget(icon)

        line_edit = QLineEdit()
        line_edit.setObjectName("formInput")
        line_edit.setPlaceholderText(placeholder)
        line_edit.setFrame(False)
        line_edit.setMinimumHeight(54)
        if password:
            line_edit.setEchoMode(QLineEdit.Password)
        field_layout.addWidget(line_edit, 1)

        if password:
            self.password_toggle = QToolButton()
            self.password_toggle.setObjectName("passwordToggle")
            self.password_toggle.setText("显示")
            self.password_toggle.setCheckable(True)
            self.password_toggle.setCursor(Qt.PointingHandCursor)
            self.password_toggle.toggled.connect(self._toggle_password_visibility)
            field_layout.addWidget(self.password_toggle)

        layout.addWidget(field)
        return block, line_edit

    def _apply_styles(self):
        self.setStyleSheet(
            """
            QWidget#loginRoot {
                font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                color: #10233f;
            }
            QFrame#introPanel {
                background: transparent;
                border: none;
            }
            QFrame#brandBlock {
                background: transparent;
            }
            QLabel#universityLogo {
                color: #eaf5ff;
                background-color: rgba(255, 255, 255, 0.10);
                border: 1px solid rgba(185, 221, 255, 0.42);
                border-radius: 29px;
                font-size: 15px;
                font-weight: 800;
                letter-spacing: 1px;
            }
            QLabel#universityName {
                color: #f5fbff;
                font-size: 28px;
                font-weight: 800;
                letter-spacing: 2px;
            }
            QLabel#universityEn {
                color: rgba(225, 241, 255, 0.80);
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 2px;
            }
            QLabel#badge {
                color: #62d3ff;
                font-size: 15px;
                font-weight: 800;
                letter-spacing: 2px;
            }
            QLabel#heroTitle {
                color: #f8fbff;
                font-size: 44px;
                font-weight: 900;
                line-height: 1.08;
                letter-spacing: 0px;
            }
            QFrame#accentLine {
                background-color: #23c2f2;
                border-radius: 2px;
            }
            QLabel#heroSubtitle {
                color: rgba(232, 242, 255, 0.88);
                font-size: 15px;
                line-height: 1.75;
            }
            QFrame#statItem {
                border-left: 1px solid rgba(126, 194, 244, 0.32);
                padding-left: 12px;
                min-width: 88px;
            }
            QLabel#statValue {
                color: #ffffff;
                font-size: 24px;
                font-weight: 900;
            }
            QLabel#statLabel {
                color: rgba(217, 235, 255, 0.74);
                font-size: 12px;
            }
            QFrame#featureRow {
                background-color: transparent;
            }
            QLabel#featureMarker {
                color: #eff8ff;
                background-color: rgba(28, 130, 213, 0.78);
                border: 1px solid rgba(142, 216, 255, 0.60);
                border-radius: 19px;
                font-size: 12px;
                font-weight: 800;
            }
            QLabel#introText {
                color: rgba(240, 247, 255, 0.91);
                font-size: 14px;
                line-height: 1.65;
            }
            QFrame#loginCard {
                background-color: rgba(248, 251, 255, 0.94);
                border: 1px solid rgba(255, 255, 255, 0.94);
                border-radius: 28px;
            }
            QLabel#formKicker {
                color: #4f88cf;
                font-size: 11px;
                font-weight: 900;
                letter-spacing: 3px;
            }
            QLabel#loginTitle {
                color: #12396f;
                font-size: 34px;
                font-weight: 900;
                letter-spacing: 0px;
            }
            QLabel#loginHint {
                color: #7487a1;
                font-size: 14px;
                line-height: 1.55;
            }
            QLabel#inputLabel {
                color: #36516f;
                font-size: 13px;
                font-weight: 800;
            }
            QFrame#fieldFrame {
                background-color: rgba(255, 255, 255, 0.96);
                border: 1px solid rgba(193, 209, 229, 0.95);
                border-radius: 16px;
            }
            QFrame#fieldFrame:hover {
                border: 1px solid rgba(77, 147, 226, 0.95);
            }
            QLabel#fieldIcon {
                color: #8aa1b8;
                background-color: rgba(227, 238, 250, 0.72);
                border-radius: 10px;
                font-size: 11px;
                font-weight: 900;
                min-height: 32px;
            }
            QLineEdit#formInput {
                color: #1d314d;
                border: none;
                background: transparent;
                selection-background-color: #2f80ed;
                font-size: 15px;
            }
            QLineEdit#formInput:focus {
                color: #10233f;
            }
            QLineEdit#formInput::placeholder {
                color: #9eacbb;
            }
            QCheckBox#rememberBox {
                color: #5d718a;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox#rememberBox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 4px;
                border: 1px solid rgba(141, 163, 190, 0.9);
                background-color: rgba(255, 255, 255, 0.88);
            }
            QCheckBox#rememberBox::indicator:checked {
                background-color: #2f80ed;
                border: 1px solid #2f80ed;
            }
            QToolButton#forgotButton,
            QToolButton#passwordToggle {
                color: #4d8ee6;
                border: none;
                background: transparent;
                font-size: 13px;
                font-weight: 700;
                padding: 4px 2px;
            }
            QToolButton#forgotButton:hover,
            QToolButton#passwordToggle:hover {
                color: #215fb6;
            }
            QLabel#errorText {
                color: #bf3d3d;
                background-color: rgba(255, 235, 235, 0.82);
                border: 1px solid rgba(235, 165, 165, 0.70);
                border-radius: 10px;
                padding: 8px 10px;
                font-size: 13px;
            }
            QPushButton#loginButton {
                min-height: 58px;
                border-radius: 17px;
                border: none;
                color: #ffffff;
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #2358e6,
                    stop: 0.55 #2f80ed,
                    stop: 1 #49a7f5
                );
                font-size: 16px;
                font-weight: 900;
                letter-spacing: 0px;
            }
            QPushButton#loginButton:hover {
                background: qlineargradient(
                    x1: 0, y1: 0, x2: 1, y2: 0,
                    stop: 0 #1f4ed0,
                    stop: 0.55 #2773da,
                    stop: 1 #3f99e6
                );
            }
            QPushButton#loginButton:pressed {
                padding-top: 2px;
                background-color: #1f63ce;
            }
            QLabel#loginNote {
                color: #6b85b1;
                font-size: 13px;
                line-height: 1.6;
            }
            QLabel#cardFooter {
                color: rgba(86, 113, 153, 0.38);
                font-size: 11px;
                font-weight: 900;
                letter-spacing: 2px;
            }
            """
        )

    def _toggle_password_visibility(self, visible):
        self.password.setEchoMode(QLineEdit.Normal if visible else QLineEdit.Password)
        self.password_toggle.setText("隐藏" if visible else "显示")

    def _load_saved_preferences(self):
        settings = QSettings("QPE-HViT", "VisualizationDemo")
        saved_user = settings.value("login/username", "", str)
        remember = settings.value("login/remember", False, bool)
        self.remember_box.setChecked(remember)
        if remember and saved_user:
            self.username.setText(saved_user)
            self.password.setFocus()
        else:
            self.username.setFocus()

    def save_preferences(self):
        settings = QSettings("QPE-HViT", "VisualizationDemo")
        settings.setValue("login/remember", self.remember_box.isChecked())
        settings.setValue("login/username", self.username.text().strip() if self.remember_box.isChecked() else "")

    def show_error(self, message):
        self.error_label.setText(message)
        self.error_label.setVisible(True)

    def clear_error(self):
        if self.error_label.isVisible():
            self.error_label.clear()
            self.error_label.setVisible(False)

    def resizeEvent(self, event):
        self._prepare_background()
        super().resizeEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)

        rect = self.rect()
        self._paint_base_gradient(painter, rect)
        self._paint_background_texture(painter)
        self._paint_data_globe(painter, rect)
        self._paint_bottom_wave(painter, rect)
        painter.end()

    def _prepare_background(self):
        if self.background_pixmap.isNull() or self.size().isEmpty():
            self._scaled_background = QPixmap()
            return
        self._scaled_background = self.background_pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatioByExpanding,
            Qt.SmoothTransformation,
        )

    def _advance_background(self):
        self._phase = (self._phase + 0.035) % math.tau
        self.update()

    def _paint_base_gradient(self, painter, rect):
        base = QLinearGradient(QPointF(0, 0), QPointF(rect.width(), rect.height()))
        base.setColorAt(0.00, QColor("#041631"))
        base.setColorAt(0.42, QColor("#08284d"))
        base.setColorAt(0.68, QColor("#255b91"))
        base.setColorAt(1.00, QColor("#edf5ff"))
        painter.fillRect(rect, QBrush(base))

        left_vignette = QRadialGradient(
            QPointF(rect.width() * 0.20, rect.height() * 0.42),
            rect.width() * 0.62,
        )
        left_vignette.setColorAt(0.0, QColor(12, 70, 125, 145))
        left_vignette.setColorAt(0.7, QColor(6, 24, 52, 70))
        left_vignette.setColorAt(1.0, QColor(4, 13, 31, 0))
        painter.fillRect(rect, QBrush(left_vignette))

        right_wash = QLinearGradient(QPointF(rect.width() * 0.56, 0), QPointF(rect.width(), rect.height()))
        right_wash.setColorAt(0.00, QColor(255, 255, 255, 0))
        right_wash.setColorAt(0.72, QColor(232, 241, 255, 178))
        right_wash.setColorAt(1.00, QColor(248, 251, 255, 230))
        painter.fillRect(rect, QBrush(right_wash))

    def _paint_background_texture(self, painter):
        if self._scaled_background.isNull():
            return
        painter.save()
        painter.setOpacity(0.22)
        painter.drawPixmap(0, 0, self._scaled_background)
        painter.restore()

    def _paint_data_globe(self, painter, rect):
        center = QPointF(rect.width() * 0.58, rect.height() * 0.51)
        radius = min(rect.width(), rect.height()) * 0.33

        halo = QRadialGradient(center, radius * 1.35)
        halo.setColorAt(0.0, QColor(57, 185, 255, 70))
        halo.setColorAt(0.45, QColor(30, 123, 218, 32))
        halo.setColorAt(1.0, QColor(20, 60, 120, 0))
        painter.fillRect(rect, QBrush(halo))

        orbit_pen = QPen(QColor(97, 202, 255, 54), 1.3)
        painter.setPen(orbit_pen)
        for index, scale in enumerate((0.78, 0.95, 1.12, 1.30)):
            orbit = QRectF(
                center.x() - radius * scale,
                center.y() - radius * scale * 0.62,
                radius * 2 * scale,
                radius * 1.24 * scale,
            )
            painter.drawArc(orbit, int((18 + index * 20 + self._phase * 16) * 16), int(235 * 16))

        grid_pen = QPen(QColor(147, 219, 255, 44), 1)
        grid_pen.setStyle(Qt.DotLine)
        painter.setPen(grid_pen)
        painter.drawEllipse(center, radius * 0.72, radius * 0.72)
        for scale in (0.28, 0.48, 0.64):
            painter.drawEllipse(center, radius * scale, radius * 0.72)

        painter.setPen(QPen(QColor(129, 219, 255, 76), 1))
        for offset in (-0.46, -0.23, 0.0, 0.23, 0.46):
            y = center.y() + radius * offset
            painter.drawLine(QPointF(center.x() - radius * 0.68, y), QPointF(center.x() + radius * 0.68, y))

        node_brush = QBrush(QColor(67, 196, 255, 210))
        painter.setBrush(node_brush)
        painter.setPen(Qt.NoPen)
        nodes = [
            (0.05, 1.22),
            (0.74, 1.06),
            (1.42, 1.28),
            (2.26, 1.03),
            (3.18, 0.92),
            (4.24, 1.18),
            (5.22, 0.86),
        ]
        for angle, scale in nodes:
            pulse = 1.0 + math.sin(self._phase * 3 + angle) * 0.22
            point = QPointF(
                center.x() + math.cos(angle + self._phase * 0.15) * radius * scale,
                center.y() + math.sin(angle + self._phase * 0.15) * radius * scale * 0.62,
            )
            painter.drawEllipse(point, 3.6 * pulse, 3.6 * pulse)

        painter.setPen(QPen(QColor(78, 185, 255, 44), 1))
        for angle in (0.34, 1.28, 2.62, 4.82):
            point = QPointF(
                center.x() + math.cos(angle) * radius * 1.18,
                center.y() + math.sin(angle) * radius * 0.73,
            )
            painter.drawLine(center, point)

    def _paint_bottom_wave(self, painter, rect):
        painter.save()
        for layer, alpha in enumerate((72, 48, 32)):
            path = QPainterPath()
            base_y = rect.height() * (0.86 + layer * 0.025)
            path.moveTo(0, base_y)
            step = 28
            for x in range(0, rect.width() + step, step):
                y = base_y + math.sin(x * 0.018 + self._phase + layer * 0.8) * (14 + layer * 4)
                y += math.cos(x * 0.010 + self._phase * 0.6) * 10
                path.lineTo(x, y)
            painter.setPen(QPen(QColor(70, 198, 255, alpha), 1.5))
            painter.drawPath(path)
        painter.restore()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec())
