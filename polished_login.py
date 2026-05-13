"""
登录界面。
"""
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QPalette, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


class LoginWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("QPE-HViT 可视化演示系统 - 登录")
        self.setMinimumSize(920, 580)
        self.resize(1120, 680)
        self.background_pixmap = QPixmap(str(Path("background3.jpg").resolve()))
        self.init_ui()

    def init_ui(self):
        self.update_background()

        root_layout = QHBoxLayout(self)
        root_layout.setContentsMargins(42, 42, 42, 42)
        root_layout.setSpacing(0)

        left_panel = QFrame()
        left_panel.setObjectName("introPanel")
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(40, 42, 40, 42)
        left_layout.setSpacing(16)

        badge = QLabel("UNDERGRADUATE THESIS VISUAL DEMO")
        badge.setObjectName("badge")
        left_layout.addWidget(badge)

        title = QLabel("QPE-HViT\n毕业设计可视化演示系统")
        title.setObjectName("heroTitle")
        title.setWordWrap(True)
        left_layout.addWidget(title)

        subtitle = QLabel(
            "面向本科毕业设计答辩场景构建的静态展示系统，"
            "集中呈现理论框架、性能对比、消融实验和可视化证据。"
        )
        subtitle.setObjectName("heroSubtitle")
        subtitle.setWordWrap(True)
        left_layout.addWidget(subtitle)

        for text in [
            "• 理论方法板块支持展示整体框架图和关键模块要点。",
            "• 实验结果页面支持直接替换论文表格数据与可视化图片。",
            "• 无需现场推理，演示过程更稳定、更适合答辩节奏。",
        ]:
            item = QLabel(text)
            item.setObjectName("introText")
            item.setWordWrap(True)
            left_layout.addWidget(item)

        left_layout.addStretch()
        root_layout.addWidget(left_panel, 3)

        right_panel = QFrame()
        right_panel.setObjectName("loginCard")
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(40, 40, 40, 40)
        right_layout.setSpacing(18)

        login_title = QLabel("进入演示系统")
        login_title.setObjectName("loginTitle")
        login_title.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(login_title)

        hint = QLabel("请输入账号和密码，进入答辩展示主界面。")
        hint.setObjectName("loginHint")
        hint.setAlignment(Qt.AlignCenter)
        hint.setWordWrap(True)
        right_layout.addWidget(hint)

        self.username = QLineEdit()
        self.username.setPlaceholderText("用户名")
        right_layout.addWidget(self.username)

        self.password = QLineEdit()
        self.password.setPlaceholderText("密码")
        self.password.setEchoMode(QLineEdit.Password)
        right_layout.addWidget(self.password)

        self.login_btn = QPushButton("进入系统")
        self.login_btn.clicked.connect(self.handle_login)
        right_layout.addWidget(self.login_btn)

        note = QLabel("默认账号用于本地演示，可在 main.py 中调整。")
        note.setObjectName("loginHint")
        note.setWordWrap(True)
        note.setAlignment(Qt.AlignCenter)
        right_layout.addWidget(note)

        right_layout.addStretch()
        root_layout.addWidget(right_panel, 2)

        self.setStyleSheet(
            """
            QWidget {
                font-family: "Microsoft YaHei UI";
            }
            #introPanel {
                background-color: rgba(14, 35, 44, 0.86);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-top-left-radius: 30px;
                border-bottom-left-radius: 30px;
            }
            #loginCard {
                background-color: rgba(255, 250, 242, 0.95);
                border: 1px solid rgba(228, 217, 201, 0.95);
                border-top-right-radius: 30px;
                border-bottom-right-radius: 30px;
            }
            #badge {
                color: #e0b76a;
                font-size: 12px;
                font-weight: 700;
                letter-spacing: 1px;
            }
            #heroTitle {
                color: #f5f2ec;
                font-size: 34px;
                font-weight: 700;
            }
            #heroSubtitle {
                color: #cdd9dd;
                font-size: 14px;
                line-height: 1.7;
            }
            #introText {
                color: #dce7ea;
                font-size: 13px;
                line-height: 1.7;
            }
            #loginTitle {
                color: #18303b;
                font-size: 28px;
                font-weight: 700;
            }
            #loginHint {
                color: #6d7c83;
                font-size: 13px;
                line-height: 1.6;
            }
            QLineEdit {
                min-height: 48px;
                padding: 10px 14px;
                border-radius: 14px;
                border: 1px solid #d6c6b0;
                background-color: rgba(255, 255, 255, 0.96);
                font-size: 14px;
            }
            QPushButton {
                min-height: 50px;
                border-radius: 14px;
                border: none;
                background-color: #1d4350;
                color: #ffffff;
                font-size: 15px;
                font-weight: 700;
            }
            QPushButton:hover {
                background-color: #2d5f6d;
            }
            """
        )

    def resizeEvent(self, event):
        self.update_background()
        super().resizeEvent(event)

    def update_background(self):
        if self.background_pixmap.isNull():
            return

        palette = QPalette()
        palette.setBrush(
            QPalette.Window,
            QBrush(
                self.background_pixmap.scaled(
                    self.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation,
                )
            ),
        )
        self.setPalette(palette)
        self.setAutoFillBackground(True)

    def handle_login(self):
        print(f"尝试登录用户：{self.username.text().strip()}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoginWindow()
    window.show()
    sys.exit(app.exec())
