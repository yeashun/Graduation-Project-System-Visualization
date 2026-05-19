import sys

from PySide6.QtGui import QFont
from PySide6.QtWidgets import QApplication

from login_window import LoginWindow
from thesis_gui import QuantumViTGUI


class Controller:
    """管理登录页与主界面的切换。"""

    def __init__(self):
        self.login = LoginWindow()
        self.main_gui = None
        self.login.login_btn.clicked.connect(self.check_login)
        self.login.show()

    def check_login(self):
        username = self.login.username.text().strip()
        password = self.login.password.text().strip()

        if username == "yeshun" and password == "2022001469":
            self.login.mark_login_pending()
            QApplication.processEvents()
            self.login.save_preferences()
            self.main_gui = QuantumViTGUI()
            self.main_gui.showMaximized()
            self.login.close()
        else:
            self.login.show_error("用户名或密码错误，请重新输入。")
            self.login.password.clear()
            self.login.password.setFocus()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei UI", 10))
    controller = Controller()
    sys.exit(app.exec())
