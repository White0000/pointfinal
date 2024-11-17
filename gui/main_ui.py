import sys
from PyQt5.QtWidgets import QApplication
from project_root.gui.gui_controls import MainWindow

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
