import sys
import os
from PyQt5.QtWidgets import QApplication
from threekneegui import threekneeGUI


if __name__ == "__main__":
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms"

    app = QApplication(sys.argv)
    window = threekneeGUI()
    window.show()
    sys.exit(app.exec_())