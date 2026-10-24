from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication

from ui.main_window import MainWindow


def main() -> int:
    """创建全局 Qt 应用，并把项目根目录交给主窗口。"""
    app = QApplication(sys.argv)
    app.setApplicationName("Underwater Realtime Enhancer")
    app.setStyle("Fusion")
    # 以 app.py 所在目录作为根目录，避免从不同终端启动时路径失效。
    window = MainWindow(Path(__file__).resolve().parent)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
