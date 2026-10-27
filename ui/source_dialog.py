"""在同一个文件选择窗口里选媒体文件或当前文件夹，不增加主界面按钮。"""
from pathlib import Path

from PySide6.QtWidgets import QDialog, QDialogButtonBox, QFileDialog, QPushButton


class SourceDialog(QFileDialog):
    """保留文件浏览操作，并提供明确的“选择当前文件夹”入口。"""
    def __init__(self, parent=None):
        super().__init__(parent, "选择图像、视频或文件夹")
        self.setOption(QFileDialog.Option.DontUseNativeDialog, True)
        self.setFileMode(QFileDialog.FileMode.ExistingFile)
        self.setNameFilters([
            "媒体文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.m4v *.wmv *.mpg *.mpeg *.webm)",
            "所有文件 (*)",
        ])
        self.chosen_path = ""
        self.folder_button = QPushButton("选择当前文件夹", self)
        self.folder_button.clicked.connect(self._choose_directory)
        self.findChild(QDialogButtonBox).addButton(self.folder_button, QDialogButtonBox.ButtonRole.ActionRole)

    def _choose_directory(self) -> None:
        """当前浏览目录就是批处理根目录；双击目录仍用于进入目录。"""
        self.chosen_path = self.directory().absolutePath()
        QDialog.accept(self)

    def accept(self) -> None:
        """普通确认只接受文件，目录选择使用专门入口避免改变导航行为。"""
        paths = self.selectedFiles()
        if paths and Path(paths[0]).is_file():
            self.chosen_path = paths[0]
        super().accept()
