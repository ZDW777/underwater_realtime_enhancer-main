"""启动默认设置窗口：继承主窗口配色，集中校验后端与分辨率组合。"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QFormLayout, QHBoxLayout,
    QLabel, QSpinBox, QVBoxLayout, QWidget,
)

from config import BACKENDS, RESOLUTIONS
from core.preferences import StartupPreferences


class SettingsDialog(QDialog):
    """只编辑启动偏好；接受后由主窗口保存，当前运行任务不会被中途修改。"""
    def __init__(self, models: list[str], preferences: StartupPreferences, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setObjectName("settingsDialog")
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self.setMinimumWidth(470)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 20)
        layout.setSpacing(16)
        description = QLabel("设置应用启动后的默认选项，保存后立即应用并在下次启动时恢复。")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        form.setSpacing(12)
        self.model_combo = QComboBox()
        self.model_combo.addItems(models)
        index = self.model_combo.findText(preferences.model_name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)
        self.backend_combo = QComboBox()
        for key, label in BACKENDS.items():
            self.backend_combo.addItem(label, key)
        self.backend_combo.setCurrentIndex(self.backend_combo.findData(preferences.backend))
        self.resolution_combo = QComboBox()
        for key, (label, _, _) in RESOLUTIONS.items():
            self.resolution_combo.addItem(label, key)
        self.resolution_combo.setCurrentIndex(self.resolution_combo.findData(preferences.resolution_key))
        form.addRow("默认增强模型", self.model_combo)
        form.addRow("默认推理模式", self.backend_combo)
        form.addRow("默认处理分辨率", self.resolution_combo)
        custom = QWidget()
        custom_layout = QHBoxLayout(custom)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        self.width_spin, self.height_spin = QSpinBox(), QSpinBox()
        for spin, maximum, value, suffix in (
            (self.width_spin, 7680, preferences.width, " W"),
            (self.height_spin, 4320, preferences.height, " H"),
        ):
            spin.setRange(64, maximum)
            spin.setValue(value)
            spin.setSuffix(suffix)
            custom_layout.addWidget(spin)
        form.addRow("自定义宽高", custom)
        layout.addLayout(form)
        note = QLabel("自动匹配仅支持 PyTorch 图像和视频，文件夹中的每个文件分别匹配原尺寸。")
        note.setWordWrap(True)
        note.setObjectName("settingsNote")
        layout.addWidget(note)
        self.buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        self.buttons.button(QDialogButtonBox.StandardButton.Save).setEnabled(bool(models))
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)
        self.backend_combo.currentIndexChanged.connect(self._update_controls)
        self.resolution_combo.currentIndexChanged.connect(self._update_controls)
        self._update_controls()

    def _update_controls(self) -> None:
        """禁止 TensorRT 选择自动匹配，自定义值在该选项下才允许编辑。"""
        allowed = self.backend_combo.currentData() == "pytorch"
        self.resolution_combo.model().item(self.resolution_combo.findData("auto")).setEnabled(allowed)
        if not allowed and self.resolution_combo.currentData() == "auto":
            self.resolution_combo.setCurrentIndex(self.resolution_combo.findData("3840x2160"))
        custom = self.resolution_combo.currentData() == "other"
        self.width_spin.setEnabled(custom)
        self.height_spin.setEnabled(custom)

    def preferences(self) -> StartupPreferences:
        """把已校验的控件值转换成持久化快照。"""
        return StartupPreferences(
            self.model_combo.currentText(), self.backend_combo.currentData(),
            self.resolution_combo.currentData(), self.width_spin.value(), self.height_spin.value(),
        )

    def showEvent(self, event) -> None:  # noqa: N802
        """按主窗口几何中心定位，同时限制在当前屏幕可用区域内。"""
        super().showEvent(event)
        if self.parentWidget() is not None:
            geometry = self.frameGeometry()
            geometry.moveCenter(self.parentWidget().frameGeometry().center())
            available = self.screen().availableGeometry()
            geometry.moveLeft(max(available.left(), min(geometry.left(), available.right() - geometry.width() + 1)))
            geometry.moveTop(max(available.top(), min(geometry.top(), available.bottom() - geometry.height() + 1)))
            self.move(geometry.topLeft())
