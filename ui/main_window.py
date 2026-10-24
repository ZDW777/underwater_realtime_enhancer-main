"""界面与显示层：配置快照、双画面绘制、运行控制和滑动性能统计。

视频使用定时消费；相机使用合并的新帧通知。实际显示 FPS 只统计新帧
进入 paintEvent 的间隔，暂停和任务切换负责清理过期通知及画面。
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
import time

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QImage,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from config import BACKENDS, RESOLUTIONS, RunConfig
from core.camera import CameraInfo
from core.model_registry import ModelRegistry
from core.weight_resolver import WeightResolver
from core.worker import IMAGE_EXTENSIONS, InferenceWorker


class OceanHeader(QWidget):
    """使用 Qt 矢量绘图生成海洋横幅，不依赖任何外部图片资源。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("oceanHeader")
        self.setMinimumHeight(94)
        self.setMaximumHeight(112)
        self.content = QHBoxLayout(self)
        self.content.setContentsMargins(24, 14, 24, 20)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 从深海蓝到青蓝的渐变形成标题背景。
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0.0, QColor("#05233d"))
        gradient.setColorAt(0.55, QColor("#07506a"))
        gradient.setColorAt(1.0, QColor("#087f86"))
        painter.setBrush(gradient)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(self.rect(), 13, 13)

        # 底部两层半透明海浪让横幅有水下流动感。
        for offset, color in ((0, QColor(22, 186, 184, 72)), (8, QColor(72, 218, 205, 48))):
            y = self.height() - 25 + offset
            wave = QPainterPath(QPointF(0, y))
            wave.cubicTo(self.width() * 0.22, y - 16, self.width() * 0.33, y + 13, self.width() * 0.52, y)
            wave.cubicTo(self.width() * 0.69, y - 12, self.width() * 0.83, y + 8, self.width(), y - 5)
            wave.lineTo(self.width(), self.height())
            wave.lineTo(0, self.height())
            wave.closeSubpath()
            painter.fillPath(wave, color)

        # 固定位置的小气泡在不同窗口尺寸下仍保持稳定、清晰。
        bubble_pen = QPen(QColor(165, 244, 239, 105), 1.2)
        painter.setPen(bubble_pen)
        painter.setBrush(QColor(198, 255, 250, 22))
        for x_ratio, y_ratio, radius in ((0.71, 0.28, 6), (0.76, 0.55, 3), (0.82, 0.23, 4), (0.91, 0.52, 7), (0.95, 0.22, 3)):
            x = self.width() * x_ratio
            y = self.height() * y_ratio
            painter.drawEllipse(QPointF(x, y), radius, radius)


class PreviewWidget(QWidget):
    """保持原始宽高比显示 OpenCV 画面，并绘制窗口标题与空状态。"""

    # 只有新帧真正进入paintEvent并完成绘制时才发出，用于统计用户实际
    # 看见的显示FPS，而不是工作线程产帧速度或定时器触发频率。
    frame_presented = Signal()

    def __init__(self, title: str, accent: str, parent=None):
        super().__init__(parent)
        self.title = title
        self.accent = QColor(accent)
        self._source_pixmap: QPixmap | None = None
        self._display_pixmap: QPixmap | None = None
        self._frame_serial = 0
        self._painted_serial = -1
        self.setMinimumSize(360, 235)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_frame(self, bgr: np.ndarray) -> None:
        """直接接收OpenCV BGR图像，避免每帧执行BGR→RGB颜色转换。"""
        if bgr.ndim != 3 or bgr.shape[2] != 3 or bgr.dtype != np.uint8:
            raise ValueError("预览帧必须是HWC排列的uint8三通道BGR图像")
        contiguous_bgr = np.ascontiguousarray(bgr)
        height, width, _ = contiguous_bgr.shape
        image = QImage(
            contiguous_bgr.data,
            width,
            height,
            contiguous_bgr.strides[0],
            QImage.Format.Format_BGR888,
        ).copy()
        # copy()让QImage拥有独立内存，工作线程随后释放numpy数组也不会影响显示。
        self._source_pixmap = QPixmap.fromImage(image)
        self._frame_serial += 1
        self._refresh_display_pixmap()
        self.update()

    def clear_frame(self) -> None:
        self._source_pixmap = None
        self._display_pixmap = None
        self.update()

    def preview_extent(self) -> tuple[int, int]:
        """返回扣除边距后的实际画面区域，供工作线程生成等尺寸缩略图。"""
        return max(1, self.width() - 14), max(1, self.height() - 14)

    def _refresh_display_pixmap(self) -> None:
        """仅在新帧或窗口尺寸变化时缩放，paintEvent只负责轻量绘制。"""
        if self._source_pixmap is None:
            self._display_pixmap = None
            return
        available_width, available_height = self.preview_extent()
        if (
            self._source_pixmap.width() <= available_width
            and self._source_pixmap.height() <= available_height
        ):
            # 工作线程已按控件尺寸生成缩略图时直接复用，避免重复缩放。
            self._display_pixmap = self._source_pixmap
        else:
            self._display_pixmap = self._source_pixmap.scaled(
                available_width,
                available_height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._refresh_display_pixmap()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 预览区使用深海渐变，视频黑边不会显得突兀。
        background = QLinearGradient(0, 0, 0, self.height())
        background.setColorAt(0, QColor("#071726"))
        background.setColorAt(1, QColor("#04101c"))
        painter.setBrush(background)
        painter.setPen(QPen(QColor("#21445a"), 1))
        painter.drawRoundedRect(self.rect().adjusted(0, 0, -1, -1), 10, 10)

        presented_new_frame = False
        if self._display_pixmap is not None:
            x = (self.width() - self._display_pixmap.width()) // 2
            y = (self.height() - self._display_pixmap.height()) // 2
            painter.drawPixmap(x, y, self._display_pixmap)
            if self._painted_serial != self._frame_serial:
                self._painted_serial = self._frame_serial
                presented_new_frame = True
        else:
            painter.setPen(QColor("#5e8295"))
            placeholder_font = QFont()
            placeholder_font.setPointSize(12)
            painter.setFont(placeholder_font)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "◌  等待画面")

        # 左上角标题条始终绘制在图像之上。
        badge = QRectF(14, 14, 105, 34)
        painter.setBrush(QColor(3, 18, 30, 218))
        painter.setPen(QPen(self.accent, 1))
        painter.drawRoundedRect(badge, 7, 7)
        painter.setBrush(self.accent)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(30, 31), 4, 4)
        painter.setPen(QColor("#eefcff"))
        font = QFont()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRectF(42, 14, 70, 34), Qt.AlignmentFlag.AlignVCenter, self.title)
        if presented_new_frame:
            self.frame_presented.emit()


class MetricCard(QFrame):
    """用于显示单项实时性能数据的小卡片。"""

    def __init__(self, icon: str, title: str, unit: str, accent: str):
        super().__init__()
        self.setObjectName("metricCard")
        self.setProperty("accent", accent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 10, 15, 10)
        layout.setSpacing(11)
        icon_label = QLabel(icon)
        icon_label.setObjectName("metricIcon")
        icon_label.setStyleSheet(f"color: {accent};")
        layout.addWidget(icon_label)

        text_box = QVBoxLayout()
        text_box.setSpacing(0)
        caption = QLabel(title)
        caption.setObjectName("metricCaption")
        self.value = QLabel(f"-- {unit}")
        self.value.setObjectName("metricValue")
        text_box.addWidget(caption)
        text_box.addWidget(self.value)
        layout.addLayout(text_box, 1)

    def set_value(self, value: float, unit: str) -> None:
        # 一位小数足以反映实时性能，也能避免无意义的小数位持续跳动。
        self.value.setText(f"{value:.1f} {unit}")

    def clear_value(self, unit: str) -> None:
        self.value.setText(f"-- {unit}")


class MainWindow(QMainWindow):
    """主窗口：组织输入、模型配置、实时画面、性能指标和运行控制。"""

    def __init__(self, root: Path):
        super().__init__()
        self.root = root
        self.registry = ModelRegistry(root / "models")
        self.resolver = WeightResolver(root / "weights")
        self.worker: InferenceWorker | None = None
        self.source: str | int | None = None
        self.source_kind: str | None = None
        # 自定义宽高只有在用户点击“确认尺寸”后才生效。输入过程中保持
        # None，防止每次 QSpinBox 数值变化都解析路径并创建临时目录。
        self._confirmed_custom_size: tuple[int, int] | None = None
        # 仅当用户主动点击红色“结束”按钮时置为 True，用于忽略工作线程
        # 已经排队但尚未送达主线程的最后几帧，确保预览区保持等待状态。
        self._clear_previews_on_finish = False
        self._playback_active = False
        self._playback_paused = False
        self._source_fps = 30.0
        self._last_display_time: float | None = None
        self._display_intervals: deque[float] = deque(maxlen=60)

        self.setWindowTitle("水下图像 / 视频实时增强")
        screen = self.screen().availableGeometry()
        # 默认占用约九成可用屏幕，同时给系统任务栏保留空间。
        self.resize(int(screen.width() * 0.92), int(screen.height() * 0.90))
        self._build_ui()
        # 视频显示定时器只从有界缓冲区取帧；相机走合并的新帧通知。
        # 指标定时器每200 ms读取一次滑动窗口快照，不随原始采集逐帧触发。
        self._display_timer = QTimer(self)
        self._display_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._display_timer.timeout.connect(self._display_next_frame)
        self._metrics_timer = QTimer(self)
        self._metrics_timer.setTimerType(Qt.TimerType.PreciseTimer)
        self._metrics_timer.setInterval(200)
        self._metrics_timer.timeout.connect(self._show_metrics)
        self._apply_style()
        self.refresh_models()
        self._update_resolution_controls()
        self._update_summary()

    def _build_ui(self) -> None:
        """构建由上到下的海洋横幅、配置区、指标区、预览区和控制区。"""
        central = QWidget()
        central.setObjectName("central")
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(16, 14, 16, 14)
        page.setSpacing(10)

        # 标题横幅。
        header = OceanHeader()
        title_box = QVBoxLayout()
        title_box.setSpacing(1)
        title = QLabel("水下图像 / 视频实时增强工作台")
        title.setObjectName("title")
        subtitle = QLabel("UNDERWATER VISION  ·  REAL-TIME ENHANCEMENT")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.content.addLayout(title_box)
        header.content.addStretch()
        self.status_badge = QLabel("●  就绪")
        self.status_badge.setObjectName("statusBadge")
        header.content.addWidget(self.status_badge, 0, Qt.AlignmentFlag.AlignVCenter)
        page.addWidget(header)

        # 配置区域左右分栏，避免所有控件挤在一条长横线上。
        configuration = QHBoxLayout()
        configuration.setSpacing(10)

        source_group = QGroupBox("01  输入源")
        source_layout = QVBoxLayout(source_group)
        source_layout.setContentsMargins(14, 18, 14, 12)
        self.source_label = QLineEdit()
        self.source_label.setReadOnly(True)
        self.source_label.setPlaceholderText("请选择图像、视频或摄像头")
        source_layout.addWidget(self.source_label)
        source_buttons = QHBoxLayout()
        self.open_file_button = QPushButton("▣  选择图像 / 视频")
        self.camera_button = QPushButton("◉  使用摄像头 0")
        source_buttons.addWidget(self.open_file_button)
        source_buttons.addWidget(self.camera_button)
        source_layout.addLayout(source_buttons)
        configuration.addWidget(source_group, 2)

        inference_group = QGroupBox("02  模型与推理")
        inference_grid = QGridLayout(inference_group)
        inference_grid.setContentsMargins(14, 18, 14, 12)
        inference_grid.setHorizontalSpacing(9)
        inference_grid.setVerticalSpacing(7)

        self.model_combo = QComboBox()
        self.refresh_button = QPushButton("↻  刷新")
        self.backend_combo = QComboBox()
        for key, text in BACKENDS.items():
            self.backend_combo.addItem(text, key)
        self.resolution_combo = QComboBox()
        for key, (text, _, _) in RESOLUTIONS.items():
            self.resolution_combo.addItem(text, key)
        self.confirm_size_button = QPushButton("✓  确认尺寸")
        self.confirm_size_button.setToolTip(
            "输入完整的自定义宽高后点击确认，届时才解析权重路径并创建目录。"
        )
        self.width_spin = QSpinBox()
        self.width_spin.setRange(64, 7680)
        self.width_spin.setValue(1920)
        self.width_spin.setSuffix(" W")
        self.height_spin = QSpinBox()
        self.height_spin.setRange(64, 4320)
        self.height_spin.setValue(1080)
        self.height_spin.setSuffix(" H")

        inference_grid.addWidget(QLabel("增强模型"), 0, 0)
        inference_grid.addWidget(self.model_combo, 0, 1)
        inference_grid.addWidget(self.refresh_button, 0, 2)
        inference_grid.addWidget(QLabel("推理模式"), 0, 3)
        inference_grid.addWidget(self.backend_combo, 0, 4)
        inference_grid.addWidget(QLabel("处理分辨率"), 1, 0)
        # “确认尺寸”与上一行“刷新”位于同一列，且只在自定义模式下可用。
        inference_grid.addWidget(self.resolution_combo, 1, 1)
        inference_grid.addWidget(self.confirm_size_button, 1, 2)
        inference_grid.addWidget(self.width_spin, 1, 3)
        inference_grid.addWidget(self.height_spin, 1, 4)
        self.weight_hint = QLabel()
        self.weight_hint.setObjectName("weightHint")
        self.weight_hint.setWordWrap(True)
        inference_grid.addWidget(self.weight_hint, 2, 0, 1, 5)
        configuration.addWidget(inference_group, 3)
        page.addLayout(configuration)

        # 保存设置单独成条，路径较长时仍有充足显示空间。
        save_bar = QFrame()
        save_bar.setObjectName("saveBar")
        save_layout = QHBoxLayout(save_bar)
        save_layout.setContentsMargins(13, 7, 13, 7)
        self.save_checkbox = QCheckBox("保存增强结果")
        self.output_dir = QLineEdit(str(self.root / "outputs"))
        self.output_dir.setEnabled(False)
        self.output_button = QPushButton("选择输出目录")
        self.output_button.setEnabled(False)
        save_layout.addWidget(self.save_checkbox)
        save_layout.addWidget(self.output_dir, 1)
        save_layout.addWidget(self.output_button)
        page.addWidget(save_bar)

        # 三个指标横向等宽排布，便于快速比较模型性能。
        metrics = QHBoxLayout()
        metrics.setSpacing(10)
        self.inference_card = MetricCard("≈", "GPU 推理耗时", "ms", "#57e3d2")
        self.inference_fps_card = MetricCard("◉", "GPU 推理帧率", "FPS", "#55bff2")
        self.total_fps_card = MetricCard("↗", "实际显示帧率", "FPS", "#8ee89a")
        metrics.addWidget(self.inference_card)
        metrics.addWidget(self.inference_fps_card)
        metrics.addWidget(self.total_fps_card)
        page.addLayout(metrics)

        # 双画面是窗口的主要视觉区域，使用 stretch=1 获取剩余空间。
        previews = QHBoxLayout()
        previews.setSpacing(10)
        self.original_view = PreviewWidget("Original", "#5bbce4")
        self.enhanced_view = PreviewWidget("Enhanced", "#55e0c4")
        self.enhanced_view.frame_presented.connect(self._record_display_timestamp)
        previews.addWidget(self.original_view, 1)
        previews.addWidget(self.enhanced_view, 1)
        page.addLayout(previews, 1)

        # 底部控制台同时显示操作按钮和当前模型摘要。
        footer = QFrame()
        footer.setObjectName("controlBar")
        footer_layout = QHBoxLayout(footer)
        footer_layout.setContentsMargins(12, 8, 12, 8)
        self.start_button = QPushButton("▶  开始增强")
        self.start_button.setObjectName("startButton")
        self.pause_button = QPushButton("Ⅱ  暂停")
        self.pause_button.setObjectName("pauseButton")
        self.pause_button.setEnabled(False)
        self.stop_button = QPushButton("■  结束")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        footer_layout.addWidget(self.start_button)
        footer_layout.addWidget(self.pause_button)
        footer_layout.addWidget(self.stop_button)
        self.summary = QLabel()
        self.summary.setObjectName("summary")
        self.summary.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        footer_layout.addWidget(self.summary, 1)
        page.addWidget(footer)

        # 所有交互信号统一在此连接，后续调整布局不会影响处理逻辑。
        self.open_file_button.clicked.connect(self._choose_file)
        self.camera_button.clicked.connect(self._choose_camera)
        self.refresh_button.clicked.connect(self.refresh_models)
        self.resolution_combo.currentIndexChanged.connect(self._on_resolution_changed)
        self.confirm_size_button.clicked.connect(self._confirm_custom_resolution)
        self.model_combo.currentIndexChanged.connect(self._update_summary)
        self.backend_combo.currentIndexChanged.connect(self._update_summary)
        self.width_spin.valueChanged.connect(self._on_custom_size_edited)
        self.height_spin.valueChanged.connect(self._on_custom_size_edited)
        self.save_checkbox.toggled.connect(self.output_dir.setEnabled)
        self.save_checkbox.toggled.connect(self.output_button.setEnabled)
        self.output_button.clicked.connect(self._choose_output_dir)
        self.start_button.clicked.connect(self.start_processing)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_processing)

    def _apply_style(self) -> None:
        """集中设置深海蓝、青绿高光和清晰的控件层级。"""
        self.setStyleSheet("""
            QMainWindow { background: #06111d; }
            QWidget { color: #dcecf4; font-size: 13px; font-family: "Microsoft YaHei UI", "Segoe UI"; }
            QWidget#central { background: #071522; }
            QLabel#title { color: #f4ffff; font-size: 24px; font-weight: 700; background: transparent; }
            QLabel#subtitle { color: #8edbd8; font-size: 10px; font-weight: 600; letter-spacing: 1px; background: transparent; }
            QLabel#statusBadge { color: #bffcf1; background: rgba(3, 30, 43, 190);
                border: 1px solid rgba(109, 235, 219, 130); border-radius: 14px;
                padding: 6px 15px; font-weight: 600; }
            QGroupBox { background: #0b1d2c; border: 1px solid #1b4256; border-radius: 9px;
                margin-top: 9px; font-weight: 600; color: #90c8d5; }
            QGroupBox::title { subcontrol-origin: margin; left: 13px; padding: 0 6px; background: #0b1d2c; }
            QComboBox, QLineEdit, QSpinBox { background: #0a1927; color: #e8f6fb;
                border: 1px solid #285066; border-radius: 6px; padding: 7px 9px; min-height: 19px; }
            QComboBox:hover, QLineEdit:hover, QSpinBox:hover { border-color: #3b8796; }
            QComboBox:focus, QLineEdit:focus, QSpinBox:focus { border: 1px solid #40bfb6; }
            QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled {
                color: #4f6b78; background: #091520; border-color: #183344; }
            QComboBox QAbstractItemView { background: #0b1d2c; color: #e5f6fb;
                selection-background-color: #126778; border: 1px solid #28566b; }
            QPushButton { background: #102b3d; color: #d9edf5; border: 1px solid #2c566d;
                border-radius: 6px; padding: 8px 13px; font-weight: 600; }
            QPushButton:hover { background: #164158; border-color: #3d8797; color: white; }
            QPushButton:pressed { background: #0e6070; }
            QPushButton:disabled { color: #506775; background: #0b1924; border-color: #1b3443; }
            QLabel#weightHint { color: #72b8c5; background: #081824; border-left: 3px solid #1c8c91;
                border-radius: 4px; padding: 5px 8px; font-size: 11px; }
            QFrame#saveBar, QFrame#controlBar { background: #0a1b29; border: 1px solid #1c3d50; border-radius: 8px; }
            QFrame#metricCard { background: #0b1e2e; border: 1px solid #1e465a; border-radius: 9px; }
            QFrame#metricCard:hover { background: #0d2536; border-color: #2a6373; }
            QLabel#metricIcon { background: #0a1825; border-radius: 19px; font-size: 23px;
                font-weight: 700; min-width: 40px; min-height: 40px; qproperty-alignment: AlignCenter; }
            QLabel#metricCaption { color: #759cae; font-size: 11px; }
            QLabel#metricValue { color: #e9fbff; font-size: 20px; font-weight: 700; }
            QPushButton#startButton { background: #087a72; border-color: #18b5a8; color: white; min-width: 100px; }
            QPushButton#startButton:hover { background: #0a978a; }
            QPushButton#pauseButton { min-width: 74px; }
            QPushButton#stopButton { background: #512d3b; border-color: #8b4a60; color: #f4dce5; min-width: 74px; }
            QPushButton#stopButton:hover { background: #6b3447; }
            QLabel#summary { color: #85aebb; font-size: 12px; padding-right: 4px; }
            QCheckBox { spacing: 8px; color: #b9d8e1; }
            QCheckBox::indicator { width: 16px; height: 16px; }
            QCheckBox::indicator:unchecked { background: #081722; border: 1px solid #31576b; border-radius: 3px; }
            QCheckBox::indicator:checked { background: #19a99a; border: 1px solid #50d9c8; border-radius: 3px; }
            QStatusBar { color: #81aeba; background: #06111d; }
        """)

    def refresh_models(self) -> None:
        """重新扫描 models 目录，并尽量保留用户之前选择的模型。"""
        previous = self.model_combo.currentText()
        models = self.registry.scan()
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        self.model_combo.addItems(models.keys())
        previous_index = self.model_combo.findText(previous)
        if previous_index >= 0:
            self.model_combo.setCurrentIndex(previous_index)
        self.model_combo.blockSignals(False)
        if self.registry.errors:
            details = "；".join(f"{name}: {error}" for name, error in self.registry.errors.items())
            self.statusBar().showMessage(f"部分模型已跳过：{details}", 15000)
        self._update_summary()

    def _current_size(self) -> tuple[str, int, int]:
        """把标准分辨率或自定义输入统一转换为 key、宽、高。"""
        key = self.resolution_combo.currentData()
        if key == "other":
            return key, self.width_spin.value(), self.height_spin.value()
        _, width, height = RESOLUTIONS[key]
        return key, width, height

    def _update_resolution_controls(self) -> None:
        # 只有选择“其他”时才允许编辑宽高，标准模式避免误输入。
        is_custom = self.resolution_combo.currentData() == "other"
        self.width_spin.setEnabled(is_custom)
        self.height_spin.setEnabled(is_custom)
        self.confirm_size_button.setEnabled(is_custom)
        self._update_summary()

    def _on_resolution_changed(self) -> None:
        """切换分辨率模式后使旧的自定义尺寸确认失效。"""
        self._confirmed_custom_size = None
        self._update_resolution_controls()

    def _on_custom_size_edited(self) -> None:
        """输入过程只更新提示，不解析权重路径，也不创建分辨率目录。"""
        if self.resolution_combo.currentData() != "other":
            return
        self._confirmed_custom_size = None
        self._update_summary()

    def _confirm_custom_resolution(self) -> None:
        """确认完整宽高；路径解析产生的目录创建从此时才允许发生。"""
        key, width, height = self._current_size()
        if key != "other":
            return
        self._confirmed_custom_size = (width, height)
        self._update_summary()
        self.statusBar().showMessage(
            f"已确认自定义处理分辨率：{width}×{height}",
            5000,
        )

    def _update_summary(self) -> None:
        """同步刷新底部摘要和当前设置对应的完整权重相对路径。"""
        if not hasattr(self, "summary"):
            return
        model = self.model_combo.currentText() or "--"
        backend_text = self.backend_combo.currentText()
        key, width, height = self._current_size()
        resolution = f"{width}x{height}"
        self.summary.setText(
            f"模型  {model}    ·    模式  {backend_text}    ·    输出  {width}×{height}"
        )

        # 自定义尺寸尚未确认时在此直接返回。这样输入 1072 的中间值
        # 1、10、107 都不会进入 WeightResolver.resolve() 创建冗余目录。
        if (
            key == "other"
            and self._confirmed_custom_size != (width, height)
        ):
            self.weight_hint.setText(
                "○ 待确认    输入完整宽高后点击“确认尺寸”"
            )
            return

        backend_key = self.backend_combo.currentData()
        if model and backend_key:
            path = self.resolver.resolve(model, resolution, backend_key)
            relative_path = path.relative_to(self.root)
            state = "✓ 已找到" if path.is_file() else "○ 待放入"
            if model == "Passthrough" and backend_key == "pytorch":
                state = "✓ 示例模型无需权重"
            self.weight_hint.setText(f"{state}    {relative_path}")

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择图像或视频",
            "",
            "媒体文件 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp *.mp4 *.avi *.mov *.mkv *.m4v);;所有文件 (*)",
        )
        if not path:
            return
        self.source = path
        self.source_kind = "image" if Path(path).suffix.lower() in IMAGE_EXTENSIONS else "video"
        source_label = "图像" if self.source_kind == "image" else "视频"
        self.source_label.setText(f"{source_label}  ·  {path}")
        self.source_label.setToolTip(str(path))

    def _choose_camera(self) -> None:
        self.source = 0
        self.source_kind = "camera"
        self.source_label.setText("实时摄像头  ·  Camera 0")
        self.source_label.setToolTip("开始增强时按处理尺寸自动协商采集模式")

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_dir.text())
        if path:
            self.output_dir.setText(path)

    def _set_controls_running(self, running: bool) -> None:
        """运行期间锁定配置，防止工作线程与界面设置不一致。"""
        configurable_widgets = (
            self.open_file_button,
            self.camera_button,
            self.model_combo,
            self.refresh_button,
            self.backend_combo,
            self.resolution_combo,
            self.confirm_size_button,
            self.width_spin,
            self.height_spin,
            self.save_checkbox,
            self.output_dir,
            self.output_button,
        )
        for widget in configurable_widgets:
            widget.setEnabled(not running)
        if not running:
            self._update_resolution_controls()
            self.output_dir.setEnabled(self.save_checkbox.isChecked())
            self.output_button.setEnabled(self.save_checkbox.isChecked())
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)

    def start_processing(self) -> None:
        """校验界面配置，创建单次工作线程并开始处理。"""
        if self.worker is not None and self.worker.isRunning():
            return
        if self.source is None or self.source_kind is None:
            QMessageBox.warning(self, "缺少输入", "请先选择图像、视频或摄像头。")
            return
        model_name = self.model_combo.currentText()
        if not model_name:
            QMessageBox.warning(self, "缺少模型", "models 文件夹中没有有效模型定义。")
            return

        key, width, height = self._current_size()
        if (
            key == "other"
            and self._confirmed_custom_size != (width, height)
        ):
            QMessageBox.warning(
                self,
                "自定义尺寸未确认",
                "请完整输入处理宽度和高度，然后点击“确认尺寸”。",
            )
            return
        original_width, original_height = self.original_view.preview_extent()
        enhanced_width, enhanced_height = self.enhanced_view.preview_extent()
        config = RunConfig(
            root=self.root,
            model_name=model_name,
            backend=self.backend_combo.currentData(),
            resolution_key=key,
            width=width,
            height=height,
            source=self.source,
            source_kind=self.source_kind,
            # 两个预览区等宽排布，使用较小的可用尺寸可确保同一组原图/增强图
            # 缩略图都能直接显示；完整增强帧仍在工作线程内写入输出文件。
            preview_width=max(1, min(original_width, enhanced_width)),
            preview_height=max(1, min(original_height, enhanced_height)),
            save_output=self.save_checkbox.isChecked(),
            output_dir=(
                Path(self.output_dir.text()).expanduser()
                if self.save_checkbox.isChecked()
                else None
            ),
        )
        # 新任务允许正常显示帧；同时先清除上一次任务留下的画面。
        self._clear_previews_on_finish = False
        self._display_timer.stop()
        self._metrics_timer.stop()
        self._playback_active = False
        self._playback_paused = False
        self._last_display_time = None
        self._display_intervals.clear()
        self.original_view.clear_frame()
        self.enhanced_view.clear_frame()
        self.inference_card.clear_value("ms")
        self.inference_fps_card.clear_value("FPS")
        self.total_fps_card.clear_value("FPS")

        # 每次运行新建线程，结束后由 Qt 安全回收。
        self.worker = InferenceWorker(config, self.registry.get(model_name), self)
        self.worker.frames_ready.connect(self._show_frames)
        self.worker.playback_ready.connect(self._start_playback)
        self.worker.prebuffer_progress.connect(self._show_prebuffer_progress)
        self.worker.state_changed.connect(self._set_state)
        self.worker.camera_info.connect(self._show_camera_info)
        self.worker.camera_frame_available.connect(self._camera_frame_available)
        self.worker.error.connect(self._show_error)
        self.worker.output_saved.connect(self._show_saved)
        self.worker.finished.connect(self._worker_finished)
        self.pause_button.setText("Ⅱ  暂停")
        self._set_controls_running(True)
        self._set_state("正在初始化…")
        self.worker.start()
        self._metrics_timer.start()

    def _show_camera_info(self, info: CameraInfo | str) -> None:
        """主框保留中文字段的运行摘要，格式、缓存和回退诊断移到悬停详情。"""
        text = info.summary_text() if isinstance(info, CameraInfo) else info
        detail = info.detail_text() if isinstance(info, CameraInfo) else info
        self.source_label.setText(text)
        self.source_label.setToolTip(detail)
        self.source_label.setCursorPosition(0)

    def toggle_pause(self) -> None:
        """暂停推理/预览；相机继续采集最新原始帧，恢复后等待新的增强结果。"""
        if self.worker is None:
            return
        should_pause = "暂停" in self.pause_button.text()
        self.worker.request_pause(should_pause)
        self._playback_paused = should_pause
        if should_pause:
            self._display_timer.stop()
        elif self._playback_active:
            # 暂停持续时间不能计入相邻两张显示帧的间隔。
            self._last_display_time = None
            if self.source_kind != "camera":
                self._update_display_interval()
                self._display_timer.start()
        self.pause_button.setText("▶  继续" if should_pause else "Ⅱ  暂停")
        if should_pause:
            self._set_state("已暂停")
        elif self._playback_active:
            self._set_state("播放中")
        else:
            self._set_state("正在预缓冲")

    def stop_processing(self) -> None:
        """立即清空显示和指标，再请求工作线程与相机采集线程安全退出。"""
        if self.worker is not None:
            # 点击“结束”后立即恢复两个预览区的黑色等待界面。工作线程可能
            # 仍有少量 frames_ready 信号排队，因此同时设置标志阻止旧帧回填。
            self._clear_previews_on_finish = True
            self._playback_active = False
            self._display_timer.stop()
            self._metrics_timer.stop()
            self.original_view.clear_frame()
            self.enhanced_view.clear_frame()
            # 红色“结束”表示本次任务已终止，性能读数不应继续保留上一帧的
            # 结果。同步清空显示时间窗口和三张指标卡，恢复程序初始状态。
            self._last_display_time = None
            self._display_intervals.clear()
            self.inference_card.clear_value("ms")
            self.inference_fps_card.clear_value("FPS")
            self.total_fps_card.clear_value("FPS")
            self.worker.request_stop()
            self.pause_button.setEnabled(False)
            self.stop_button.setEnabled(False)
            self._set_state("正在结束…")

    def _show_frames(self, original: np.ndarray, enhanced: np.ndarray) -> None:
        """显示单张图像结果；视频和摄像头由_display_next_frame处理。"""
        if self._clear_previews_on_finish:
            return
        self.original_view.set_frame(original)
        self.enhanced_view.set_frame(enhanced)
        # 单张图像可能在第一个200 ms定时器触发前就处理完成，因此在收到
        # 图像结果时主动读取一次性能快照，避免卡片始终显示“--”。
        if self.source_kind == "image":
            self._show_metrics()

    def _show_prebuffer_progress(self, current: int, target: int) -> None:
        if not self._playback_active and not self._clear_previews_on_finish:
            self._set_state(f"正在预缓冲 {current}/{target}")

    def _start_playback(self, source_fps: float) -> None:
        """视频缓存就绪后定时播放；相机仅标记运行，等新帧唤醒。"""
        if self._clear_previews_on_finish or self.worker is None:
            return
        self._source_fps = source_fps if source_fps > 0 else 30.0
        self._playback_active = True
        self._last_display_time = None
        if self.source_kind == "camera":
            self._display_timer.stop()
        else:
            self._update_display_interval()
            if not self._playback_paused:
                self._display_timer.start()
        self._set_state("摄像头运行中" if self.source_kind == "camera" else "播放中")

    def _update_display_interval(self) -> None:
        """仅调整视频节奏；相机无轮询定时器，也不套用视频低水位降速。"""
        if self.worker is None or self.source_kind == "camera":
            return
        target_fps = self.worker.recommended_display_fps()
        interval_ms = max(1, round(1000.0 / max(target_fps, 1.0)))
        if self._display_timer.interval() != interval_ms:
            self._display_timer.setInterval(interval_ms)

    def _camera_frame_available(self) -> None:
        """只响应当前任务的通知，防止已结束相机的排队事件误取新任务画面。"""
        if self.sender() is self.worker and self.source_kind == "camera":
            self._display_next_frame()

    def _display_next_frame(self) -> None:
        """视频定时器或相机通知消费一组同步缩略图，空队列不重复显示旧帧。"""
        if (
            self.worker is None
            or self._clear_previews_on_finish
            or self._playback_paused
        ):
            return
        frames = self.worker.take_preview_frame()
        if frames is None:
            return
        original, enhanced = frames
        self.original_view.set_frame(original)
        self.enhanced_view.set_frame(enhanced)

    def _record_display_timestamp(self) -> None:
        """只记录 Enhanced 真正完成的新帧绘制，不把取帧/通知次数当作显示 FPS。"""
        now = time.perf_counter()
        if self._last_display_time is not None:
            self._display_intervals.append(now - self._last_display_time)
        self._last_display_time = now

    def _actual_display_fps(self) -> float | None:
        if not self._display_intervals:
            return None
        mean_interval = sum(self._display_intervals) / len(self._display_intervals)
        return 1.0 / max(mean_interval, 1e-6)

    def _show_metrics(self) -> None:
        """每200 ms刷新一次稳定指标，显示最近60帧的平均结果。"""
        if self.worker is None:
            return
        inference_ms, inference_fps, _processing_fps, sample_count = (
            self.worker.metrics_snapshot()
        )
        if sample_count > 0 and inference_ms is not None and inference_fps is not None:
            self.inference_card.set_value(inference_ms, "ms")
            self.inference_fps_card.set_value(inference_fps, "FPS")

        display_fps = self._actual_display_fps()
        if display_fps is not None:
            self.total_fps_card.set_value(display_fps, "FPS")

        # 处理能力或缓冲水位发生变化时平滑调整消费速度；计时器最多每
        # 200 ms改一次，避免逐帧修改间隔造成新的抖动。
        if self._playback_active and not self._playback_paused and self.source_kind != "camera":
            self._update_display_interval()

    def _set_state(self, state: str) -> None:
        self.status_badge.setText(f"●  {state}")

    def _show_error(self, message: str) -> None:
        QMessageBox.critical(self, "处理失败", message)

    def _show_saved(self, path: str) -> None:
        self.statusBar().showMessage(f"结果已保存：{path}", 12000)

    def _worker_finished(self) -> None:
        self._display_timer.stop()
        self._metrics_timer.stop()
        self._playback_active = False
        self._set_controls_running(False)
        if ("正在结束" in self.status_badge.text()
                or "正在等待摄像头驱动释放" in self.status_badge.text()):
            self._set_state("已结束")
        if self._clear_previews_on_finish:
            # 线程彻底退出后再清理一次，作为对异步尾帧的最终保护。
            self.original_view.clear_frame()
            self.enhanced_view.clear_frame()
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None

    def closeEvent(self, event) -> None:  # noqa: N802
        # 先请求线程安全退出；仍在释放大型 engine 时暂缓关闭，防止 QThread 被强制销毁。
        if self.worker is not None and self.worker.isRunning():
            self._display_timer.stop()
            self._metrics_timer.stop()
            self.worker.request_stop()
            if not self.worker.wait(8000):
                self._set_state("正在释放相机 / GPU 资源…" if self.source_kind == "camera" else "正在释放 GPU 资源…")
                event.ignore()
                return
        event.accept()
