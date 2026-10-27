"""启动偏好持久化：保存用户选择，不把本机设置写入项目源码目录。"""
from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QSettings

from config import BACKENDS, RESOLUTIONS


@dataclass(frozen=True)
class StartupPreferences:
    """启动默认值；自定义尺寸也随分辨率选项一起保存。"""
    model_name: str = ""
    backend: str = "pytorch"
    resolution_key: str = "3840x2160"
    width: int = 1920
    height: int = 1080


class PreferencesStore:
    """使用用户级 INI 文件；可注入独立 QSettings 以隔离测试与真实偏好。"""
    def __init__(self, settings: QSettings | None = None):
        self.settings = settings if settings is not None else QSettings(
            QSettings.Format.IniFormat, QSettings.Scope.UserScope,
            "UnderwaterVision", "RealtimeEnhancer",
        )

    def load(self) -> StartupPreferences:
        """损坏或过时的字段回退到安全默认值，模型存在性由扫描后的界面判断。"""
        defaults = StartupPreferences()
        model = str(self.settings.value("startup/model", defaults.model_name))
        backend = str(self.settings.value("startup/backend", defaults.backend))
        resolution = str(self.settings.value("startup/resolution", defaults.resolution_key))
        backend = backend if backend in BACKENDS else defaults.backend
        resolution = resolution if resolution in RESOLUTIONS else defaults.resolution_key
        if resolution == "auto" and backend != "pytorch":
            resolution = defaults.resolution_key
        try:
            width = int(self.settings.value("startup/width", defaults.width))
            height = int(self.settings.value("startup/height", defaults.height))
            if not 64 <= width <= 7680 or not 64 <= height <= 4320:
                raise ValueError("自定义尺寸超出界面范围")
        except (TypeError, ValueError, OverflowError):
            width, height = defaults.width, defaults.height
        return StartupPreferences(model, backend, resolution, width, height)

    def save(self, preferences: StartupPreferences) -> None:
        """只有用户点击保存才持久化；取消对话框不会改变下次启动值。"""
        for key, value in {
            "model": preferences.model_name, "backend": preferences.backend,
            "resolution": preferences.resolution_key,
            "width": preferences.width, "height": preferences.height,
        }.items():
            self.settings.setValue(f"startup/{key}", value)
        self.settings.sync()
        if self.settings.status() != QSettings.Status.NoError:
            raise OSError(f"无法保存启动设置：{self.settings.fileName()}")
