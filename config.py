from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# 键名直接使用“宽x高”，它同时也是权重目录名和文件名中的分辨率字段。
# 这样界面显示、目录结构和权重命名始终使用同一套规则。
RESOLUTIONS: dict[str, tuple[str, int, int]] = {
    "3840x2160": ("3840 × 2160（4K UHD）", 3840, 2160),
    "2560x1440": ("2560 × 1440（QHD）", 2560, 1440),
    "1920x1080": ("1920 × 1080（Full HD）", 1920, 1080),
    "1280x720": ("1280 × 720（HD）", 1280, 720),
    "other": ("其他 / 自定义", 0, 0),
    # 原尺寸在读取每个媒体文件后确定，不用 0x0 解析 TensorRT 权重目录。
    "auto": ("自动匹配", 0, 0),
}

BACKENDS = {
    "pytorch": "PyTorch",
    "trt16": "TensorRT FP16",
    "trt8": "TensorRT INT8",
}


@dataclass(frozen=True)
class RunConfig:
    """界面点击“开始”后生成的不可变运行参数快照。"""

    # root 用于定位 models、weights 和默认输出目录。
    root: Path
    model_name: str
    backend: str
    resolution_key: str
    width: int
    height: int
    source: str | int
    source_kind: str
    # 工作线程只把与界面控件大小接近的缩略图放入预览缓冲区；完整分辨率
    # 的增强结果仍直接交给视频写入器或图像保存逻辑，不会降低保存质量。
    preview_width: int = 1280
    preview_height: int = 720
    save_output: bool = False
    output_dir: Path | None = None

    @property
    def weight_resolution_key(self) -> str:
        """返回权重目录与文件名统一使用的具体分辨率字段。"""
        return f"{self.width}x{self.height}"

    @property
    def auto_match(self) -> bool:
        """自动匹配仅用于 PyTorch 图像/视频，后端逐帧保留源尺寸。"""
        return self.resolution_key == "auto"
