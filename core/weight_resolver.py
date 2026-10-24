from __future__ import annotations

from pathlib import Path


class WeightResolver:
    """根据后端、分辨率和模型名生成唯一的权重路径。

    PyTorch 模型通常支持可变输入尺寸，因此同一模型的所有分辨率共用
    ``weights/pth/模型名.pth``。TensorRT engine 与构建尺寸绑定，仍按
    ``后端/分辨率`` 分类保存。
    """

    # 三种后端分别保存，避免同名文件混淆。
    FOLDERS = {"pytorch": "pth", "trt16": "trt16", "trt8": "trt8"}

    def __init__(self, weights_root: Path):
        self.weights_root = weights_root
        for folder in self.FOLDERS.values():
            (weights_root / folder).mkdir(parents=True, exist_ok=True)

    def expected_name(self, model: str, resolution: str, backend: str) -> str:
        """生成权重文件名，例如 PUIENet_1920x1080_trt8.engine。"""
        if backend == "pytorch":
            return f"{model}.pth"
        if backend in ("trt16", "trt8"):
            return f"{model}_{resolution}_{backend}.engine"
        raise ValueError(f"未知推理后端：{backend}")

    def resolve(self, model: str, resolution: str, backend: str) -> Path:
        """返回权重路径，并按需建立 TensorRT 的分辨率分类目录。"""
        folder = self.FOLDERS[backend]

        # PyTorch checkpoint 不按输入尺寸划分；切换界面分辨率时继续加载
        # 同一个模型权重，避免保存多份内容相同的 .pth 文件。
        if backend == "pytorch":
            return self.weights_root / folder / self.expected_name(
                model,
                resolution,
                backend,
            )

        # TensorRT engine 通常是固定尺寸构建的，必须保留分辨率子目录。
        resolution_dir = self.weights_root / folder / resolution
        resolution_dir.mkdir(parents=True, exist_ok=True)
        return resolution_dir / self.expected_name(model, resolution, backend)
