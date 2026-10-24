from __future__ import annotations

import importlib
import platform
import sys


def inspect_module(name: str) -> tuple[str, str]:
    """安全导入单个依赖，并返回状态和版本或错误原因。"""
    try:
        module = importlib.import_module(name)
        return "OK", str(getattr(module, "__version__", "已安装"))
    except Exception as exc:
        return "MISSING", str(exc)


def main() -> None:
    """打印 Python、核心依赖、CUDA 和 GPU 信息，便于启动前排查环境。"""
    print(f"Python: {sys.version.split()[0]} ({platform.system()})")
    for module_name in ("numpy", "cv2", "PySide6", "torch", "tensorrt"):
        status, version = inspect_module(module_name)
        print(f"{module_name:10s}: {status:7s} {version}")
    # torch 可用时进一步检查它是否识别到 CUDA 和目标显卡。
    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        print(f"Torch CUDA: {torch.version.cuda}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
