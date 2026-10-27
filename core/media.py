"""媒体文件枚举与原尺寸视频保存；采集、推理和相机适配仍由各自模块负责。"""
from __future__ import annotations

from pathlib import Path
import subprocess
import tempfile

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".mpg", ".mpeg", ".webm"}


class MediaInputError(RuntimeError):
    """文件夹中坏文件可跳过；模型、显存与写盘错误仍应中止任务。"""


def media_kind(path: Path) -> str | None:
    """按扩展名识别支持的媒体类型，后续仍验证实际能否解码。"""
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def folder_media(folder: Path) -> list[tuple[Path, str]]:
    """只扫描当前目录，开始时固定清单；子目录与非媒体文件不参与处理。"""
    if not folder.is_dir():
        raise MediaInputError(f"文件夹不存在：{folder}")
    return [(p, kind) for p in sorted(folder.iterdir(), key=lambda p: (p.name.casefold(), p.name))
            if p.is_file() and (kind := media_kind(p)) is not None]


class ExactSizeVideoWriter:
    """奇数尺寸使用 H.264 4:4:4，避免 OpenCV mp4v 静默截去最后一行/列。

    FFmpeg 由已安装的 imageio-ffmpeg 提供；stderr 写匿名临时文件以免管道
    填满阻塞。释放时完成 MP4 索引并检查编码退出状态，临时句柄随后关闭。
    """
    def __init__(self, path: Path, fps: float, width: int, height: int):
        import imageio_ffmpeg

        self.width, self.height = width, height
        self._process = None
        self._stderr = tempfile.TemporaryFile()
        try:
            self._process = subprocess.Popen([
                imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pixel_format", "bgr24", "-video_size", f"{width}x{height}",
                "-framerate", f"{fps:.12g}", "-i", "pipe:0", "-an", "-c:v", "libx264",
                "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv444p", str(path),
            ], stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=self._stderr,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        except BaseException:
            self._stderr.close()
            raise

    def isOpened(self) -> bool:  # noqa: N802
        return self._process is not None and self._process.poll() is None

    def write(self, frame: np.ndarray) -> None:
        """逐帧写入完整 BGR 数据；不通过补到偶数再编码来改变保存尺寸。"""
        if frame.dtype != np.uint8 or frame.shape != (self.height, self.width, 3):
            raise ValueError("编码帧必须与已确定的原尺寸一致")
        if self._process is None:
            raise RuntimeError("视频编码器已关闭")
        try:
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except (BrokenPipeError, OSError) as exc:
            raise RuntimeError("原尺寸视频编码失败，请检查 FFmpeg、磁盘空间与输出路径") from exc

    def release(self) -> None:
        """只关闭自己持有的子进程；重复释放无操作，异常时也清理句柄。"""
        process, self._process = self._process, None
        if process is None:
            return
        try:
            try:
                process.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise RuntimeError("原尺寸视频编码收尾超时，已释放编码进程") from exc
            if code:
                self._stderr.seek(0)
                detail = self._stderr.read().decode("utf-8", errors="replace")[-1500:]
                raise RuntimeError(f"原尺寸视频编码失败：{detail}")
        finally:
            self._stderr.close()


def create_video_writer(path: Path, fps: float, width: int, height: int, exact_size: bool = False):
    """仅自动匹配的奇数尺寸走专用编码，其余情况保留原 mp4v 保存路径。"""
    if exact_size and (width % 2 or height % 2):
        return ExactSizeVideoWriter(path, fps, width, height)
    return cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
