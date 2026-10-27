"""处理调度层：图像单次推理、视频有界预缓冲、相机最新帧流水线。

CUDA 后端只在本工作线程中创建和调用。相机原始采集由独立线程负责，
界面只消费缩略预览；计时与保存始终使用真实处理结果和完整输出尺寸。
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QThread, Signal

from config import RunConfig
from core.backends import create_backend
from core.camera import CameraCaptureError, CameraInfo, LatestFrameCapture, invalidate_camera_cache, open_camera
from core.model_registry import ModelSpec
from core.media import IMAGE_EXTENSIONS, MediaInputError, create_video_writer, folder_media
from core.weight_resolver import WeightResolver


class InferenceWorker(QThread):
    """后台完成媒体读取、GPU 推理、完整结果保存和预览帧生产。"""

    # 单张图像只发送一次信号；视频由主线程定时消费有界缓冲区；
    # 相机只发送合并的唤醒通知，不把高分辨率帧塞入 Qt 事件队列。
    frames_ready = Signal(object, object)
    playback_ready = Signal(float)
    prebuffer_progress = Signal(int, int)
    state_changed = Signal(str)
    error = Signal(str)
    output_saved = Signal(str)
    camera_info = Signal(object)
    camera_frame_available = Signal()
    # 文件夹逐项切换使用握手，防止上一视频的定时器消费下一项的队列。
    media_started = Signal(str, str, int, int)
    media_skipped = Signal(str)
    processing_size_changed = Signal(int, int)

    VIDEO_PREBUFFER_FRAMES = 30
    METRIC_WINDOW = 60
    TENSORRT_WARMUP_FRAMES = 20

    def __init__(self, config: RunConfig, model_spec: ModelSpec, parent=None):
        super().__init__(parent)
        self.config = config
        self.model_spec = model_spec
        self._stop_event = threading.Event()
        self._pause_condition = threading.Condition()
        self._paused = False

        # 视频使用30帧有界队列平滑短时间推理波动；摄像头使用同一个容器，
        # 但每次都覆盖旧帧，因此摄像头预览不会逐渐累积延迟。
        self._preview_condition = threading.Condition()
        self._preview_queue: deque[tuple[np.ndarray, np.ndarray]] = deque()
        # 通知最多挂起一次。epoch 使暂停/恢复前正在推理的旧结果失效。
        self._camera_notify_pending = False
        self._camera_pause_epoch = 0
        self._camera_capture: LatestFrameCapture | None = None
        self._camera_description: CameraInfo | None = None
        self._last_camera_info_time = 0.0

        # 性能样本只保留最近60帧。主线程每200 ms读取一次快照，不再为每帧
        # 发送指标信号，从而减少主线程事件队列压力。
        self._metrics_lock = threading.Lock()
        self._inference_samples: deque[float] = deque(maxlen=self.METRIC_WINDOW)
        self._processing_samples: deque[float] = deque(maxlen=self.METRIC_WINDOW)
        self._source_fps = 30.0
        self._item_ack = threading.Event()
        self._image_ack = threading.Event()

    def acknowledge_item(self) -> None:
        """主线程完成逐项播放状态重置后，允许后台开始读取下一文件。"""
        self._item_ack.set()

    def acknowledge_image(self) -> None:
        """主线程已复制图像预览，文件夹可以继续，避免堆积大图信号。"""
        self._image_ack.set()

    def _wait_for_ack(self, event: threading.Event) -> None:
        """握手等待可以被停止打断；暂停仍通过现有Condition控制。"""
        while not event.wait(0.1):
            if self._stop_event.is_set():
                raise InterruptedError("已结束文件夹处理")
        if self._stop_event.is_set():
            raise InterruptedError("已结束文件夹处理")

    def request_pause(self, paused: bool) -> None:
        """设置暂停状态；Condition 可避免暂停时持续占用 CPU。"""
        with self._pause_condition:
            self._paused = paused
            if self.config.source_kind == "camera":
                self._camera_pause_epoch += 1
            self._pause_condition.notify_all()
        with self._preview_condition:
            if self.config.source_kind == "camera":
                # 暂停期间采集仍继续；恢复只显示新一轮处理的结果。
                self._preview_queue.clear()
                self._camera_notify_pending = False
            self._preview_condition.notify_all()

    def request_stop(self) -> None:
        """请求安全结束，并唤醒可能处于暂停状态的线程。"""
        self._stop_event.set()
        capture = self._camera_capture
        if capture is not None:
            capture.request_stop()
        with self._pause_condition:
            self._paused = False
            self._pause_condition.notify_all()
        with self._preview_condition:
            # “结束”必须立即丢弃尚未播放的缓存帧，否则工作线程会等待主线程
            # 消费缓冲区，且旧帧可能再次覆盖黑色等待界面。
            self._preview_queue.clear()
            self._camera_notify_pending = False
            self._preview_condition.notify_all()

    def _wait_if_paused(self) -> None:
        with self._pause_condition:
            while self._paused and not self._stop_event.is_set():
                self._pause_condition.wait(timeout=0.2)

    def _output_path(self, suffix: str, source: str | int | None = None) -> Path | None:
        """媒体文件保留原名称，视频保持MP4格式；无源文件名的摄像头使用时间戳。"""
        if not self.config.save_output or self.config.output_dir is None:
            return None
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        if source is not None and self.config.source_kind != "camera":
            # 单文件和文件夹使用相同命名，不附加enhanced、日期或唯一后缀。
            # 图像保留原扩展名；视频保留主文件名，沿用原有MP4输出格式。
            original = Path(str(source))
            name = (original.name if suffix == ".png" or original.suffix.lower() == suffix
                    else original.with_suffix(suffix).name)
            output = self.config.output_dir / name
            # 保持原先“保存不会覆盖输入”的行为，包含符号链接/硬链接情况。
            if (output.resolve() == original.resolve()
                    or (output.exists() and output.samefile(original))):
                raise ValueError("输出文件与原文件是同一文件，请选择不同的输出目录，避免覆盖原文件")
            return output
        stamp = time.strftime("%Y%m%d_%H%M%S")
        # 摄像头没有原文件名，保持原有时间戳规则。
        unique = time.time_ns() % 1_000_000_000
        return self.config.output_dir / f"enhanced_{stamp}_{unique:09d}{suffix}"

    def take_preview_frame(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Qt定时器或相机唤醒槽取出同步预览；空缓冲区立即返回。"""
        with self._preview_condition:
            self._camera_notify_pending = False
            if not self._preview_queue:
                return None
            frames = self._preview_queue.popleft()
            self._preview_condition.notify_all()
            return frames

    def preview_buffer_size(self) -> int:
        """返回当前待显示帧数，用于自适应显示节奏。"""
        with self._preview_condition:
            return len(self._preview_queue)

    def metrics_snapshot(self) -> tuple[float | None, float | None, float | None, int]:
        """返回滑动窗口内的推理耗时、GPU FPS、处理FPS和有效样本数。"""
        with self._metrics_lock:
            sample_count = len(self._inference_samples)
            if sample_count == 0:
                return None, None, None, 0
            inference_ms = sum(self._inference_samples) / sample_count
            processing_ms = sum(self._processing_samples) / len(self._processing_samples)
        return (
            inference_ms,
            1000.0 / max(inference_ms, 1e-6),
            1000.0 / max(processing_ms, 1e-6),
            sample_count,
        )

    def recommended_display_fps(self) -> float:
        """根据源FPS和最近处理能力给出不会持续耗空缓存的显示FPS。"""
        _, _, processing_fps, _ = self.metrics_snapshot()
        if processing_fps is None:
            return self._source_fps

        # 推理较快时严格跟随源视频；较慢时按实际生产能力播放。缓冲区接近
        # 耗尽时留出少量余量，用来吸收解码、系统调度和写盘的短时抖动。
        target = min(self._source_fps, processing_fps)
        if (self.config.source_kind != "camera"
                and self.preview_buffer_size() <= 5 and processing_fps < self._source_fps):
            target = min(target, processing_fps * 0.97)
        return max(1.0, target)

    def _record_metrics(self, inference_ms: float, processing_ms: float) -> None:
        with self._metrics_lock:
            self._inference_samples.append(float(inference_ms))
            self._processing_samples.append(float(processing_ms))

    def _make_preview(self, frame: np.ndarray) -> np.ndarray:
        """生成适配预览控件的BGR缩略图，不修改用于保存的完整结果。"""
        height, width = frame.shape[:2]
        max_width = max(1, int(self.config.preview_width))
        max_height = max(1, int(self.config.preview_height))
        scale = min(max_width / width, max_height / height, 1.0)
        if scale >= 0.999:
            return np.ascontiguousarray(frame.copy())
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        return np.ascontiguousarray(
            cv2.resize(frame, target, interpolation=cv2.INTER_AREA)
        )

    def _make_preview_pair(
        self,
        original: np.ndarray,
        enhanced: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._make_preview(original), self._make_preview(enhanced)

    def _enqueue_video_preview(
        self,
        frames: tuple[np.ndarray, np.ndarray],
    ) -> bool:
        """向30帧有界队列写入视频预览；队列满时等待显示线程消费。"""
        with self._preview_condition:
            while (
                len(self._preview_queue) >= self.VIDEO_PREBUFFER_FRAMES
                and not self._stop_event.is_set()
            ):
                self._preview_condition.wait(timeout=0.1)
            if self._stop_event.is_set():
                return False
            self._preview_queue.append(frames)
        return True

    def _replace_camera_preview(
        self,
        frames: tuple[np.ndarray, np.ndarray],
        epoch: int | None = None,
    ) -> None:
        """最新增强预览覆盖旧帧；无待处理通知时才唤醒一次主线程。"""
        notify = False
        with self._preview_condition:
            if (self._stop_event.is_set() or self._paused
                    or (epoch is not None and epoch != self._camera_pause_epoch)):
                return
            self._preview_queue.clear()
            self._preview_queue.append(frames)
            if not self._camera_notify_pending:
                self._camera_notify_pending = True
                notify = True
            self._preview_condition.notify_all()
        # 在锁外发信号，主线程可立即取帧，避免锁与 Qt 事件调度互相等待。
        if notify:
            self.camera_frame_available.emit()

    def _report_camera_info(self, info: CameraInfo | str) -> None:
        """保存结构化协商结果；初始化字符串仅作进度，不冒充最终配置。"""
        if isinstance(info, CameraInfo):
            self._camera_description = info
        self.camera_info.emit(info)

    def _update_camera_info(self) -> None:
        """每秒最多一次报告采集线程的实测速率，不为原始帧逐帧发送 Qt 信号。"""
        now = time.perf_counter()
        if (self._camera_capture is None or self._camera_description is None
                or now - self._last_camera_info_time < 1.0):
            return
        fps = self._camera_capture.capture_fps()
        if fps is not None:
            # 实采由读取完成间隔统计；缓存验证/比选采样不替代运行实采FPS。
            self.camera_info.emit(replace(self._camera_description, actual_fps=fps))
            self._last_camera_info_time = now

    def _wait_for_preview_drain(self) -> None:
        """视频读完后等待最后的缓存帧显示完，再结束工作线程。"""
        with self._preview_condition:
            while self._preview_queue and not self._stop_event.is_set():
                self._preview_condition.wait(timeout=0.1)

    def run(self) -> None:
        """在同一工作线程创建后端，文件夹按固定清单顺序处理并复用模型。"""
        backend = None
        try:
            if self.config.auto_match and (self.config.backend != "pytorch" or self.config.source_kind == "camera"):
                raise ValueError("自动匹配仅支持PyTorch图像、视频或文件夹；摄像头请选择固定尺寸")
            is_folder = self.config.source_kind == "folder"
            if is_folder:
                items = folder_media(Path(str(self.config.source)))
                if not items:
                    raise MediaInputError("文件夹当前目录中没有支持的图像或视频")
            else:
                items = [(self.config.source, self.config.source_kind)]
            if is_folder and self.config.save_output and self.config.output_dir is not None:
                # AVI/MP4等同主名输入会映射到同一个MP4，先检查整批，
                # 不加后缀也不能让后一个来源静默覆盖前一个来源的结果。
                planned: dict[Path, Path] = {}
                for source, kind in items:
                    output = self._output_path(".png" if kind == "image" else ".mp4", str(source))
                    if output in planned:
                        raise ValueError(
                            f"文件夹中 {planned[output].name} 与 {Path(source).name} 的输出文件名均为 {output.name}，"
                            "请分开处理或调整源文件名，避免结果互相覆盖"
                        )
                    planned[output] = Path(source)
            resolver = WeightResolver(self.config.root / "weights")
            weight_path = resolver.resolve(
                self.config.model_name,
                self.config.weight_resolution_key,
                self.config.backend,
            )
            backend = create_backend(
                self.config.backend,
                self.model_spec,
                weight_path,
                self.config.width,
                self.config.height,
                auto_match=self.config.auto_match,
            )
            succeeded, skipped = 0, 0
            for index, (source, kind) in enumerate(items, 1):
                self._wait_if_paused()
                if self._stop_event.is_set():
                    break
                if is_folder:
                    # 每项先由主线程停旧定时器、清显示计时，再开放生产；
                    # 不同文件的FPS、预热和处理耗时不能混成同一个滑动窗口。
                    with self._metrics_lock:
                        self._inference_samples.clear()
                        self._processing_samples.clear()
                    self._item_ack.clear()
                    self._source_fps = 30.0
                    self.media_started.emit(str(source), kind, index, len(items))
                    self._wait_for_ack(self._item_ack)
                    self._wait_if_paused()
                    if self._stop_event.is_set():
                        break
                try:
                    self._process_media(backend, str(source) if kind != "camera" else source, kind, is_folder)
                    succeeded += 1
                except MediaInputError as exc:
                    if not is_folder:
                        raise
                    skipped += 1
                    self.media_skipped.emit(f"{Path(str(source)).name}：{exc}")
            if self._stop_event.is_set():
                self.state_changed.emit("已结束")
            elif is_folder:
                self.state_changed.emit(f"文件夹处理完成 · 成功{succeeded}项 · 跳过{skipped}项")
            else:
                self.state_changed.emit("图像处理完成" if self.config.source_kind == "image" else "视频处理完成")
        except InterruptedError:
            self.state_changed.emit("已结束")
        except Exception as exc:
            if isinstance(exc, CameraCaptureError):
                # 模型/写盘异常不误删相机成功配置，采集故障才清缓存。
                invalidate_camera_cache()
            self.error.emit(f"{type(exc).__name__}: {exc}")
            self.state_changed.emit("发生错误")
        finally:
            if backend is not None:
                backend.close()

    def _process_media(self, backend, source: str | int, source_kind: str, is_folder: bool) -> None:
        """处理一个文件或相机；文件夹复用此路径，资源仍按媒体项及时释放。"""
        capture, writer, output_path = None, None, None
        try:
            if source_kind == "image":
                # imdecode + fromfile 可正确读取含中文字符的 Windows 路径。
                try:
                    raw = np.fromfile(str(source), dtype=np.uint8)
                except OSError as exc:
                    raise MediaInputError(f"无法读取图像：{exc}") from exc
                try:
                    frame = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size else None
                except cv2.error as exc:
                    raise MediaInputError(f"无法解码所选图像：{exc}") from exc
                if frame is None:
                    raise MediaInputError("无法解码所选图像")
                width, height = (frame.shape[1], frame.shape[0]) if self.config.auto_match else (self.config.width, self.config.height)
                self.processing_size_changed.emit(width, height)
                started = time.perf_counter()
                enhanced, inference_ms = backend.infer(frame)
                total_ms = (time.perf_counter() - started) * 1000.0
                self._record_metrics(inference_ms, total_ms)
                original_preview, enhanced_preview = self._make_preview_pair(frame, enhanced)
                if is_folder:
                    self._image_ack.clear()
                self.frames_ready.emit(original_preview, enhanced_preview)
                output_path = self._output_path(".png", source)
                if output_path is not None:
                    # 名字与源文件一致时也按原扩展名编码，不能把PNG字节伪装成JPG。
                    ok, encoded = cv2.imencode(output_path.suffix.lower(), enhanced)
                    if not ok:
                        raise RuntimeError(f"结果保存失败：{output_path}")
                    encoded.tofile(str(output_path))
                    self.output_saved.emit(str(output_path))
                if is_folder:
                    self._wait_for_ack(self._image_ack)
                return

            first_frame = None
            if source_kind == "camera":
                self.state_changed.emit("正在协商摄像头模式…")
                capture, first_frame, source_fps = open_camera(
                    int(source), self.config.width, self.config.height,
                    self._stop_event.is_set, self._report_camera_info,
                )
            else:
                capture = cv2.VideoCapture(str(source))
                source_fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened():
                raise MediaInputError("无法打开视频或摄像头")
            if not np.isfinite(source_fps) or source_fps <= 0:
                source_fps = 30.0
            self._source_fps = source_fps
            if source_kind == "camera":
                # 从此处起由采集线程独占读取/释放；丢弃启动验证帧，等待最新帧。
                capture = LatestFrameCapture(capture, self._stop_event.is_set, self.state_changed.emit)
                self._camera_capture = capture
                first_frame = None
            is_camera = source_kind == "camera"
            playback_announced = False
            frame_number = 0
            camera_read_failures = 0
            video_size = None
            warmup_frames = (
                self.TENSORRT_WARMUP_FRAMES
                if self.config.backend in ("trt16", "trt8")
                else 0
            )
            if is_camera:
                # 摄像头不等待30帧，通过新帧通知显示，应用层不积累历史画面。
                self.state_changed.emit("摄像头运行中")
                self.playback_ready.emit(source_fps)
                playback_announced = True
            else:
                self.state_changed.emit("正在预缓冲")

            while not self._stop_event.is_set():
                self._wait_if_paused()
                if self._stop_event.is_set():
                    break
                with self._pause_condition:
                    frame_epoch = self._camera_pause_epoch

                # 视频从解码开始计时；相机从等待新增原始帧开始计时，解码由
                # 独立采集线程完成。后续包含推理、后处理、可选写盘和预览缩放，
                # 不包含视频队列满时的等待。相机处理 FPS 因而也受实采速度限制。
                started = time.perf_counter()
                if first_frame is not None:
                    ok, frame = True, first_frame
                    first_frame = None
                else:
                    ok, frame = capture.read()
                if not ok:
                    if is_camera:
                        camera_read_failures += 1
                        if camera_read_failures >= 10:
                            raise RuntimeError("摄像头连续读取失败，请检查连接或是否被其他程序占用")
                        self._stop_event.wait(0.05)
                        continue
                    break
                camera_read_failures = 0
                if video_size is None:
                    # 视频尺寸以第一张真实解码帧为准，不依赖可能失真的元数据。
                    video_size = ((frame.shape[1], frame.shape[0]) if self.config.auto_match else
                                  (self.config.width, self.config.height))
                    self.processing_size_changed.emit(*video_size)
                    output_path = self._output_path(".mp4", source)
                    if output_path is not None:
                        writer = create_video_writer(output_path, source_fps, *video_size, exact_size=self.config.auto_match)
                        if not writer.isOpened():
                            raise RuntimeError(f"无法创建输出视频：{output_path}")
                elif self.config.auto_match and (frame.shape[1], frame.shape[0]) != video_size:
                    raise ValueError("视频中途改变了帧尺寸，无法写入固定尺寸视频，请先拆分该视频")
                enhanced, inference_ms = backend.infer(frame)
                if writer is not None:
                    # 保存路径始终接收完整分辨率增强结果，预览缩略图不会影响输出。
                    if enhanced.shape[:2] != (video_size[1], video_size[0]):
                        raise ValueError("模型输出尺寸与保存尺寸不一致，请检查网络输入约束")
                    writer.write(enhanced)

                preview_pair = self._make_preview_pair(frame, enhanced)
                processing_ms = (time.perf_counter() - started) * 1000.0
                frame_number += 1

                # TensorRT前20帧用于CUDA上下文、内核和缓存预热，正常显示和
                # 保存，但不进入FPS统计；之后使用最近60帧滑动窗口。
                if frame_number > warmup_frames:
                    self._record_metrics(inference_ms, processing_ms)

                if is_camera:
                    self._replace_camera_preview(preview_pair, frame_epoch)
                    self._update_camera_info()
                else:
                    if not self._enqueue_video_preview(preview_pair):
                        break
                    if not playback_announced:
                        buffered = self.preview_buffer_size()
                        # 进度信号只在最初30帧发送；进入播放阶段后完全通过
                        # 线程安全缓冲区交换画面，不再制造逐帧Qt事件。
                        self.prebuffer_progress.emit(
                            buffered,
                            self.VIDEO_PREBUFFER_FRAMES,
                        )
                        if buffered >= self.VIDEO_PREBUFFER_FRAMES:
                            self.playback_ready.emit(source_fps)
                            playback_announced = True

            if frame_number == 0 and not self._stop_event.is_set():
                raise MediaInputError("视频未解码到任何有效帧")
            # 短视频不足30帧时，在文件结束后也必须启动播放；普通视频则等待
            # 已经处理好的尾帧显示完毕，保证预览顺序完整。
            if not is_camera and not self._stop_event.is_set():
                if not playback_announced and self.preview_buffer_size() > 0:
                    self.playback_ready.emit(source_fps)
                self._wait_for_preview_drain()

            if writer is not None:
                completed, writer = writer, None
                completed.release()
            if output_path is not None and output_path.exists():
                self.output_saved.emit(str(output_path))
        except InterruptedError:
            # “结束”可能恰好打断等待相机新帧；仍需完成已打开的视频文件并
            # 通知保存路径，避免只在finally释放却漏掉“结果已保存”的提示。
            if writer is not None:
                completed, writer = writer, None
                completed.release()
                if output_path is not None and output_path.exists():
                    self.output_saved.emit(str(output_path))
            raise
        finally:
            # 无论正常结束还是发生异常，都释放文件句柄和摄像头资源。
            # 编码收尾本身也可能失败；嵌套finally确保读取器仍然被释放。
            try:
                if writer is not None:
                    writer.release()
            finally:
                try:
                    if capture is not None:
                        capture.release()
                finally:
                    self._camera_capture = None
