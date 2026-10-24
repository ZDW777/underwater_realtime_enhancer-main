"""相机适配：首次检测、进程内配置复用、异常重协商与独立最新帧采集。

驱动属性回读不等于生效；不同后端的设备序号和 FOURCC 含义也可能不同。
本模块验证实际帧，运行阶段由唯一采集线程读取/释放设备，避免并发驱动调用。
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import ctypes
import math
import os
import re
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np


@dataclass(frozen=True)
class CameraMode:
    """驱动公布的固定尺寸、最高标称帧率和传输格式。"""
    width: int
    height: int
    fps: float
    fourcc: str


@dataclass(frozen=True)
class CameraDevice:
    """保留后端枚举顺序及设备路径，用于跨后端定位同一台相机。"""
    index: int
    name: str
    path: str = ""


@dataclass(frozen=True)
class CameraInfo:
    """将运行摘要与诊断详情分开，避免界面解析、裁剪一段长字符串。"""
    device_name: str
    backend: str
    capture_width: int
    capture_height: int
    model_width: int
    model_height: int
    nominal_fps: float
    format_description: str
    startup_kind: str
    startup_seconds: float
    probe_fps: float | None = None
    actual_fps: float | None = None
    notes: tuple[str, ...] = ()

    def summary_text(self) -> str:
        """主框保留中文字段；实采速率未形成样本时明确显示“测量中”。"""
        name = " ".join(self.device_name.split())
        fps = f"{self.actual_fps:.1f} FPS" if self.actual_fps is not None else "测量中"
        warning = " · ⚠ 低帧率模式" if self.nominal_fps < 15 else ""
        return (f"设备-{name} · 后端-{self.backend} · "
                f"采集尺寸-{self.capture_width}×{self.capture_height} → "
                f"模型尺寸-{self.model_width}×{self.model_height} · 实采-{fps}{warning}")

    def detail_text(self) -> str:
        """悬停显示格式、标称/实测速率、缓存路径、协商耗时及回退原因。"""
        actual = f"{self.actual_fps:.2f} FPS" if self.actual_fps is not None else "测量中"
        probe = f"{self.probe_fps:.2f} FPS" if self.probe_fps is not None else "快速复用未重复测速"
        conversion = ("采集与模型同尺寸，无需输入缩放"
                      if (self.capture_width, self.capture_height) == (self.model_width, self.model_height)
                      else "采集尺寸与模型不同，推理前进行一次输入缩放")
        lines = [f"设备：{self.device_name}", f"后端：{self.backend}",
                 f"采集尺寸：{self.capture_width}×{self.capture_height}",
                 f"模型尺寸：{self.model_width}×{self.model_height}",
                 f"格式：{self.format_description}", f"标称帧率：{self.nominal_fps:.2f} FPS",
                 f"比选采样：{probe}", f"实采帧率：{actual}",
                 f"启动方式：{self.startup_kind}",
                 f"协商耗时：{self.startup_seconds:.2f} 秒（不含模型加载）", conversion]
        lines.extend(dict.fromkeys(self.notes))
        return "\n".join(lines)


@dataclass(frozen=True)
class _CachedMode:
    """仅记住成功后端和诊断信息，不保留设备句柄或旧的设备序号。"""
    api: int
    notes: tuple[str, ...]


# 缓存只在本次程序运行内有效。设备列表每次重新核对，能力及后端按唯一路径
# 和原生采集模式复用；模型名不参与键，4K/2K模型可共享同一4K采集配置。
_cache_lock = threading.RLock()
_inventory_snapshot: tuple[CameraDevice, ...] | None = None
_capability_cache: dict[str, tuple[CameraMode, ...]] = {}
_successful_modes: dict[tuple[str, CameraMode], _CachedMode] = {}
_target_modes: dict[tuple[str, int, int], CameraMode] = {}


def invalidate_camera_cache() -> None:
    """设备清单变化或运行期采集异常后清缓存，下次启动执行完整检测。"""
    global _inventory_snapshot
    with _cache_lock:
        _inventory_snapshot = None
        _capability_cache.clear()
        _successful_modes.clear()
        _target_modes.clear()


class CameraCaptureError(RuntimeError):
    """区分采集故障与模型错误，避免模型推理失败误删相机成功配置。"""


def parse_modes(text: str) -> list[CameraMode]:
    """读取 DirectShow 公布的固定尺寸模式；不把任意请求值当作硬件能力。"""
    modes = set()
    formats = {"mjpeg": "MJPG", "yuyv422": "YUY2", "uyvy422": "UYVY",
               "nv12": "NV12", "rgb24": "RGB3", "bgr24": "BGR3"}
    pattern = r"(?:vcodec|pixel_format)=(\w+)\s+min s=(\d+)x(\d+) fps=([\d.]+)\s+max s=(\d+)x(\d+) fps=([\d.]+)"
    for match in re.finditer(pattern, text):
        fmt, w, h, _minimum, max_w, max_h, fps = match.groups()
        if (fmt in formats and (w, h) == (max_w, max_h)
                and int(w) > 0 and int(h) > 0 and math.isfinite(float(fps)) and float(fps) > 0):
            modes.add(CameraMode(int(w), int(h), float(fps), formats[fmt]))
    return list(modes)


def rank_modes(modes: list[CameraMode], width: int, height: int) -> list[CameraMode]:
    """同尺寸优先，其次相近宽高比；优先避免放大，再选择较小采集尺寸。"""
    def key(mode):
        return (
            (mode.width, mode.height) != (width, height),
            round(abs(math.log((mode.width / mode.height) / (width / height))), 5),
            mode.width < width or mode.height < height,
            abs(math.log(mode.width * mode.height / (width * height))),
            -mode.fps,
            mode.fourcc != "MJPG",
            mode.fourcc,
        )
    return sorted(modes, key=key)


def _query_dshow(arguments: list[str]) -> str:
    """FFmpeg 仅查询 Windows 驱动能力；图像采集仍交给 OpenCV。"""
    import imageio_ffmpeg
    result = subprocess.run(
        [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", *arguments], capture_output=True,
        timeout=15, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    # list_devices/list_options 查询正常完成时也会以非零状态退出。
    return result.stderr.decode("utf-8", errors="replace")


def _dshow_devices() -> list[CameraDevice]:
    """保留虚拟视频设备的序号，使用 Alternative name 区分同名相机。"""
    devices = []
    current = None
    for line in _query_dshow(["-list_devices", "true", "-f", "dshow", "-i", "dummy"]).splitlines():
        match = re.search(r'"([^"\r\n]+)" \((video|none|audio)\)', line)
        if match:
            current = None
            if match[2] != "audio":
                current = len(devices)
                devices.append(CameraDevice(current, match[1]))
        elif current is not None:
            alternate = re.search(r'Alternative name "([^"\r\n]+)"', line)
            if alternate:
                device = devices[current]
                devices[current] = CameraDevice(device.index, device.name, alternate[1])
    return devices


def _device_modes(device: CameraDevice) -> list[CameraMode]:
    """通过唯一设备路径查询能力，不将同名设备的规格混用。"""
    detail = _query_dshow(["-list_options", "true", "-f", "dshow", "-i",
                           f"video={device.path or device.name}"])
    modes = parse_modes(detail)
    if not modes:
        raise RuntimeError("驱动未公布可用的 MJPEG/原始像素固定尺寸模式")
    return modes


def discover_modes(index: int) -> tuple[str, list[CameraMode]]:
    """供诊断工具使用的查询接口；不打开或持有相机。"""
    devices = _dshow_devices()
    if index < 0 or index >= len(devices):
        raise RuntimeError(f"未枚举到摄像头 {index}")
    return devices[index].name, _device_modes(devices[index])


def _msmf_devices() -> list[CameraDevice]:
    """使用 MFEnumDeviceSources 取得 MSMF 的真实设备顺序与符号链接。

    只枚举、不激活设备。通过标准库 ctypes 调用 Windows API，所有 COM
    引用、枚举数组和字符串都在当前线程释放，初始化与关闭严格配对。
    """
    if os.name != "nt":
        return []
    class GUID(ctypes.Structure):
        _fields_ = [("data", ctypes.c_ubyte * 16)]

    def guid(value):
        return GUID.from_buffer_copy(uuid.UUID(value).bytes_le)

    def checked(hr):
        if hr < 0:
            raise OSError(f"Media Foundation HRESULT 0x{hr & 0xffffffff:08X}")

    def method(pointer, slot, result, *arguments):
        table = ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(result, ctypes.c_void_p, *arguments)(table[slot])

    ole = ctypes.OleDLL("ole32")
    mfplat, mf = ctypes.WinDLL("mfplat"), ctypes.WinDLL("mf")
    ole.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    ole.CoUninitialize.argtypes = []
    ole.CoUninitialize.restype = None
    ole.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    ole.CoTaskMemFree.restype = None
    mfplat.MFStartup.argtypes = [ctypes.c_ulong, ctypes.c_ulong]
    mfplat.MFCreateAttributes.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint]
    mf.MFEnumDeviceSources.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p)), ctypes.POINTER(ctypes.c_uint)]
    attributes = ctypes.c_void_p()
    devices = ctypes.POINTER(ctypes.c_void_p)()
    count = ctypes.c_uint()
    initialized = started = False
    try:
        # 若当前线程已有另一种 COM apartment，则沿用且不代替它关闭 COM。
        try:
            ole.CoInitializeEx(None, 0)
            initialized = True
        except OSError as exc:
            if (exc.winerror or 0) & 0xffffffff != 0x80010106:
                raise
        checked(mfplat.MFStartup(0x20070, 0))
        started = True
        checked(mfplat.MFCreateAttributes(ctypes.byref(attributes), 1))
        source_type = guid("c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3")
        video_type = guid("8ac3587a-4ae7-42d8-99e0-0a6013eef90f")
        # IMFAttributes：SetGUID 位于 vtable 24，GetAllocatedString 位于 13。
        checked(method(attributes, 24, ctypes.c_long, ctypes.POINTER(GUID), ctypes.POINTER(GUID))(
            attributes, ctypes.byref(source_type), ctypes.byref(video_type)))
        checked(mf.MFEnumDeviceSources(attributes, ctypes.byref(devices), ctypes.byref(count)))

        def string(pointer, key):
            key, value, length = guid(key), ctypes.c_void_p(), ctypes.c_uint()
            try:
                checked(method(pointer, 13, ctypes.c_long, ctypes.POINTER(GUID), ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_uint))(
                    pointer, ctypes.byref(key), ctypes.byref(value), ctypes.byref(length)))
                return ctypes.wstring_at(value, length.value)
            finally:
                if value:
                    ole.CoTaskMemFree(value)

        return [CameraDevice(i,
                             string(devices[i], "60d0e559-52f8-4fa2-bbce-acdb34a8ec01"),
                             string(devices[i], "58f0aad8-22bf-4f8a-bb3d-d2c4978c6e2f"))
                for i in range(count.value)]
    finally:
        if devices:
            for i in range(count.value):
                method(devices[i], 2, ctypes.c_ulong)(devices[i])
            ole.CoTaskMemFree(devices)
        if attributes:
            method(attributes, 2, ctypes.c_ulong)(attributes)
        if started:
            mfplat.MFShutdown()
        if initialized:
            ole.CoUninitialize()


def _matching_msmf_index(device: CameraDevice, dshow: list[CameraDevice]) -> int | None:
    """优先匹配设备路径，名称仅在两个后端中都唯一时才可用。"""
    devices = _msmf_devices()
    path = device.path.removeprefix("@device_pnp_").casefold()
    matches = [d.index for d in devices if path and path == d.path.casefold()]
    if len(matches) == 1:
        return matches[0]
    matches = [d.index for d in devices if d.name == device.name]
    if len(matches) == 1 and sum(d.name == device.name for d in dshow) == 1:
        return matches[0]
    return None


def _fourcc(capture) -> str:
    """仅用于 DirectShow，MSMF 的格式数值可能表示解码后的输出布局。"""
    value = int(capture.get(cv2.CAP_PROP_FOURCC))
    return "".join(chr((value >> (8 * i)) & 255) for i in range(4)).strip("\x00")


def _probe(capture, stopped, fps: float):
    """排除两次启动读取后测量真实帧间隔，同时验证 uint8 BGR 通路。"""
    times = []
    target = max(3, min(18, math.ceil(fps * 0.6)))
    frame = None
    i = 0
    while True:
        if stopped():
            raise InterruptedError("已取消摄像头初始化")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError("模式设置后无法读取图像")
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError("摄像头未返回 uint8 三通道 BGR 图像")
        if i >= 2:
            times.append(time.perf_counter())
            elapsed = times[-1] - times[0]
            # 至少观察0.6秒，避免启动缓存的瞬时突发误判为高帧率；低速设备约2秒后结束采样。
            if len(times) >= 3 and ((len(times) >= target and elapsed >= 0.6) or elapsed >= 2.0):
                break
        i += 1
    measured = (len(times) - 1) / max(times[-1] - times[0], 1e-6)
    return frame, measured


class _DeviceUnavailable(RuntimeError):
    """设备打不开时不重复遍历同一后端的所有分辨率。"""


def _open_mode(mode: CameraMode, api: int, index: int, stopped, strict: bool, measure: bool):
    """打开并验证实际帧；完整检测测速，快速复用只校验帧与生效参数。"""
    if stopped():
        raise InterruptedError("已取消摄像头初始化")
    capture = cv2.VideoCapture(index, api)
    try:
        if stopped():
            raise InterruptedError("已取消摄像头初始化")
        if not capture.isOpened():
            raise _DeviceUnavailable("设备无法打开，可能被其他程序占用或未授权")
        # 修改宽度可能同时改变高度；逐项回读，只设置尚未符合的参数，减少
        # MSMF重复重建采集链路。回读仅用于省去重复设置，最终仍验证真实帧。
        for prop, value in ((cv2.CAP_PROP_FRAME_WIDTH, mode.width),
                            (cv2.CAP_PROP_FRAME_HEIGHT, mode.height),
                            (cv2.CAP_PROP_FPS, mode.fps)):
            if stopped():
                raise InterruptedError("已取消摄像头初始化")
            current = float(capture.get(prop))
            if not math.isfinite(current) or abs(current - value) > 0.01:
                capture.set(prop, value)
        # DirectShow修改尺寸/FPS可能重置格式，FOURCC最后处理。MSMF回报22等
        # 输出格式值，不能据此判定传输格式相同，仍需请求驱动公布的格式。
        if mode.fourcc and (api != cv2.CAP_DSHOW or _fourcc(capture) != mode.fourcc):
            capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*mode.fourcc))
        # 驱动未必支持缓冲大小；真正的单帧缓存由LatestFrameCapture实现。
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if measure:
            frame, measured = _probe(capture, stopped, mode.fps)
        else:
            if stopped():
                raise InterruptedError("已取消摄像头初始化")
            ok, frame = capture.read()
            if not ok or frame is None:
                raise RuntimeError("模式设置后无法读取图像")
            measured = None
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise RuntimeError("摄像头未返回 uint8 三通道 BGR 图像")
        actual_h, actual_w = frame.shape[:2]
        if strict and (actual_w, actual_h) != (mode.width, mode.height):
            raise RuntimeError(f"驱动未采用请求尺寸，实际 {actual_w}×{actual_h}")
        if api == cv2.CAP_DSHOW and strict and _fourcc(capture) != mode.fourcc:
            raise RuntimeError(f"驱动未采用 {mode.fourcc}，实际 {_fourcc(capture)}")
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not math.isfinite(fps) or fps <= 0:
            fps = mode.fps
        if strict and abs(fps - mode.fps) > max(0.1, mode.fps * 0.02):
            raise RuntimeError(f"驱动回报 {fps:g} FPS，与公布模式不一致")
        if strict and measured is not None and measured < min(10.0, mode.fps * 0.4):
            raise RuntimeError(f"标称 {mode.fps:g} FPS，但短测仅 {measured:.2f} FPS")
        if stopped():
            raise InterruptedError("已取消摄像头初始化")
        return capture, frame, fps, measured
    except BaseException:
        # 取消初始化也必须释放，不能只捕获普通协商错误。
        capture.release()
        raise


def open_camera(index: int, width: int, height: int, stopped, notify):
    """首次比选、后续复用、异常重协商；返回采集器、验证帧和标称FPS。

    notify接收进度字符串或CameraInfo。每次核对设备清单和后端身份，缓存
    不保存句柄/序号；两个后端不会同时持有同一台相机，停止后照常释放。
    """
    global _inventory_snapshot
    started = time.perf_counter()
    device = CameraDevice(index, f"Camera {index}")
    modes, notes, failures = [], [], []
    key = ""
    snapshot = None
    apis = [(cv2.CAP_DSHOW, index, "DirectShow")] if os.name == "nt" else [(cv2.CAP_ANY, index, "默认后端")]
    if stopped():
        raise InterruptedError("已取消摄像头初始化")
    if os.name == "nt":
        try:
            devices = _dshow_devices()
            if not 0 <= index < len(devices):
                raise RuntimeError(f"未枚举到摄像头 {index}")
            device = devices[index]
            snapshot = tuple(devices)
            key = device.path.casefold()  # 缺少唯一路径时不缓存，避免同名设备误复用。
            with _cache_lock:
                if snapshot != _inventory_snapshot:
                    invalidate_camera_cache()
                    _inventory_snapshot = snapshot
                known = _capability_cache.get(key) if key else None
            modes = list(known) if known is not None else _device_modes(device)
            with _cache_lock:
                if key and _inventory_snapshot == snapshot:
                    _capability_cache[key] = tuple(modes)
            try:
                msmf_index = _matching_msmf_index(device, devices)
                if msmf_index is not None:
                    # 常见优胜者MSMF最后测试，胜出时可直接保留已打开对象。
                    apis.append((cv2.CAP_MSMF, msmf_index, "MSMF"))
                else:
                    notes.append("无法确认 MSMF 设备身份，仅使用 DirectShow")
            except (OSError, RuntimeError) as exc:
                notes.append(f"MSMF 设备匹配不可用：{exc}")
        except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
            invalidate_camera_cache()
            key = ""
            notes.append(f"能力枚举不可用（{exc}），按目标尺寸兼容协商，未验证最高帧率")

    def candidates():
        """能力未知时请求保守30 FPS；不把请求值当成设备最高帧率。"""
        return rank_modes(modes, width, height) if modes else [
            CameraMode(width, height, 30.0, "MJPG"), CameraMode(width, height, 30.0, "")]

    def finish(result, mode, api, label, kind, messages):
        """转交设备所有权并发布结构化信息；摘要不混入格式和长诊断文本。"""
        capture, frame, fps, measured = result
        try:
            actual_h, actual_w = frame.shape[:2]
            fmt = (f"BGR 输出（格式值 {capture.get(cv2.CAP_PROP_FOURCC):g}）"
                   if api == cv2.CAP_MSMF else (_fourcc(capture) or "默认格式"))
            details = list(messages)
            if (actual_w, actual_h) != (width, height):
                details.append("目标原生模式不可用或协商未成功，已采用其他采集尺寸")
            if fps < 15:
                details.append("警告：当前采用低帧率模式")
            info = CameraInfo(device.name, label, actual_w, actual_h, width, height,
                              fps, fmt, kind, time.perf_counter() - started, measured,
                              notes=tuple(dict.fromkeys(details)))
            notify(info)
            with _cache_lock:
                if key and modes and _inventory_snapshot == snapshot:
                    if kind != "快速复用":
                        _successful_modes[key, mode] = _CachedMode(api, tuple(dict.fromkeys(messages)))
                    _target_modes[key, width, height] = mode
            return capture, frame, fps
        except BaseException:
            capture.release()
            raise

    # 原生采集模式相同时跨模型复用；曾回退的目标也记住成功模式。
    ordered = candidates()
    with _cache_lock:
        remembered = _target_modes.get((key, width, height), ordered[0])
        cached = _successful_modes.get((key, remembered)) if key else None
    if cached is not None:
        notify(f"正在复用摄像头配置 · {device.name} · {remembered.width}×{remembered.height}")
        try:
            matching = [a for a in apis if a[0] == cached.api]
            if not matching:
                raise RuntimeError("当前设备清单无法确认缓存后端的身份")
            api, device_index, label = matching[0]
            if remembered not in modes:
                raise RuntimeError("缓存模式不在当前设备能力中")
            result = _open_mode(remembered, api, device_index, stopped, True, False)
            messages = notes + [f"上次完整检测：{message}" for message in cached.notes]
            return finish(result, remembered, api, label, "快速复用", messages)
        except InterruptedError:
            raise
        except Exception as exc:
            # 只对失效设备清缓存，并重新查询能力；不继续使用旧清单盲目比选。
            failures.append(f"缓存配置失效，重新协商：{exc}")
            with _cache_lock:
                _capability_cache.pop(key, None)
                for table in (_successful_modes, _target_modes):
                    for cache_key in list(table):
                        if cache_key[0] == key:
                            del table[cache_key]
            notify(f"缓存校验失败，正在重新协商摄像头 · {device.name}")
            try:
                modes = _device_modes(device)
            except (ImportError, OSError, RuntimeError, subprocess.TimeoutExpired) as refresh_error:
                modes = []
                key = ""
                notes.append(f"重新查询能力失败（{refresh_error}），按目标尺寸兼容协商")
            with _cache_lock:
                if key and _inventory_snapshot == snapshot:
                    _capability_cache[key] = tuple(modes)
            ordered = candidates()

    unavailable = set()
    for mode in ordered:
        if stopped():
            raise InterruptedError("已取消摄像头初始化")
        best_nominal = max((m.fps for m in modes if (m.width, m.height) == (mode.width, mode.height)), default=mode.fps)
        if mode.fps < best_nominal * 0.5:
            failures.append(f"跳过 {mode.width}×{mode.height} {mode.fourcc} {mode.fps:g} FPS 低速模式")
            continue
        tested, held, held_api = [], None, None
        try:
            for api, device_index, label in apis:
                if api in unavailable:
                    continue
                if held is not None:
                    held[0].release()
                    held = None
                notify(f"正在协商摄像头 · {device.name} · 测试 {label} {mode.width}×{mode.height}")
                try:
                    held = _open_mode(mode, api, device_index, stopped, bool(modes), True)
                    held_api = api
                    tested.append((min(held[3], held[2]), api, device_index, label, held[3]))
                except InterruptedError:
                    raise
                except Exception as exc:
                    if isinstance(exc, _DeviceUnavailable):
                        unavailable.add(api)
                    failures.append(f"{label} {mode.width}×{mode.height}/{mode.fourcc}: {exc}")
            if not tested:
                continue
            fastest = max(item[0] for item in tested)
            tested.sort(key=lambda item: (item[0] < fastest * 0.95, item[1] != cv2.CAP_MSMF, -item[0]))
            for _, api, device_index, label, measured in tested:
                try:
                    if held is not None and held_api == api:
                        result, held = held, None
                    else:
                        if held is not None:
                            held[0].release()
                            held = None
                        # 优胜者已释放才重开；验证实际帧，不再重复完整测速。
                        reopened = _open_mode(mode, api, device_index, stopped, bool(modes), False)
                        result = (*reopened[:3], measured)
                except InterruptedError:
                    raise
                except Exception as exc:
                    failures.append(f"{label} 重新打开失败：{exc}")
                    continue
                kind = "异常后完整检测" if cached is not None else "完整检测"
                return finish(result, mode, api, label, kind, notes + failures)
        finally:
            # 成功时句柄已转交；任何取消/异常都释放仍由协商函数持有的设备。
            if held is not None:
                held[0].release()
    raise RuntimeError("摄像头模式协商失败：" + "；".join(dict.fromkeys(notes + failures)))


class LatestFrameCapture:
    """相机读取线程拥有 VideoCapture，消费者只取不同序号的最新完整原始帧。

    暂停推理不会暂停采集，应用层原始帧缓存始终最多一帧。停止仅设置事件，
    不从其他线程强行 release 正在 read 的设备，避免驱动崩溃或死锁。
    """
    def __init__(self, capture, stopped, on_wait=None):
        self._capture = capture
        self._external_stop = stopped
        self._on_wait = on_wait
        self._stop = threading.Event()
        self._condition = threading.Condition()
        self._frame = None
        self._sequence = self._consumed = 0
        self._failure = None
        self._closed = False
        self._times = deque(maxlen=61)
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def _run(self):
        """唯一读取/释放者；交换帧引用后立即释放锁，不阻塞推理。"""
        try:
            while not self._stop.is_set() and not self._external_stop():
                ok, frame = self._capture.read()
                if not ok or frame is None:
                    raise RuntimeError("摄像头读取失败，请检查 USB 连接或设备占用")
                with self._condition:
                    self._frame = frame
                    self._sequence += 1
                    self._times.append(time.perf_counter())
                    self._condition.notify_all()
        except Exception as exc:
            with self._condition:
                self._failure = exc
                self._condition.notify_all()
        finally:
            try:
                self._capture.release()
            finally:
                with self._condition:
                    self._closed = True
                    self._condition.notify_all()

    def read(self):
        """等待新序号，不重复提交上一帧；错误与超时交给工作线程明确报告。"""
        with self._condition:
            deadline = time.monotonic() + 5.0
            while self._sequence <= self._consumed:
                if self._stop.is_set() or self._external_stop():
                    raise InterruptedError("已结束摄像头读取")
                if self._failure:
                    raise CameraCaptureError(f"摄像头采集线程异常：{self._failure}") from self._failure
                if self._closed:
                    return False, None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise CameraCaptureError("摄像头超过 5 秒未返回新帧，请检查连接或驱动")
                self._condition.wait(min(0.1, remaining))
            if self._stop.is_set() or self._external_stop():
                raise InterruptedError("已结束摄像头读取")
            self._consumed = self._sequence
            return True, self._frame

    def capture_fps(self) -> float | None:
        """实际读取完成间隔的滑动统计，与模型处理和界面绘制 FPS 分开。"""
        with self._condition:
            if len(self._times) < 2:
                return None
            return (len(self._times) - 1) / max(self._times[-1] - self._times[0], 1e-6)

    def request_stop(self):
        """可由界面线程调用，只唤醒等待者，不并发操作驱动对象。"""
        self._stop.set()
        with self._condition:
            self._condition.notify_all()

    def release(self):
        """等待所有权线程释放；异常驱动长期阻塞时提示并继续安全等待。"""
        self.request_stop()
        started = time.monotonic()
        warned = False
        while self._thread.is_alive():
            self._thread.join(0.1)
            if not warned and time.monotonic() - started >= 4:
                warned = True
                if self._on_wait:
                    self._on_wait("正在等待摄像头驱动释放，请检查 USB 连接")
