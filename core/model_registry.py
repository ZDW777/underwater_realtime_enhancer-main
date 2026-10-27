from __future__ import annotations

import importlib.util
import inspect
import re
import sys
import tomllib
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any


DEFAULT_OUTPUT_RANGE = "0_1"
_OUTPUT_RANGE_ALIASES = {
    "[0,1]": "0_1",
    "0-1": "0_1",
    "[-1,1]": "minus1_1",
    "-1_1": "minus1_1",
    "-1-1": "minus1_1",
    "[0,255]": "0_255",
    "0-255": "0_255",
}
_warned_auto_ranges: set[tuple[str, str]] = set()


def normalize_output_range(value: Any) -> str:
    """统一范围名称；auto 仅作为使用默认值的旧配置标记。"""
    if not isinstance(value, str):
        raise TypeError("output_range 必须是字符串")
    name = value.strip().lower().replace(" ", "")
    name = _OUTPUT_RANGE_ALIASES.get(name, name)
    if name not in {"0_1", "minus1_1", "0_255", "auto"}:
        raise ValueError(f"不支持的 output_range：{value}；请选择 0_1、minus1_1 或 0_255")
    return name


def resolve_output_range(
    value: Any,
    default_range: str = DEFAULT_OUTPUT_RANGE,
    *,
    source: str = "output_range",
) -> str:
    """将配置解析为固定范围，不读取图像极值或逐帧猜测。"""
    name = normalize_output_range(value)
    if name != "auto":
        return name
    fallback = normalize_output_range(default_range)
    if fallback == "auto":
        fallback = DEFAULT_OUTPUT_RANGE
    # 每个来源在相同默认值下只提示一次，刷新或逐帧转换不会反复刷屏。
    key = (source, fallback)
    if key not in _warned_auto_ranges:
        warnings.warn(
            f'{source}="auto" 已停用范围猜测，将固定使用默认输出范围 {fallback}。'
            "如模型采用其他范围，请显式配置 output_range。",
            UserWarning,
            stacklevel=2,
        )
        _warned_auto_ranges.add(key)
    return fallback


@dataclass(frozen=True)
class ModelSpec:
    """记录一个已成功导入的模型入口及其加载要求。"""
    name: str
    file: Path
    module: ModuleType
    requires_weights: bool
    constructor: type
    init_kwargs: dict[str, Any] = field(default_factory=dict)
    # 模型识别时确定范围，PyTorch 和 TensorRT 直接读取，不修改模型模块元数据。
    output_range: str = DEFAULT_OUTPUT_RANGE
    # 仅 PyTorch 自动匹配使用：边缘补齐到该倍数，输出裁回原尺寸。
    # 默认8兼容当前模型；更特殊的网络可在集中配置覆盖，无需修改网络代码。
    input_multiple: int = 8

    def create(self):
        """使用识别出的主网络类和构造参数创建模型。"""
        from torch import nn

        network_class = self.constructor
        if not inspect.isclass(network_class) or not issubclass(network_class, nn.Module):
            raise TypeError(f"{self.file.name} 的主网络必须是 torch.nn.Module 子类")
        try:
            model = network_class(**self.init_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"模型 {self.name} 构造失败（{self.file.name}）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        if not isinstance(model, nn.Module):
            raise TypeError(f"模型 {self.name} 的构造结果必须是 torch.nn.Module")
        return model


class ModelRegistry:
    """扫描 models/*.py，并把有效模型注册到界面下拉框。"""
    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self.registrations_file = models_dir.parent / "model_registrations.toml"
        self._models: dict[str, ModelSpec] = {}
        self._errors: dict[str, str] = {}

    @property
    def models(self) -> dict[str, ModelSpec]:
        return dict(self._models)

    @property
    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def _load_registrations(self, errors: dict[str, str]) -> tuple[dict[str, Any], str]:
        """集中配置可省略；文件错误单独报告，不影响其他模型的自动识别。"""
        if not self.registrations_file.exists():
            return {}, DEFAULT_OUTPUT_RANGE
        try:
            with self.registrations_file.open("rb") as file:
                data = tomllib.load(file)
            if set(data) - {"models", "defaults"}:
                raise ValueError("顶层只支持 [defaults] 和 [models.模型名] 配置表")
            registrations = data.get("models", {})
            if not isinstance(registrations, dict):
                raise TypeError("models 必须是配置表")
        except Exception as exc:
            errors[self.registrations_file.name] = f"{type(exc).__name__}: {exc}"
            return {}, DEFAULT_OUTPUT_RANGE

        # 全局默认项只影响未声明范围或显式使用 auto 的模型，不改变识别方式。
        default_range = DEFAULT_OUTPUT_RANGE
        try:
            defaults = data.get("defaults", {})
            if not isinstance(defaults, dict):
                raise TypeError("defaults 必须是配置表")
            if set(defaults) - {"output_range"}:
                raise ValueError("defaults 只支持 output_range 字段")
            default_range = resolve_output_range(
                defaults.get("output_range", DEFAULT_OUTPUT_RANGE),
                source=f"{self.registrations_file.name} [defaults].output_range",
            )
        except Exception as exc:
            label = f"{self.registrations_file.name} [defaults]"
            errors[label] = f"{type(exc).__name__}: {exc}"
        return registrations, default_range

    def _registration_path(self, name: str, options: dict[str, Any]) -> Path:
        """配置中的文件始终位于 models/，省略 file 时使用注册名.py。"""
        filename = options.get("file", f"{name}.py")
        if (
            not isinstance(filename, str)
            or not filename.endswith(".py")
            or "/" in filename
            or "\\" in filename
            or ":" in filename
        ):
            raise ValueError("file 必须是 models/ 下的单个 .py 文件名")
        return self.models_dir / filename

    @staticmethod
    def _validate_options(options: dict[str, Any]) -> None:
        allowed = {"file", "class_name", "init_kwargs", "requires_weights", "output_range", "input_multiple"}
        unknown = set(options) - allowed
        if unknown:
            raise ValueError(f"不支持的注册字段：{', '.join(sorted(unknown))}")
        if "class_name" in options:
            value = options["class_name"]
            if not isinstance(value, str) or not value.isidentifier():
                raise ValueError("class_name 必须是有效的 Python 类名")
        kwargs = options.get("init_kwargs", {})
        if not isinstance(kwargs, dict) or any(
            not isinstance(key, str) or not key.isidentifier() for key in kwargs
        ):
            raise TypeError("init_kwargs 必须是以参数名为键的配置表")
        if "requires_weights" in options and not isinstance(options["requires_weights"], bool):
            raise TypeError("requires_weights 必须是 true 或 false")
        if "output_range" in options:
            normalize_output_range(options["output_range"])
        if "input_multiple" in options:
            ModelRegistry._validate_input_multiple(options["input_multiple"])

    @staticmethod
    def _validate_input_multiple(value: Any) -> int:
        """拒绝布尔值、非整数与过大补边，避免自动匹配时误分配巨量显存。"""
        if type(value) is not int or not 1 <= value <= 1024:
            raise ValueError("input_multiple 必须是1至1024之间的整数")
        return value

    @staticmethod
    def _find_constructor(module: ModuleType, file: Path, options: dict[str, Any]):
        """使用集中配置指定的类名，未指定时按文件同名查找。"""
        from torch import nn

        class_name = options.get("class_name", file.stem)
        constructor = getattr(module, class_name, None)
        if constructor is None:
            raise RuntimeError(
                f"未找到主网络类 {class_name}；请让文件名与主网络类名一致，"
                "或在 model_registrations.toml 中指定 class_name"
            )
        if not inspect.isclass(constructor) or not issubclass(constructor, nn.Module):
            raise TypeError(f"主网络 {class_name} 必须是 torch.nn.Module 子类")
        if constructor is nn.Module or inspect.isabstract(constructor):
            raise TypeError(f"主网络 {class_name} 必须是可实例化的具体网络类")
        if constructor.__module__ != module.__name__:
            raise ValueError(f"主网络 {class_name} 必须在 {file.name} 中定义")

        # 刷新只校验签名，不实例化网络，也不在此申请 GPU 显存。
        try:
            inspect.signature(constructor).bind(**options.get("init_kwargs", {}))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{getattr(constructor, '__name__', '模型入口')} 无法按当前参数构造：{exc}。"
                "自动识别使用默认构造；必填参数请在 model_registrations.toml "
                "的 init_kwargs 中填写"
            ) from exc
        return constructor

    def scan(self) -> dict[str, ModelSpec]:
        """逐文件扫描；单个模型导入失败不会影响其他模型。"""
        found: dict[str, ModelSpec] = {}
        errors: dict[str, str] = {}
        self.models_dir.mkdir(parents=True, exist_ok=True)
        models_path = str(self.models_dir.resolve())
        if models_path not in sys.path:
            sys.path.insert(0, models_path)

        # 集中注册的文件不再重复自动注册；同一文件可配置多个不同名称的模型。
        requests: list[tuple[Path, str | None, dict[str, Any]]] = []
        claimed_files: set[Path] = set()
        registrations, default_range = self._load_registrations(errors)
        for name, options in registrations.items():
            label = f"{self.registrations_file.name} [{name}]"
            try:
                if not name.strip() or any(char in name for char in '/\\:*?"<>|'):
                    raise ValueError("注册名不能为空或包含路径、文件名非法字符")
                if not isinstance(options, dict):
                    claimed_files.add((self.models_dir / f"{name}.py").resolve())
                    raise TypeError("每个模型必须使用 [models.模型名] 配置表")
                file = self._registration_path(name, options)
                claimed_files.add(file.resolve())
                self._validate_options(options)
                requests.append((file, name, options))
            except Exception as exc:
                errors[label] = f"{type(exc).__name__}: {exc}"

        for file in sorted(self.models_dir.glob("*.py")):
            # 下划线开头的文件可作为公共网络层或辅助模块，不显示在界面中。
            if file.name.startswith("_") or file.resolve() in claimed_files:
                continue
            requests.append((file, None, {}))

        modules: dict[Path, ModuleType] = {}
        for file, registered_name, options in requests:
            label = file.name if registered_name is None else (
                f"{self.registrations_file.name} [{registered_name}] ({file.name})"
            )
            try:
                if not file.is_file():
                    raise FileNotFoundError(f"模型文件不存在：{file}")
                resolved_file = file.resolve()
                module = modules.get(resolved_file)
                if module is None:
                    safe_stem = re.sub(r"[^0-9a-zA-Z_]", "_", file.stem)
                    spec = importlib.util.spec_from_file_location(
                        f"underwater_user_model_{safe_stem}", file
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError(f"无法导入模型文件：{file.name}")
                    module = importlib.util.module_from_spec(spec)
                    # dataclass 等装饰器会在导入时查询 sys.modules。
                    previous_module = sys.modules.get(spec.name)
                    sys.modules[spec.name] = module
                    try:
                        spec.loader.exec_module(module)
                    except BaseException:
                        if previous_module is None:
                            sys.modules.pop(spec.name, None)
                        else:
                            sys.modules[spec.name] = previous_module
                        raise
                    modules[resolved_file] = module
                constructor = self._find_constructor(module, file, options)
                name = registered_name if registered_name is not None else file.stem
                if name in found:
                    raise ValueError(f"模型名 {name} 重复，已由 {found[name].file.name} 使用")
                # 在识别阶段解析输出范围，保存到各自独立的 ModelSpec；
                # 同一源码的不同注册项不会互相修改 OUTPUT_RANGE。
                if "output_range" in options:
                    range_value = options["output_range"]
                    range_source = f"{label}.output_range"
                else:
                    range_value = getattr(module, "OUTPUT_RANGE", default_range)
                    range_source = f"{file.name} ({name}).OUTPUT_RANGE"
                output_range = resolve_output_range(
                    range_value,
                    default_range,
                    source=range_source,
                )
                found[name] = ModelSpec(
                    name=name,
                    file=file,
                    module=module,
                    requires_weights=options.get(
                        "requires_weights", bool(getattr(module, "REQUIRE_WEIGHTS", True))
                    ),
                    constructor=constructor,
                    init_kwargs=dict(options.get("init_kwargs", {})),
                    output_range=output_range,
                    input_multiple=self._validate_input_multiple(
                        options.get("input_multiple", getattr(module, "INPUT_MULTIPLE", 8))
                    ),
                )
            except Exception as exc:
                # 保存详细错误供状态栏显示，同时继续扫描下一个文件。
                errors[label] = f"{type(exc).__name__}: {exc}"
                continue
        self._models = found
        self._errors = errors
        return self.models

    def get(self, name: str) -> ModelSpec:
        try:
            return self._models[name]
        except KeyError as exc:
            raise RuntimeError(f"找不到模型定义：{name}") from exc
