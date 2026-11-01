"""Optional TensorRT 8.x REFIT backend.

Build one engine per board size, then refit the current model weights into it
(no full rebuild).  The PyTorch GoEvaluator remains the default and is used as
an automatic fallback whenever TensorRT/engine/refit is unavailable.

Environment:
  GO_TRT_ENGINE_DIR  engine cache dir (default: <repo>/trt_engines)
  GO_TRT_LIB_DIR     directory(s) containing nvinfer/cuDNN 8 DLLs
  GO_TRT_PYTHONPATH  directory containing the `tensorrt` Python package
"""
import hashlib
import json
import os
import sys
import threading
import time
from datetime import datetime

import numpy as np
import torch

from evaluator import GoEvaluator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ENGINE_DIR = os.path.join(ROOT, "trt_engines")
GRAPH_VERSION = 1
DEFAULT_BATCH = 16
DEFAULT_PRECISION = "fp16"

_TRT = None
_TRT_ERR = None
_DLL_DIRS = []


def _noop_log(_msg):
    pass


def _add_dll_dir(path):
    if not path or not os.path.isdir(path):
        return
    if path not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        try:
            _DLL_DIRS.append(add(path))
        except OSError:
            pass


def _setup_runtime_paths():
    for item in (os.environ.get("GO_TRT_LIB_DIR"),):
        if item:
            for p in str(item).split(os.pathsep):
                _add_dll_dir(p.strip())
    _add_dll_dir(os.path.join(ROOT, "trt_runtime", "bin"))
    for item in (os.environ.get("GO_TRT_PYTHONPATH"),):
        if item:
            for p in str(item).split(os.pathsep):
                p = p.strip()
                if p and os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)
    local_py = os.path.join(ROOT, "trt_runtime", "python")
    if os.path.isdir(local_py) and local_py not in sys.path:
        sys.path.insert(0, local_py)


def import_tensorrt():
    """Import TensorRT once. Returns module or None (reason in last_error())."""
    global _TRT, _TRT_ERR
    if _TRT is not None:
        return _TRT if _TRT is not False else None
    _setup_runtime_paths()
    try:
        import tensorrt as trt  # noqa: F401
        _TRT = trt
        _TRT_ERR = None
    except Exception as exc:
        _TRT = False
        _TRT_ERR = "%s: %s" % (type(exc).__name__, exc)
    return _TRT if _TRT is not False else None


def last_error():
    return _TRT_ERR


def engine_dir(path=None):
    return os.path.abspath(path or os.environ.get("GO_TRT_ENGINE_DIR")
                            or DEFAULT_ENGINE_DIR)


def engine_paths(directory, board_size, batch=DEFAULT_BATCH,
                 precision=DEFAULT_PRECISION):
    base = "go_%d_b%d_%s_refit" % (int(board_size), int(batch), precision)
    return (os.path.join(directory, base + ".engine"),
            os.path.join(directory, base + ".json"))


def architecture_signature(model):
    return {
        "format_version": int(getattr(model, "FORMAT_VERSION", 0)),
        "architecture": str(getattr(model, "ARCHITECTURE", "")),
        "input_channels": int(getattr(model, "input_channels", 0)),
        "channels": int(getattr(model, "channels", 0)),
        "num_res_blocks": int(getattr(model, "num_res_blocks", 0)),
        "head_channels": int(getattr(model, "head_channels", 0)),
        "dropout_rate": float(getattr(model, "dropout_rate", 0.0)),
        "graph_version": GRAPH_VERSION,
    }


def architecture_hash(model):
    raw = json.dumps(architecture_signature(model), sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _device_info():
    if not torch.cuda.is_available():
        return "", ""
    try:
        cc = torch.cuda.get_device_capability(0)
        return (torch.cuda.get_device_name(0), "%d.%d" % (cc[0], cc[1]))
    except Exception:
        return "", ""


def _atomic_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _read_manifest(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("manifest 不是 JSON 对象")
    return data


def manifest_problem(manifest, model, board_size, batch, precision):
    """Return None if the cached engine can be used, else a short reason."""
    if int(manifest.get("board_size", -1)) != int(board_size):
        return "board_size 不匹配"
    if int(manifest.get("batch", -1)) != int(batch):
        return "batch 不匹配"
    if str(manifest.get("precision", "")) != str(precision):
        return "precision 不匹配"
    if int(manifest.get("graph_version", -1)) != GRAPH_VERSION:
        return "导出图版本变化"
    if manifest.get("arch") != architecture_hash(model):
        return "模型结构/通道数变化"
    trt = import_tensorrt()
    if trt is None:
        return "TensorRT 不可用（%s）" % (_TRT_ERR or "未知")
    if str(manifest.get("tensorrt_version", "")) != str(trt.__version__):
        return "TensorRT 版本变化（引擎 %s，当前 %s）" % (
            manifest.get("tensorrt_version"), trt.__version__)
    gpu, cc = _device_info()
    if manifest.get("gpu") and gpu and manifest.get("gpu") != gpu:
        return "GPU 变化"
    if manifest.get("compute_capability") and cc and \
            manifest.get("compute_capability") != cc:
        return "GPU 算力变化"
    names = manifest.get("weight_names")
    if not isinstance(names, list) or not names:
        return "manifest 缺少 weight_names"
    return None


class _PolicyOutputs(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        policy, value, _ownership, win = self.model(x)
        return policy, value, win


def build_engine(model, board_size, engine_dir_path=None,
                 batch=DEFAULT_BATCH, precision=DEFAULT_PRECISION,
                 force=False, log=_noop_log):
    """Export + build one REFIT engine for a board size. CPU model expected."""
    trt = import_tensorrt()
    if trt is None:
        raise RuntimeError("TensorRT 不可用: %s" % (_TRT_ERR or "未知"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")
    if int(batch) < 1:
        raise ValueError("batch 必须 > 0")
    directory = engine_dir(engine_dir_path)
    os.makedirs(directory, exist_ok=True)
    engine_path, manifest_path = engine_paths(
        directory, board_size, batch, precision)
    if os.path.exists(engine_path) and not force:
        raise FileExistsError("引擎已存在: %s（用 --force 覆盖）" % engine_path)

    import copy
    cpu_model = copy.deepcopy(model).cpu().eval()
    dummy = torch.zeros(int(batch), int(cpu_model.input_channels),
                        int(board_size), int(board_size), dtype=torch.float32)
    onnx_path = engine_path + ".onnx.tmp"
    tmp_engine = engine_path + ".tmp"
    log("[TensorRT] 导出 ONNX: %d 路 batch=%d %s" % (board_size, batch, precision))
    t0 = time.perf_counter()
    torch.onnx.export(
        _PolicyOutputs(cpu_model).eval(), (dummy,), onnx_path,
        opset_version=17,
        input_names=["state"], output_names=["policy", "value", "win"],
        dynamic_axes={"state": {0: "batch"},
                      "policy": {0: "batch"},
                      "value": {0: "batch"},
                      "win": {0: "batch"}},
        do_constant_folding=False,
    )
    try:
        import onnx
    except Exception as exc:
        raise RuntimeError("构建 TensorRT 引擎需要 onnx 包: %s" % exc)
    onnx_model = onnx.load(onnx_path)
    weight_names = [i.name for i in onnx_model.graph.initializer
                    if i.name.startswith("model.")]
    if not weight_names:
        raise RuntimeError("ONNX 中没有 model.* 权重，无法 refit")

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("TensorRT 解析 ONNX 失败:\n" + "\n".join(errors))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    config.set_flag(trt.BuilderFlag.REFIT)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("当前 GPU 不支持快速 FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    profile = builder.create_optimization_profile()
    shape = (1, int(cpu_model.input_channels), int(board_size), int(board_size))
    opt = (int(batch), int(cpu_model.input_channels), int(board_size), int(board_size))
    profile.set_shape("state", shape, opt, opt)
    config.add_optimization_profile(profile)
    log("[TensorRT] 构建引擎（首次可能 1–6 分钟）...")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 构建失败")
    with open(tmp_engine, "wb") as f:
        f.write(serialized)
    os.replace(tmp_engine, engine_path)
    build_sec = time.perf_counter() - t0
    gpu, cc = _device_info()
    manifest = {
        "format": 1,
        "engine": os.path.basename(engine_path),
        "board_size": int(board_size),
        "batch": int(batch),
        "precision": str(precision),
        "tensorrt_version": str(trt.__version__),
        "cuda_version": str(torch.version.cuda or ""),
        "gpu": gpu,
        "compute_capability": cc,
        "arch": architecture_hash(cpu_model),
        "architecture": architecture_signature(cpu_model),
        "graph_version": GRAPH_VERSION,
        "weight_names": weight_names,
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "build_seconds": round(build_sec, 2),
    }
    _atomic_json(manifest_path, manifest)
    for p in (onnx_path, tmp_engine):
        try:
            if os.path.exists(p):
                os.remove(p)
        except OSError:
            pass
    log("[TensorRT] 引擎已就绪: %s（%.1fs，%d 个可 refit 权重）"
        % (engine_path, build_sec, len(weight_names)))
    return engine_path, manifest_path


class TrtEngine:
    """Loaded TensorRT engine + reusable execution context + in-place refit."""

    def __init__(self, engine_path, manifest, model, log=_noop_log):
        trt = import_tensorrt()
        if trt is None:
            raise RuntimeError("TensorRT 不可用: %s" % (_TRT_ERR or "未知"))
        self.trt = trt
        self.engine_path = engine_path
        self.manifest = manifest
        self.log = log
        self.n = int(manifest["board_size"])
        self.batch = int(manifest["batch"])
        self.precision = str(manifest["precision"])
        self.weight_names = list(manifest["weight_names"])
        self.logger = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("无法反序列化引擎: %s" % engine_path)
        if not getattr(self.engine, "refittable", False):
            raise RuntimeError("引擎不是 REFIT 引擎: %s" % engine_path)
        self.context = self.engine.create_execution_context()
        try:
            self.context.active_optimization_profile = 0
        except Exception:
            pass
        self.input_idx = None
        self.tensors = []
        self._build_tensors()
        self.lock = threading.Lock()
        self._current_b = None
        self.refit_from_model(model)

    def _build_tensors(self):
        trt = self.trt
        input_channels = None
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            is_input = self.engine.binding_is_input(i)
            dtype = self.engine.get_binding_dtype(i)
            fmt = self.engine.get_binding_format(i)
            if dtype != trt.DataType.FLOAT or fmt != trt.TensorFormat.LINEAR:
                raise RuntimeError("不支持的绑定格式: %s dtype=%s format=%s"
                                   % (name, dtype, fmt))
            if is_input:
                self.input_idx = i
                shape = tuple(self.engine.get_binding_shape(i))
                if len(shape) != 4 or shape[1] <= 0:
                    raise RuntimeError("输入绑定形状异常: %s" % (shape,))
                input_channels = int(shape[1])
                tensor = torch.zeros((self.batch, input_channels,
                                      self.n, self.n), dtype=torch.float32,
                                     device="cuda")
            else:
                if name == "policy":
                    h = (self.batch, self.n * self.n + 1)
                elif name in ("value", "win"):
                    h = (self.batch, 1)
                else:
                    raise RuntimeError("未知输出绑定: %s" % name)
                tensor = torch.zeros(h, dtype=torch.float32, device="cuda")
            self.tensors.append({"name": name, "input": is_input,
                                 "t": tensor})
        if self.input_idx is None:
            raise RuntimeError("引擎没有输入绑定")
        self.input_channels = input_channels
        self.input_t = next(t["t"] for t in self.tensors if t["input"])
        self.outputs = {t["name"]: t["t"] for t in self.tensors if not t["input"]}
        if self.context.set_binding_shape(
                self.input_idx,
                (self.batch, input_channels, self.n, self.n)) is False:
            raise RuntimeError("set_binding_shape 失败")

    def refit_from_model(self, model):
        """Copy all model weights into the existing engine; no rebuild/context."""
        with self.lock:
            state = {k: v.detach().float().cpu().numpy()
                     for k, v in model.named_parameters()}
            dtype = np.float16 if self.precision == "fp16" else np.float32
            refitter = self.trt.Refitter(self.engine, self.logger)
            keep = []
            for name in self.weight_names:
                key = name[6:] if name.startswith("model.") else name
                if key not in state:
                    raise RuntimeError("权重缺失: %s（模型结构变化？）" % name)
                arr = np.ascontiguousarray(state[key], dtype=dtype)
                weights = self.trt.Weights(arr)
                keep.append((arr, weights))
                if not refitter.set_named_weights(name, weights):
                    raise RuntimeError("TensorRT 拒绝权重: %s" % name)
            missing = refitter.get_missing_weights()
            if missing:
                raise RuntimeError("refit 缺少权重: %s" % ", ".join(missing[:5]))
            if not refitter.refit_cuda_engine():
                raise RuntimeError("refit_cuda_engine 失败")
            torch.cuda.synchronize()
            # `keep` must outlive refit_cuda_engine; it does (local scope).

    def evaluate(self, states, legal_masks):
        trt = self.trt
        with self.lock:
            b = int(states.shape[0])
            if b <= 0:
                total = self.n * self.n + 1
                return (np.zeros((0, total), np.float32),
                        np.zeros((0,), np.float32),
                        np.zeros((0,), np.float32))
            if b > self.batch:
                raise ValueError("batch %d > 引擎上限 %d" % (b, self.batch))
            with torch.inference_mode():
                if self._current_b != b:
                    if self.context.set_binding_shape(
                            self.input_idx,
                            (b, self.input_channels, self.n, self.n)) is False:
                        raise RuntimeError("set_binding_shape(%d) 失败" % b)
                    self._current_b = b
                self.input_t[:b].copy_(torch.from_numpy(
                    np.ascontiguousarray(states, dtype=np.float32)))
                bindings = [int(t["t"].data_ptr()) for t in self.tensors]
                ok = self.context.execute_async_v2(
                    bindings=bindings,
                    stream_handle=torch.cuda.current_stream().cuda_stream)
                if not ok:
                    raise RuntimeError("execute_async_v2 失败")
                torch.cuda.current_stream().synchronize()
                logits = self.outputs["policy"][:b].float().cpu().numpy()
                values = self.outputs["value"][:b, 0].float().cpu().numpy()
                wins = self.outputs["win"][:b, 0].float().cpu().numpy()
            logits = np.array(logits, dtype=np.float32, copy=True)
            masks = legal_masks.astype(bool, copy=False)
            logits[~masks] = -1e9
            logits -= logits.max(axis=1, keepdims=True)
            np.exp(logits, out=logits)
            logits /= logits.sum(axis=1, keepdims=True)
            return logits, values.astype(np.float32, copy=False), \
                wins.astype(np.float32, copy=False)


class TrtEvaluator:
    """GoEvaluator-compatible wrapper with automatic PyTorch fallback."""

    def __init__(self, trt_engine, fallback, model, device, board_size,
                 max_batch, log=_noop_log):
        self._engine = trt_engine
        self._fallback = fallback
        self.model = model
        self.device = device
        self.n = int(board_size)
        self.max_batch = int(max_batch)
        self._active = True
        self._lock = threading.Lock()
        self._log = log

    def evaluate(self, states, legal_masks):
        if not self._active:
            return self._fallback.evaluate(states, legal_masks)
        try:
            states = np.asarray(states, dtype=np.float32)
            legal_masks = np.asarray(legal_masks, dtype=np.float32)
            if states.ndim == 3:
                states = states[None, ...]
            if legal_masks.ndim == 1:
                legal_masks = legal_masks[None, ...]
            if states.shape[0] == 0:
                total = self.n * self.n + 1
                return (np.zeros((0, total), np.float32),
                        np.zeros((0,), np.float32),
                        np.zeros((0,), np.float32))
            if states.shape[0] > self.max_batch:
                ps, vs, ws = [], [], []
                for i in range(0, states.shape[0], self.max_batch):
                    p, v, w = self.evaluate(states[i:i + self.max_batch],
                                            legal_masks[i:i + self.max_batch])
                    ps.append(p); vs.append(v); ws.append(w)
                return (np.concatenate(ps), np.concatenate(vs),
                        np.concatenate(ws))
            return self._engine.evaluate(states, legal_masks)
        except Exception as exc:
            self._deactivate("推理失败: %s" % exc)
            return self._fallback.evaluate(states, legal_masks)

    def refit_from_model(self, model):
        if not self._active:
            return False
        try:
            self._engine.refit_from_model(model)
            return True
        except Exception as exc:
            self._deactivate("refit 失败: %s" % exc)
            return False

    def _deactivate(self, reason):
        with self._lock:
            if self._active:
                self._active = False
                try:
                    self._log("[TensorRT] %s；已自动回退 PyTorch" % reason)
                except Exception:
                    pass


def make_evaluator(model, board_size=None, device="cuda",
                   max_batch=DEFAULT_BATCH, use_tensorrt=False,
                   trt_engine_dir=None, log=_noop_log):
    """Return TrtEvaluator when a compatible cached engine exists, else PyTorch."""
    board_size = int(board_size or getattr(model, "board_size", 19))
    if not use_tensorrt:
        return GoEvaluator(model, device=device, max_batch=max_batch)
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        log("[TensorRT] 未启用：需要 CUDA")
        return GoEvaluator(model, device=device, max_batch=max_batch)
    trt = import_tensorrt()
    if trt is None:
        log("[TensorRT] 未启用：%s" % (_TRT_ERR or "TensorRT 不可用"))
        return GoEvaluator(model, device=device, max_batch=max_batch)
    directory = engine_dir(trt_engine_dir)
    engine_path, manifest_path = engine_paths(
        directory, board_size, max_batch, DEFAULT_PRECISION)
    if not os.path.exists(engine_path) or not os.path.exists(manifest_path):
        log("[TensorRT] 未找到缓存引擎 %d 路: %s" % (board_size, engine_path))
        log("[TensorRT] 请先运行: python python/build_trt.py --board-size %d"
            % board_size)
        return GoEvaluator(model, device=device, max_batch=max_batch)
    try:
        manifest = _read_manifest(manifest_path)
        problem = manifest_problem(manifest, model, board_size, max_batch,
                                   DEFAULT_PRECISION)
    except Exception as exc:
        log("[TensorRT] 读取 manifest 失败: %s" % exc)
        return GoEvaluator(model, device=device, max_batch=max_batch)
    if problem:
        log("[TensorRT] 缓存引擎不可用（%s），回退 PyTorch" % problem)
        return GoEvaluator(model, device=device, max_batch=max_batch)
    fallback = GoEvaluator(model, device=device, max_batch=max_batch)
    try:
        engine = TrtEngine(engine_path, manifest, model, log=log)
        evaluator = TrtEvaluator(engine, fallback, model, device,
                                 board_size, max_batch, log=log)
        log("[TensorRT] 已启用 %d 路 REFIT 引擎（batch=%d, %s）"
            % (board_size, max_batch, manifest.get("precision")))
        return evaluator
    except Exception as exc:
        log("[TensorRT] 加载/refit 失败: %s；回退 PyTorch" % exc)
        return fallback
