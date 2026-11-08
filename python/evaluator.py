"""围棋神经网络推理器：任意棋盘尺寸、批量前向、FP16 + channels_last + CUDA Graph。

模型是全卷积，空间输入 + 独立全局向量；同一份权重可直接接收 7–23 的奇数路输入。
策略输出为 n*n+1（含 pass）。CUDA Graph 按棋盘尺寸分别缓存，互不干扰；
每个缓存同时保存 state 和 global 两个输入缓冲区。
"""
import threading

import numpy as np
import torch



def resolve_device(device="auto"):
    if device is None or device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if str(device).startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return str(device)


class GoEvaluator:
    """把 (B,INPUT_CHANNELS,n,n) 局面 + 全局向量 + 合法掩码变成 (策略, 目差, 胜率 logit)。

    同一实例可依次处理 7–23 的奇数路；每种尺寸单独捕获一张 CUDA Graph。
    捕获失败自动退回普通前向。
    """

    def __init__(self, model, device="auto", max_batch=16, amp=True,
                 channels_last=True, graph=True):
        self.device = resolve_device(device)
        self.model = model.to(self.device).eval()
        self.n = int(getattr(model, "board_size", 19))
        self.input_channels = int(getattr(model, "input_channels", 9))
        self.global_dim = int(getattr(model, "global_dim", 0))
        self.max_batch = max(1, int(max_batch))
        self.amp = bool(amp) and str(self.device).startswith("cuda")
        self.channels_last = (bool(channels_last)
                              and str(self.device).startswith("cuda"))
        if self.channels_last:
            self.model = self.model.to(memory_format=torch.channels_last)
        torch.backends.cudnn.benchmark = True
        self._lock = threading.Lock()
        self._graph_enabled = (bool(graph)
                               and str(self.device).startswith("cuda")
                               and hasattr(torch.cuda, "CUDAGraph"))
        self._graphs = {}
        self._graph_broken = set()
        self._graph_ptr = next(self.model.parameters()).data_ptr()

    # ---------------- CUDA Graph（每种 n 一张） ----------------
    def _capture(self, n, cap):
        try:
            self.model.eval()
            g_input = torch.zeros(
                (cap, self.input_channels, n, n),
                dtype=torch.float32, device=self.device)
            if self.channels_last:
                g_input = g_input.to(memory_format=torch.channels_last)
            g_global = torch.zeros(
                (cap, self.global_dim), dtype=torch.float32, device=self.device)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                with torch.inference_mode():
                    for _ in range(3):
                        with torch.autocast("cuda", dtype=torch.float16,
                                            enabled=self.amp):
                            self.model(g_input, g_global)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            g = torch.cuda.CUDAGraph()
            with torch.inference_mode():
                with torch.cuda.graph(g):
                    with torch.autocast("cuda", dtype=torch.float16,
                                        enabled=self.amp):
                        policy, value, _own, win = self.model(g_input, g_global)
            self._graphs[n] = {
                "graph": g, "input": g_input, "global": g_global,
                "policy": policy, "value": value, "win": win,
                "cap": cap,
            }
            return True
        except Exception:
            self._graph_broken.add(n)
            self._graphs.pop(n, None)
            return False

    def _graph_forward(self, x_np, g_np):
        b, _c, n, _w = x_np.shape
        if (not self._graph_enabled or n in self._graph_broken
                or b < 4 or b > self.max_batch):
            return None
        ptr = next(self.model.parameters()).data_ptr()
        if ptr != self._graph_ptr:
            self._graphs.clear()
            self._graph_broken.clear()
            self._graph_ptr = ptr
        cache = self._graphs.get(n)
        if cache is None or cache["cap"] < b:
            if not self._capture(n, self.max_batch):
                return None
            cache = self._graphs.get(n)
        try:
            with torch.inference_mode():
                src = torch.from_numpy(x_np)
                if self.channels_last:
                    src = src.to(memory_format=torch.channels_last)
                cache["input"][:b].copy_(src)
                cache["global"][:b].copy_(torch.from_numpy(
                    np.ascontiguousarray(g_np, dtype=np.float32)))
                cache["graph"].replay()
                logits = cache["policy"][:b].float().cpu().numpy()
                values = cache["value"][:b, 0].float().cpu().numpy()
                wins = cache["win"][:b, 0].float().cpu().numpy()
            return logits, values, wins
        except Exception:
            self._graph_broken.add(n)
            self._graphs.pop(n, None)
            return None

    # ---------------- 对外接口 ----------------
    @torch.inference_mode()
    def evaluate(self, states, legal_masks, global_features=None):
        """states: (B,INPUT_CHANNELS,n,n)；global_features: (B,GLOBAL_DIM)。

        返回 (probs, values, win_logits)，均为 numpy。values 是归属头逐点求和
        （≈ 当前行棋方视角的目差估计，含贴目）。
        """
        states = np.asarray(states, dtype=np.float32)
        legal_masks = np.asarray(legal_masks, dtype=np.float32)
        if states.ndim == 3:
            states = states[None, ...]
        if legal_masks.ndim == 1:
            legal_masks = legal_masks[None, ...]
        b, _c, n, _w = states.shape
        if global_features is None:
            global_features = np.zeros((b, self.global_dim), dtype=np.float32)
        else:
            global_features = np.asarray(global_features, dtype=np.float32)
            if global_features.ndim == 1:
                global_features = global_features[None, ...]
            if global_features.shape != (b, self.global_dim):
                raise ValueError("global_features 形状 %r != %r"
                                 % (global_features.shape, (b, self.global_dim)))
        total = n * n + 1
        if legal_masks.shape[-1] != total:
            raise ValueError("legal mask 长度 %d != %d" % (legal_masks.shape[-1], total))
        if b == 0:
            return (np.zeros((0, total), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32),
                    np.zeros((0,), dtype=np.float32))
        if b > self.max_batch:
            ps, vs, ws = [], [], []
            for i in range(0, b, self.max_batch):
                p, v, w, s = self.evaluate(states[i:i + self.max_batch],
                                           legal_masks[i:i + self.max_batch],
                                           global_features[i:i + self.max_batch])
                ps.append(p); vs.append(v); ws.append(w)
            return (np.concatenate(ps), np.concatenate(vs),
                    np.concatenate(ws))

        with self._lock:
            self.model.eval()
            out = self._graph_forward(states, global_features)
            if out is None:
                x = torch.as_tensor(states, device=self.device)
                gt = torch.as_tensor(global_features, device=self.device)
                if self.channels_last:
                    x = x.to(memory_format=torch.channels_last)
                with torch.autocast("cuda", dtype=torch.float16,
                                    enabled=self.amp):
                    logits_t, value_t, _own_t, win_t = self.model(x, gt)
                logits = logits_t.float().cpu().numpy()
                values = value_t[:, 0].float().cpu().numpy()
                wins = win_t[:, 0].float().cpu().numpy()
            else:
                logits, values, wins = out

            logits = np.array(logits, dtype=np.float32, copy=True)
            legal_masks = legal_masks.astype(bool, copy=False)
            logits[~legal_masks] = -1e9
            logits -= logits.max(axis=1, keepdims=True)
            np.exp(logits, out=logits)
            logits /= logits.sum(axis=1, keepdims=True)
            return (logits, values.astype(np.float32, copy=False),
                    wins.astype(np.float32, copy=False))