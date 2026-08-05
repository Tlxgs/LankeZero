"""
蒙特卡洛树搜索
"""
import numpy as np
import math
import os
from game import PASS_MOVE

_torch_ready = False

def _ensure_torch():
    """延迟加载 torch：GUI 走 TRT numpy 路径时完全不加载 torch（省 ~1GB 内存）。
    只有回退到 torch 推理时才导入并设置全局性能开关。"""
    global _torch_ready
    import torch
    import torch.nn.functional as F
    if not _torch_ready:
        # ---- 推理/训练性能优化（全局生效）----
        torch.backends.cudnn.benchmark = True          # cuDNN 自动选择最优卷积算法
        torch.set_float32_matmul_precision('high')     # 矩阵乘允许 TF32（torch 内部）
        _torch_ready = True
    return torch, F

from hyperparams import (BOARD_SIZE, C_PUCT, NUM_SIMULATIONS, TEMPERATURE, KOMI,
                         DIRICHLET_ALPHA, DIRICHLET_EPSILON, VIRTUAL_LOSS,
                         BATCH_SIZE_MCTS, MCTS_CAP, OMP_NUM_THREADS,
                         TRT_WORKSPACE, TRT_OPT_LEVEL, TRT_FP16, ALPHA)
os.environ["OMP_NUM_THREADS"] = str(OMP_NUM_THREADS)
os.environ["OMP_WAIT_POLICY"] = "PASSIVE"
os.environ["OMP_DYNAMIC"] = "FALSE"


# ============================================================
# 推理部分公共基类（torch 加载 + 单/批量推理 + pass bonus）
# ============================================================
def _setup_dll_paths():
    """注册 CUDA/TensorRT/cuDNN 运行时 DLL 搜索路径（进程级，与启动方式无关）。
    TensorRTModel 与 calibrate.py 共用。"""
    dll_dirs = [
        r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
        r'D:\software\NVIDIA\cudnn-windows-x86_64-8.9.7.29_cuda12-archive\bin',  # cuDNN 8.9.7
        r'C:\Program Files\NVIDIA',                      # cuDNN 8.x 备份位置
        r'D:\software\tensorrt\TensorRT-8.6.1.6\lib',   # TensorRT 8.6
    ]
    # pip 版 NVIDIA 运行库自动探测（nvidia-cudnn-cu11/cu12、nvidia-cuda-runtime-cu11 等）
    try:
        import site as _site, glob as _glob
        for _sp in _site.getsitepackages():
            for _b in _glob.glob(os.path.join(_sp, 'nvidia', '*', 'bin')):
                dll_dirs.append(_b)
    except Exception:
        pass
    # torch 自带 CUDA/cuDNN 运行库（torch/lib，兜底；不 import torch，避免 GUI 进程加载 ~1GB）
    try:
        import site as _site, glob as _glob
        for _sp in _site.getsitepackages():
            for _d in _glob.glob(os.path.join(_sp, 'torch', 'lib')):
                dll_dirs.append(_d)
    except Exception:
        pass
    for d in dll_dirs:
        if os.path.isdir(d):
            try:
                os.add_dll_directory(d)
            except Exception:
                pass
            try:
                # PATH 兜底：部分加载路径不认 add_dll_directory
                os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')
            except Exception:
                pass
    return dll_dirs


class TensorRTModel:
    """onnxruntime 推理封装（优先 TensorRT EP，自动回退 CUDA/CPU EP）。
    接口与 torch 模型一致：__call__(x) -> (policy, value, ownership)，均为 torch 张量。
    依赖 onnxruntime-gpu；未安装或加载失败时由调用方回退 torch 推理。"""
    _executor_logged = False

    def __init__(self, onnx_path, device='cuda', provider='auto'):
        """provider: 'auto'/'trt'=TRT 优先（训练/评估，最快）；'cuda'=CUDA EP（GUI 默认，无引擎构建/缓存，
        杜绝 TRT 缓存损坏导致的静默原生崩溃）；'cpu'=纯 CPU。均带自动降级。"""
        _setup_dll_paths()
        import onnxruntime as ort
        # 日志压到 ERROR：每局新建会话，INT64→INT32 等 WARNING 会刷屏（0=VERBOSE 1=INFO 2=WARNING 3=ERROR 4=FATAL）
        try:
            ort.logging.set_default_logger_severity(3)
        except Exception:
            pass
        available = ort.get_available_providers()
        cache_dir = os.path.join(os.path.dirname(onnx_path), 'trt_cache')
        trt_opts = {
            'device_id': 0,
            'trt_fp16_enable': TRT_FP16,
            'trt_engine_cache_enable': True,
            'trt_engine_cache_path': cache_dir,
            'trt_max_workspace_size': TRT_WORKSPACE,    # 256MB workspace：构建期峰值内存减半，降低4GB显存下构建崩溃概率
            'trt_builder_optimization_level': TRT_OPT_LEVEL,  # 3级：构建更快、内存更小，推理性能影响很小（引擎已缓存）
            'trt_timing_cache_enable': True,            # 加速后续引擎重建
            'trt_timing_cache_path': os.path.join(cache_dir, 'timing.cache'),
        }
        sess_options = ort.SessionOptions()
        sess_options.log_severity_level = 3
        # 引擎缓存新鲜度自检：onnx 变更（如 13路→19路）后旧引擎会导致 ORT 原生崩溃，宁可删除重建
        if provider in ('auto', 'trt'):
            self._check_engine_cache(onnx_path, cache_dir, trt_opts)
        # 分级尝试：TRT EP → CUDA EP → CPU EP（版本不匹配时自动降级，避免 ORT 内部报错刷屏）
        candidates = []
        if provider in ('auto', 'trt') and 'TensorrtExecutionProvider' in available:
            candidates.append([('TensorrtExecutionProvider', trt_opts), 'CUDAExecutionProvider', 'CPUExecutionProvider'])
        if provider in ('auto', 'trt', 'cuda') and 'CUDAExecutionProvider' in available:
            candidates.append(['CUDAExecutionProvider', 'CPUExecutionProvider'])
        candidates.append(['CPUExecutionProvider'])
        self.sess = None
        last_err = None
        for prov in candidates:
            try:
                self.sess = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=prov)
                # 会话创建后立即试跑一次：提前暴露引擎/输入形状不兼容（Python 层错误），失败换下一个执行器
                self._warmup_run()
                # ORT 会内部静默丢弃加载失败的 EP；只打印一次实际生效的执行器，避免每局刷屏
                if not TensorRTModel._executor_logged:
                    print(f'[TensorRTModel] 推理执行器: {self.sess.get_providers()}')
                    TensorRTModel._executor_logged = True
                break
            except Exception as e:
                last_err = e
        if self.sess is None:
            raise last_err
        # 打印 onnx 输入形状（诊断"模型与棋盘尺寸不符"类问题）
        try:
            _inp = self.sess.get_inputs()[0]
            print(f'[TensorRTModel] onnx输入: {_inp.name} shape={_inp.shape}')
        except Exception:
            pass
        self.device = device

    @staticmethod
    def _check_engine_cache(onnx_path, cache_dir, trt_opts):
        """onnx 变更（mtime/size）或构建选项变更时删除旧引擎缓存，
        避免 ORT 加载不兼容引擎导致的原生崩溃（静默闪退）。"""
        import json
        meta_path = os.path.join(cache_dir, 'engine_meta.json')
        try:
            st = os.stat(onnx_path)
            sig = {
                'onnx': os.path.basename(onnx_path),
                'mtime': st.st_mtime,
                'size': st.st_size,
                'workspace': trt_opts.get('trt_max_workspace_size'),
                'opt_level': trt_opts.get('trt_builder_optimization_level'),
            }
            fresh = os.path.exists(meta_path) and json.load(open(meta_path, encoding='utf-8')) == sig
            if not fresh:
                import shutil
                if os.path.isdir(cache_dir):
                    shutil.rmtree(cache_dir, ignore_errors=True)
                os.makedirs(cache_dir, exist_ok=True)
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(sig, f)
                print('[TensorRTModel] onnx 已变更，已清空旧 TRT 引擎缓存（将重建）')
        except Exception:
            pass

    def _warmup_run(self):
        """会话创建后立即用空输入试跑一次：提前暴露引擎/输入形状问题（失败抛异常 → 换下一个执行器）。"""
        try:
            inp = self.sess.get_inputs()[0]
            shape = []
            for s in inp.shape:
                shape.append(1 if not isinstance(s, int) or s <= 0 else s)
            self.sess.run(None, {inp.name: np.zeros(tuple(shape), dtype=np.float32)})
        except Exception as e:
            raise RuntimeError(f'推理试跑失败: {type(e).__name__}: {e}') from e

    def eval(self):
        return self

    def __call__(self, x):
        # 兼容接口：接受 torch 或 numpy (B,6,H,W)，返回 torch 张量（与 torch 模型一致）
        if hasattr(x, 'detach'):          # torch 张量（不 import torch 也能识别）
            x = x.detach().cpu().numpy()
        out = self.sess.run(None, {'input': x})
        import torch
        return tuple(torch.from_numpy(o).to(self.device) for o in out)

    def run_numpy(self, x):
        """快速路径：numpy 直进直出（跳过 torch 往返），返回 (logits, value, ownership) numpy"""
        return self.sess.run(None, {'input': x})

    def get_policy_value_ownership(self, state, legal_mask=None, device='cpu'):
        """与 PolicyValueNet 同名接口一致（GUI估值/归属/胜率显示用）：
        返回 (policy, value, ownership, win_logit)：policy(362,) numpy、value float、
        ownership(19,19) numpy ∈(-1,1)、win_logit float（胜率头 logit，tanh 后 = 当前玩家胜率 ±1）"""
        if hasattr(state, 'detach'):
            x = state.detach().cpu().numpy()
        else:
            x = np.asarray(state, dtype=np.float32)
        if x.ndim == 3:
            x = x[np.newaxis]
        outs = self.sess.run(None, {'input': x})
        logits, value, ownership_raw = outs[0], outs[1], outs[2]
        win_logit = float(outs[3][0, 0]) if len(outs) > 3 else float('nan')  # 旧3输出 ONNX → logit缺失(NaN)，GUI显示50%
        if legal_mask is not None:
            mask = np.asarray(legal_mask, dtype=np.float32)
            logits = logits.copy()
            logits[:, mask == 0] = -1e4
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        policy = (e / e.sum(axis=1, keepdims=True))[0]
        v = float(value[0, 0])
        ownership = np.tanh(ownership_raw[0, 0])
        return policy, v, ownership, win_logit   # win_logit 已在上方转为标量 float

# 搜索树节点容量上限（numba 数组越界写会导致静默原生崩溃，必须给足余量）：
# 训练默认 80万（≤1000模拟/步 × 361子节点 ≈43万，2倍余量）；GUI 按 GUI_MAX_SIMS×n²×1.2 传更大 cap（见 gui.py）
# 统一配置：hyperparams.MCTS_CAP（保留 _CAP 别名，供外部脚本引用）
_CAP = MCTS_CAP

class _BaseMCTS:
    def __init__(self, c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                 temperature=TEMPERATURE, alpha=ALPHA, onnx_path='model.onnx', device='cpu',
                 dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON,
                 board_size=BOARD_SIZE, batch_size=BATCH_SIZE_MCTS, provider='auto', cap=_CAP):
        self.c_puct = c_puct
        self._cap = cap
        self.num_simulations = num_simulations
        self.temperature = temperature
        self.alpha = alpha   # MCTS保守系数（GUI可运行时调整）
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.board_size = board_size
        self.batch_size = batch_size
        self.pass_idx = board_size * board_size

        model_path = onnx_path.replace('.onnx', '.pt')
        if not os.path.exists(model_path):
            model_path = 'model.pt'
        self._model_path = model_path   # TRT 失败时降级 torch 用
        self.device = device
        # 优先 TensorRT/onnxruntime 推理（TRT 走纯 numpy 路径，不需要 torch）；
        # 只有 TRT 加载失败才回退 torch（延迟导入，GUI 场景省 ~1GB 内存）
        self.model = None
        if os.path.exists(onnx_path):
            try:
                self.model = TensorRTModel(onnx_path, device=self.device, provider=provider)
            except Exception:
                self.model = None
        if self.model is None:
            torch, _ = _ensure_torch()
            from model import PolicyValueNet
            self.device = device if torch.cuda.is_available() else 'cpu'
            self.model = PolicyValueNet.load_model(model_path, device=self.device)
            self.model.eval()

    def _fallback_to_torch(self, err):
        """TRT 推理失败 → 永久降级 torch 推理（GUI 不闪退、搜索不中断）。"""
        if not isinstance(self.model, TensorRTModel):
            return
        torch, _ = _ensure_torch()
        from model import PolicyValueNet
        try:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = PolicyValueNet.load_model(self._model_path, device=self.device)
            self.model.eval()
            print(f'[MCTS] TRT 推理失败，已自动降级 torch 推理: {type(err).__name__}: {err}')
        except Exception as e2:
            self.model = None
            print(f'[MCTS] 降级 torch 推理也失败: {type(e2).__name__}: {e2}')

    def _inference(self, state, legal_mask):
        if isinstance(self.model, TensorRTModel):
            try:
                out = self.model.run_numpy(state.astype(np.float32, copy=False)[np.newaxis])
                logits, value = out[0], out[1]   # 兼容旧3输出/新4输出 ONNX（胜率logit由_batch_inference使用）
                v = float(value[0, 0])      # value = 期望领地目差（无界，领地头求和）
                return self._process(logits[0], legal_mask, v), v
            except Exception as e:
                # TRT 引擎损坏/输入形状不匹配 → 永久降级 torch（防止搜索每步崩溃）
                self._fallback_to_torch(e)
                if self.model is None:
                    raise
        torch, _ = _ensure_torch()
        with torch.inference_mode():
            t = torch.from_numpy(state).unsqueeze(0).to(self.device)
            logits, value, ownership, _ = self.model(t)
            v = value[0, 0].item()          # value = 期望领地目差（无界，领地头求和）
        policy = self._process(logits[0], legal_mask, v)
        return policy, v


    def _batch_inference(self, states, legal_masks):
        n = len(states)
        if isinstance(self.model, TensorRTModel):
            try:
                # TRT/ORT 快速路径：全程 numpy，跳过 torch 转换与 GPU 往返
                x = np.stack(states).astype(np.float32, copy=False)
                out = self.model.run_numpy(x)
                logits, values = out[0], out[1]   # 兼容旧3输出/新4输出 ONNX
                wls = out[3][:, 0] if len(out) > 3 else np.zeros(n, dtype=np.float32)  # 胜率logit（旧3输出ONNX无 →0）
                mask = np.stack(legal_masks)
                logits[mask == 0] = -1e4
                logits -= logits.max(axis=1, keepdims=True)
                p = np.exp(logits)
                p /= p.sum(axis=1, keepdims=True)
                v = values[:, 0]                    # 期望领地目差（无界，领地头求和）
                for i in range(n):
                    self._pass_bonus(p[i], legal_masks[i], v[i])
                return list(p), v, wls
            except Exception as e:
                # TRT 引擎损坏/输入形状不匹配 → 永久降级 torch（防止搜索每步崩溃）
                self._fallback_to_torch(e)
                if self.model is None:
                    raise
        torch, F = _ensure_torch()
        with torch.inference_mode():
            batch = torch.from_numpy(np.stack(states)).to(self.device)
            mask_t = torch.from_numpy(np.stack(legal_masks)).to(self.device)
            logits, values, ownership, win_logits = self.model(batch)
            logits = logits.masked_fill(mask_t == 0, -1e4)
            probs = F.softmax(logits, dim=1)
            p = probs.cpu().numpy()
            v = values[:, 0].cpu().numpy()      # 期望领地目差（无界，领地头求和）
            wls = win_logits[:, 0].cpu().numpy()   # 胜率logit
        for i in range(n):
            self._pass_bonus(p[i], legal_masks[i], v[i])    # pass bonus 仍用有界胜率
        return list(p), v, wls

    def _process(self, logits, legal_mask, value):
        arr = logits.cpu().numpy().copy() if hasattr(logits, 'cpu') else logits.copy()
        if legal_mask is not None:
            arr[legal_mask == 0] = -1e8
        arr -= arr.max()
        arr = np.exp(arr)
        arr /= arr.sum()
        self._pass_bonus(arr, legal_mask, value)
        return arr

    @staticmethod
    def _pass_bonus(policy, legal_mask, value):
        pidx = BOARD_SIZE * BOARD_SIZE
        if legal_mask is None or legal_mask[pidx] != 1:
            return
        p = policy[pidx]
        p +=0.005
        policy[pidx] = p
        policy /= policy.sum()


try:
    from numba import njit
    from game import _make_move, _legal_moves_and_mask, _HAS_NUMBA as _GAME_HAS_NUMBA
    _HAS_NUMBA = True
except Exception:
    _HAS_NUMBA = False

if _HAS_NUMBA and _GAME_HAS_NUMBA:
    @njit(cache=True)
    def _select_child(cmove, cnode, cstart, ccount, visits, vsum, priors, vloss,
                      node, c_puct, move_count, n):
        start = cstart[node]
        end = start + ccount[node]
        if start == end:
            return -1
        if visits[node] > 0:
            pv = vsum[node] / visits[node]
        else:
            pv = 0.0

        # ---- Q min-max 归一化（KataGo风格）：适配无界混合值，探索项与Q同尺度 ----
        qmin = 1e18
        qmax = -1e18
        for s in range(start, end):
            ch = cnode[s]
            vis = visits[ch]
            q = -vsum[ch] / vis if vis > 0 else pv
            if q < qmin:
                qmin = q
            if q > qmax:
                qmax = q
        qdiff = qmax - qmin
        qscale = 0.0 if qdiff < 1e-6 else 1.0 / qdiff

        sqrt_n = math.sqrt(1.0 + visits[node] + vloss[node])
        late = move_count > 300
        b = 0.0002 * (move_count - 300) if late else 0.0
        pass_move = n * n
        best_slot = start
        best_score = -1e18
        for s in range(start, end):
            ch = cnode[s]
            vis = visits[ch]
            # 子节点自身价值（视角取反）；未访问用父均值基准
            q = -vsum[ch] / vis if vis > 0 else pv
            qn = 0.5 if qdiff < 1e-6 else (q - qmin) * qscale   # 归一化到[0,1]
            score = qn + c_puct * priors[ch] * sqrt_n / (1.0 + vis + vloss[ch])
            if late and cmove[s] == pass_move:
                score += b
            if score > best_score:
                best_score = score
                best_slot = s
        return best_slot

    # ---------- 选择路径（等价原 _simulate_select）----------
    @njit(cache=True)
    def _select_np(board, st, go, fp, cap,
                   parent, cstart, ccount, cmove, cnode,
                   visits, vsum, priors, vloss, expanded,
                   root, c_puct, vl, n, path):
        node = root
        path_len = 0
        path[path_len] = node
        path_len += 1
        vloss[node] += vl
        while expanded[node] == 1 and ccount[node] > 0:
            s = _select_child(cmove, cnode, cstart, ccount, visits, vsum, priors, vloss,
                              node, c_puct, st[5], n)
            if s < 0:
                for i in range(path_len):
                    vloss[path[i]] -= vl
                return -1, 0, path_len
            move = cmove[s]
            child = cnode[s]
            if move == n * n:
                r = -1
                c = -1
            else:
                r = move // n
                c = move % n
            _make_move(board, st, go, fp, r, c, cap)
            vloss[child] += vl
            path[path_len] = child
            path_len += 1
            node = child
        if go[0]:
            return node, 1, path_len      # 终局
        return node, 0, path_len          # 叶节点（待展开）

    # ---------- 回溯+展开（等价原 _simulate_backup）----------
    @njit(cache=True)
    def _backup_np(board, st, go, fp, cap,
                   parent, cstart, ccount, cmove, cnode,
                   visits, vsum, priors, vloss, expanded,
                   leaf, path, path_len, value, policy, has_policy,
                   legal, legal_cap, vl, n, next_id_ptr, cpool_ptr):
        for i in range(path_len - 1, -1, -1):
            node = path[i]
            vloss[node] -= vl
            visits[node] += 1
            vsum[node] += value
            value = -value
        if has_policy and expanded[leaf] == 0:
            legal_len = _legal_moves_and_mask(board, st, n, legal, legal_cap)
            cstart[leaf] = cpool_ptr[0]
            for i in range(legal_len):
                mv = legal[i]
                new_id = next_id_ptr[0]
                next_id_ptr[0] += 1
                parent[new_id] = leaf
                cstart[new_id] = -1
                ccount[new_id] = 0
                visits[new_id] = 0
                vsum[new_id] = 0.0
                priors[new_id] = policy[mv]
                vloss[new_id] = 0
                expanded[new_id] = 0
                slot = cpool_ptr[0]
                cpool_ptr[0] += 1
                cmove[slot] = mv
                cnode[slot] = new_id
            ccount[leaf] = legal_len
            expanded[leaf] = 1

    # ---------- 压缩：把当前根子树重排到数组头部，回收死节点 ----------
    @njit(cache=True)
    def _compact(parent, cstart, ccount, cmove, cnode, visits, vsum, priors, vloss, expanded,
                 root, n, cap, next_id_ptr, cpool_ptr,
                 scratch_id, scratch_order, scratch_stack, scratch_cmove, scratch_cnode):
        new_id = scratch_id
        new_id[:] = -1
        order = scratch_order
        stack = scratch_stack
        new_id[root] = 0
        order[0] = root
        norder = 1
        top = 0
        stack[top] = root
        top += 1
        while top > 0:
            top -= 1
            node = stack[top]
            for s in range(cstart[node], cstart[node] + ccount[node]):
                ch = cnode[s]
                if new_id[ch] == -1:
                    new_id[ch] = norder
                    order[norder] = ch
                    norder += 1
                    stack[top] = ch
                    top += 1
        t_cmove = scratch_cmove
        t_cnode = scratch_cnode
        t_cstart = np.zeros(norder, dtype=np.int32)
        t_ccount = np.zeros(norder, dtype=np.int32)
        t_parent = np.empty(norder, dtype=np.int32)
        t_visits = np.empty(norder, dtype=np.int32)
        t_vsum = np.empty(norder, dtype=np.float32)
        t_priors = np.empty(norder, dtype=np.float32)
        t_vloss = np.empty(norder, dtype=np.int32)
        t_expanded = np.empty(norder, dtype=np.int8)
        pool = 0
        for i in range(norder):
            node = order[i]
            t_parent[i] = parent[node]
            t_cstart[i] = pool
            t_ccount[i] = ccount[node]
            t_visits[i] = visits[node]
            t_vsum[i] = vsum[node]
            t_priors[i] = priors[node]
            t_vloss[i] = vloss[node]
            t_expanded[i] = expanded[node]
            for s in range(cstart[node], cstart[node] + ccount[node]):
                t_cmove[pool] = cmove[s]
                t_cnode[pool] = new_id[cnode[s]]
                pool += 1
        for i in range(norder):
            parent[i] = -1 if i == 0 else new_id[t_parent[i]]
            cstart[i] = t_cstart[i]
            ccount[i] = t_ccount[i]
            visits[i] = t_visits[i]
            vsum[i] = t_vsum[i]
            priors[i] = t_priors[i]
            vloss[i] = t_vloss[i]
            expanded[i] = t_expanded[i]
        for i in range(pool):
            cmove[i] = t_cmove[i]
            cnode[i] = t_cnode[i]
        next_id_ptr[0] = norder
        cpool_ptr[0] = pool

    # ---------- 供 gui 读取的轻量视图（只读数组树，不复制）----------
    class _NodeView:
        __slots__ = ('m', 'id')
        def __init__(self, m, nid):
            self.m = m
            self.id = nid
        @property
        def visit_count(self):
            return int(self.m._visits[self.id])
        @property
        def prior_prob(self):
            return float(self.m._priors[self.id])
        def get_value(self, default=0.0):
            v = self.m._visits[self.id]
            return default if v == 0 else float(self.m._vsum[self.id] / v)

    class _RootView:
        __slots__ = ('m',)
        def __init__(self, m):
            self.m = m
        @property
        def children(self):
            m = self.m
            root = m._root
            n = m.board_size
            d = {}
            for s in range(m._cstart[root], m._cstart[root] + m._ccount[root]):
                mv = m._cmove[s]
                move = PASS_MOVE if mv == n * n else (mv // n, mv % n)
                d[move] = _NodeView(m, m._cnode[s])
            return d

    class MCTS(_BaseMCTS):
        def __init__(self, c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                     temperature=TEMPERATURE, alpha=ALPHA, onnx_path='model.onnx', device='cpu',
                     dirichlet_alpha=DIRICHLET_ALPHA, dirichlet_epsilon=DIRICHLET_EPSILON,
                     board_size=BOARD_SIZE, batch_size=BATCH_SIZE_MCTS, provider='auto', cap=_CAP):
            super().__init__(c_puct, num_simulations, temperature, alpha, onnx_path, device,
                             dirichlet_alpha, dirichlet_epsilon, board_size, batch_size, provider, cap)
            CAP = cap
            self._parent = np.full(CAP, -1, dtype=np.int32)
            self._cstart = np.zeros(CAP, dtype=np.int32)
            self._ccount = np.zeros(CAP, dtype=np.int32)
            self._visits = np.zeros(CAP, dtype=np.int32)
            self._vsum = np.zeros(CAP, dtype=np.float32)
            self._priors = np.zeros(CAP, dtype=np.float32)
            self._vloss = np.zeros(CAP, dtype=np.int32)
            self._expanded = np.zeros(CAP, dtype=np.int8)
            self._cmove = np.zeros(CAP, dtype=np.int32)
            self._cnode = np.zeros(CAP, dtype=np.int32)
            self._next = np.zeros(1, dtype=np.int32)
            self._cpool = np.zeros(1, dtype=np.int32)
            self._path = np.zeros(2 * self.board_size * self.board_size + 4, dtype=np.int32)
            self._legal = np.zeros(self.board_size * self.board_size + 1, dtype=np.int32)
            self._mask = np.zeros(self.board_size * self.board_size + 1, dtype=np.float32)
            self._s_id = np.full(CAP, -1, dtype=np.int32)
            self._s_order = np.zeros(CAP, dtype=np.int32)
            self._s_stack = np.zeros(CAP, dtype=np.int32)
            self._s_cmove = np.zeros(CAP, dtype=np.int32)
            self._s_cnode = np.zeros(CAP, dtype=np.int32)
            self._root = None
            self._root_key = None   # 根节点对应的局面指纹（current_player + board）
            self.root = None

        def _idx(self, move):
            if move == PASS_MOVE or (move[0] == -1 and move[1] == -1):
                return self.pass_idx
            return move[0] * self.board_size + move[1]

        def _game_key(self, game):
            """局面指纹：当前玩家 + 棋盘字节。用于校验树根与局面一致（评估对局双模型交替时防止错位）"""
            return (game.current_player, game.board.tobytes())

        def init_root(self, game):
            self._root_key = self._game_key(game)
            legal_moves, legal_mask = game.get_legal_moves_and_mask()
            if not legal_moves:
                self._root = None
                self.root = None
                return
            state = game.get_canonical_state()
            policy, _ = self._inference(state, legal_mask)
            moves = list(legal_moves)
            probs = np.array([policy[self._idx(m)] for m in moves], dtype=np.float32)
            noise = np.random.dirichlet([self.dirichlet_alpha] * len(moves))
            mixed = (1 - self.dirichlet_epsilon) * probs + self.dirichlet_epsilon * noise
            mixed /= mixed.sum()
            self._parent[:] = -1
            self._cstart[:] = 0
            self._ccount[:] = 0
            self._visits[:] = 0
            self._vsum[:] = 0.0
            self._priors[:] = 0.0
            self._vloss[:] = 0
            self._expanded[:] = 0
            self._next[0] = 1
            self._cpool[0] = 0
            root = 0
            self._expanded[root] = 1
            self._ccount[root] = 0
            for i, m in enumerate(moves):
                new_id = self._next[0]
                self._next[0] += 1
                self._parent[new_id] = root
                self._cstart[new_id] = -1
                self._ccount[new_id] = 0
                self._visits[new_id] = 0
                self._vsum[new_id] = 0.0
                self._priors[new_id] = mixed[i]
                self._vloss[new_id] = 0
                self._expanded[new_id] = 0
                slot = self._cpool[0]
                self._cpool[0] += 1
                self._cmove[slot] = self._idx(m)
                self._cnode[slot] = new_id
            self._ccount[root] = len(moves)
            self._root = root
            self.root = _RootView(self)

        def update_root(self, game, last_move):
            if self._root is None:
                self.init_root(game)
                return
            root = self._root
            mv = self._idx(last_move)
            found = -1
            for s in range(self._cstart[root], self._cstart[root] + self._ccount[root]):
                if self._cmove[s] == mv:
                    found = self._cnode[s]
                    break
            if found >= 0:
                self._root = found
                self._root_key = self._game_key(game)   # 根节点推进到新局面
                self._compact()          # 每步压缩一次，回收死分支，保持容量
            else:
                self.init_root(game)
            self.root = _RootView(self) if self._root is not None else None

        def _compact(self):
            if self._root is None:
                return
            _compact(self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                     self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                     self._root, self.board_size, len(self._parent), self._next, self._cpool,
                     self._s_id, self._s_order, self._s_stack, self._s_cmove, self._s_cnode)
            self._root = 0  
        def simulate_batch(self, game, n):
            if self._root is None:
                return
            remaining = n
            while remaining > 0:
                # ---- 节点容量安全网 ----
                # 19路 每展开一个叶子最多新建 n² 个子节点，长搜索会撑爆容量数组：
                # 越界写 → 静默原生崩溃（无任何 Python 报错）。
                # 注意：_compact 只对"落子后的子树"有效；当前根即整棵树时压缩无效，必须重建树根。
                # 容量按场景区分（训练 _CAP=80万 / GUI GUI_CAP≈433万），此处是超出容量后的兜底。
                if self._next[0] > self._cap - 20000:
                    self._compact()
                    if self._next[0] > self._cap - 20000:
                        self.init_root(game)   # 重建树根（1次推理），搜索继续不中断
                        print(f'[MCTS] 树节点达到容量上限，已重建树根（剩余 {remaining} 模拟）')
                        if self._root is None:
                            return
                batch = min(self.batch_size, remaining)
                sims = []
                for _ in range(batch):
                    gc = game.copy()
                    leaf, is_term, plen = _select_np(
                        gc.board, gc._st, gc._go, gc._fp, gc._cap,
                        self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                        self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                        self._root, self.c_puct, VIRTUAL_LOSS, self.board_size, self._path)
                    if leaf >= 0:
                        sims.append((leaf, gc, is_term, self._path[:plen].copy(), plen))
                nn = [i for i, s in enumerate(sims) if not s[2]]
                policies = [None] * len(sims)
                values = [None] * len(sims)
                wls = [None] * len(sims)   # 叶子胜率logit（网络输出）
                if nn:
                    states = [sims[i][1].get_canonical_state() for i in nn]
                    masks = [sims[i][1].get_legal_moves_and_mask()[1] for i in nn]
                    ps, vs, xw = self._batch_inference(states, masks)
                    for j, i in enumerate(nn):
                        policies[i] = ps[j]
                        values[i] = vs[j]
                        wls[i] = xw[j]
                for i, (leaf, gc, is_term, path, plen) in enumerate(sims):
                    if is_term:
                        if gc.winner is None:
                            gc._compute_winner()
                        value = 0.0 if gc.winner == 0 else gc.get_terminal_value(gc.current_player)
                        policy = np.zeros(0, dtype=np.float32)
                        has_policy = 0
                    else:
                        value = values[i]
                        policy = policies[i]
                        has_policy = 1
                        # 胜率logit 与归属头目差（含贴目，平局=0）线性混合；混合值进 vsum，
                        # 由 _select_child 的 Q min-max 归一化（KataGo式）适配尺度后用于选择。
                        # α=0 → 纯胜率logit；α=1 → 纯目差（归属头求和 + komi，当前玩家视角）。
                        score = value - KOMI * gc.current_player
                        value = (1.0 - self.alpha) * wls[i] + self.alpha * score
                    _backup_np(
                        gc.board, gc._st, gc._go, gc._fp, gc._cap,
                        self._parent, self._cstart, self._ccount, self._cmove, self._cnode,
                        self._visits, self._vsum, self._priors, self._vloss, self._expanded,
                        leaf, path, plen, value, policy, has_policy,
                        self._legal, self._mask, VIRTUAL_LOSS, self.board_size,
                        self._next, self._cpool)
                remaining -= batch

        def get_move_probs(self, game, temp=None, target_visits=1):
            if temp is None:
                temp = self.temperature
            if self._root is None or self._root_key != self._game_key(game):
                self.init_root(game)   # 局面不匹配（如评估对局双模型交替）则重建树
            if self._root is None:
                return {}
            current = int(self._visits[self._root])
            remaining = max(1, target_visits - current)
            while remaining > 0:
                b = min(self.batch_size, remaining)
                self.simulate_batch(game, b)
                remaining -= b
            return self._get_visit_probs(temp)[0]

        def _get_visit_probs(self, temp):
            root = self._root
            n = self.board_size
            move_visits = {}
            for s in range(self._cstart[root], self._cstart[root] + self._ccount[root]):
                mv = self._cmove[s]
                move = PASS_MOVE if mv == n * n else (mv // n, mv % n)
                move_visits[move] = int(self._visits[self._cnode[s]])
            if temp <= 0.1:
                best = max(move_visits, key=move_visits.get)
                return {best: 1.0}, best
            moves = list(move_visits.keys())
            visits = np.array([move_visits[m] for m in moves])
            probs = visits ** (1.0 / temp)
            probs /= probs.sum()
            return {m: p for m, p in zip(moves, probs)}, moves[np.argmax(probs)]
