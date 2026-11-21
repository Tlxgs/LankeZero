"""本机围棋训练界面：单进程自对弈 + 训练 + 回放。

不再包含 --headless / 多 worker / ONNX 评估子进程——这些已归档到 legacy/server。
自对弈在后台线程运行，Tk 主线程只处理界面与队列；模型与搜索共用同一份权重。
"""
import os
import sys
import json
import queue
import random
import threading
import time
import traceback
import ctypes
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from array import array
from collections import deque
from datetime import datetime

import numpy as np

from game import PASS_MOVE
from hyperparams import (BOARD_SIZE, DATA_DIR, NUM_SIMULATIONS,
                         KOMI, C_PUCT, TEMPERATURE, BATCH_SIZE,
                         SAVE_INTERVAL, TRAIN_GAMES, LEARNING_RATE,
                         WEIGHT_DECAY, MOMENTUM, OPTIMIZER_NAME,
                         UI_UPDATE_BATCHES, SAVE_INTERVAL_BATCHES, ALPHA,
                         TRAIN_STEPS_PER_GAME, BATCH_SIZE_MCTS, VIRTUAL_LOSS,
                         DIRICHLET_ALPHA, DIRICHLET_EPSILON, GUI_MAX_SIMS,
                         PCR_ENABLED, PCR_PROB, PCR_FAST_DIV, RESIGN_THRESHOLD,
                         board_temperature_zero_for_area, LOG_DIR)
import shapes
from ui_utils import GameBoard, TrainingStatsPanel
from selfplay import play_one_game
from rules_adapter import GO_RULES

if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

MAX_REPLAY_GAMES = 200
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "train_gui_settings.json")


def snap_half_step(value, minimum=0.0):
    """把输入夹到 [minimum, +∞) 并对齐到 0.5 的整数倍（非法值按 minimum 处理）。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return float(minimum)
    if not np.isfinite(v) or v < minimum:
        v = float(minimum)
    return round(v * 2.0) / 2.0


def sample_komi(base, span, rng=None):
    """每局贴目：在 base±span 的 0.5 格点上均匀抽一个。

    span=0 时原样返回 base —— 保持「固定贴目」的旧行为。
    例：base=6.5、span=1.5 → 5.0/5.5/…/8.0 共 7 个值，各 1/7。
    base 本身不在 0.5 格点上时先对齐（数据加载器只接受 0.5 的整数倍）。
    """
    span = snap_half_step(span)
    if span <= 0.0:
        return float(base)
    base = round(float(base) * 2.0) / 2.0
    steps = int(round(span * 2.0))
    return base + 0.5 * int((random if rng is None else rng).randint(-steps, steps))


class GameResult:
    __slots__ = ("game_id", "moves", "players", "winner", "board_size", "hole_mask",
                 "score_diff", "start_stones", "resigned")

    def __init__(self, game_id, moves, winner, board_size=BOARD_SIZE, hole_mask=None,
                 score_diff=0.0, start_stones=None, resigned=False):
        self.game_id = game_id
        self.winner = winner
        self.score_diff = float(score_diff)      # 黑视角目差；认输/截断局为 0
        self.resigned = bool(resigned)           # 认输结束（没有终局目差/归属）
        self.board_size = int(board_size)
        self.hole_mask = (shapes.solid(self.board_size) if hole_mask is None
                          else np.asarray(hole_mask, dtype=bool))
        self.start_stones = (None if start_stones is None
                             else np.asarray(start_stones, dtype=np.int8))
        n = self.board_size + 1
        arr = array("h")
        pl = array("b")
        for m in moves:
            r, c = int(m[0]), int(m[1])
            arr.append(0 if (r < 0 or c < 0) else (r + 1) * n + (c + 1))
            # 落子方：让子开局的前几手全是黑，回放时必须按"摆子"处理，不能轮转
            pl.append(int(m[2]) if len(m) > 2 else 1)
        self.moves = arr
        self.players = pl

    def get_moves(self):
        n = self.board_size + 1
        return [((-1, -1) if code == 0 else (code // n - 1, code % n - 1))
                for code in self.moves]


def result_text(winner, score_diff=0.0, resigned=False):
    """对局结果文字：胜方 + 目差。score_diff 是黑视角目差，没有目差（认输/截断）时只显示胜方。"""
    if winner == 0:
        return "平局"
    side = "黑胜" if winner == 1 else "白胜"
    if resigned:
        return side + "（认输）"
    if abs(float(score_diff)) < 1e-6:
        return side
    return "%s %.1f 目" % (side, abs(float(score_diff)))


class TrainingGUI:
    def __init__(self, rules=None):
        self.rules = rules or GO_RULES
        self.ui = self.rules.ui
        self.root = tk.Tk()
        self.root.title("%s AI 训练" % self.ui.title)
        self.root.geometry("2000x1300")
        self.board_size = self.ui.default_size
        self.hole_mask = shapes.solid(self.board_size)
        self.start_stones = None          # 自定义开局的棋子（None = 空盘）
        self.shape_name = "整盘"
        self.trainer = None
        self.is_training = False
        self.worker = None
        self.update_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.game_history = deque(maxlen=MAX_REPLAY_GAMES)
        self.current_history_index = -1
        self.current_game_moves = []
        self.current_game_players = []
        self.current_step = 0
        self._log_file = None
        self._settings = self._read_settings()
        self._create_widgets()
        self._apply_saved_settings()
        self._draw_empty_board(self.start_stones)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(100, self._process_queue)
        self.root.mainloop()

    def _create_widgets(self):
        main = ttk.Frame(self.root, padding="10")
        main.pack(fill=tk.BOTH, expand=True)
        left = ttk.LabelFrame(main, text="训练控制", padding="10")
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        self._create_model_settings(left)
        self._create_mcts_settings(left)
        self._create_training_settings(left)
        self._create_optimizer_settings(left)
        self._create_data_settings(left)
        self._create_buttons(left)
        right = ttk.Frame(main)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        self._create_board_area(right)
        self._create_status_area(right)
        self._draw_empty_board()

    # ---------------- 参数持久化 ----------------
    @staticmethod
    def _read_settings():
        try:
            with open(SETTINGS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _collect_settings(self):
        return {
            "version": 1,
            "board_size": int(self.board_size_var.get()),
            "shape": shapes.to_text_board(self.hole_mask, self.start_stones),
            "komi": float(self.komi_var.get()) if hasattr(self, "komi_var") else KOMI,
            "komi_range": (snap_half_step(self.komi_range_var.get())
                           if hasattr(self, "komi_range_var") else 0.0),
            "device": self.device_var.get(),
            "simulations": int(self.simulations_var.get()),
            "mcts_batch": int(self.mcts_batch_var.get()),
            "c_puct": float(self.c_puct_var.get()),
            "temperature": float(self.temperature_var.get()),
            "alpha": float(self.alpha_var.get()),
            "vloss": int(self.vloss_var.get()),
            "pcr": bool(self.pcr_var.get()),
            "pcr_prob": float(self.pcr_prob_var.get()),
            "dir_alpha": float(self.dir_alpha_var.get()),
            "dir_eps": float(self.dir_eps_var.get()),
            "resign": float(self.resign_var.get()),
            "train_only": bool(self.train_only_var.get()),
            "games": int(self.total_var.get()),
            "batch": int(self.batch_size_var.get()),
            "steps": int(self.steps_var.get()),
            "save_interval": int(SAVE_INTERVAL),
            "ui_interval": int(UI_UPDATE_BATCHES),
            "save_batch": int(self.save_batch_var.get()),
            "freeze": self.freeze_var.get(),
            "block_drop": self.block_drop_var.get(),
            "drop_blocks": self.drop_blocks_var.get(),
            "optimizer": self.optimizer_var.get(),
            "lr": float(self.lr_var.get()),
            "momentum": float(self.momentum_var.get()),
            "wd": float(self.wd_var.get()),
            "load_data": bool(self.load_data_var.get()),
            "save_data": bool(self.save_data_var.get()),
            "tensorrt": bool(self.tensorrt_var.get()),
            "load_mode": self.load_mode_var.get(),
            "data_dir": self.data_dir_var.get(),
        }

    def _save_settings(self):
        """原子写入本机 GUI 设置；失败只记日志，不影响训练/关闭。"""
        try:
            tmp = SETTINGS_PATH + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(self._collect_settings(), f, ensure_ascii=False, indent=2)
            os.replace(tmp, SETTINGS_PATH)
        except Exception as e:
            if hasattr(self, "status_text"):
                self._log_message("[设置] 保存失败: %s" % e)

    @staticmethod
    def _set_var(var, value, kind):
        try:
            if kind == "bool":
                if isinstance(value, str):
                    value = value.strip().lower() not in ("", "0", "false", "no", "off")
                var.set(bool(value))
            elif kind == "int":
                var.set(int(value))
            elif kind == "float":
                var.set(float(value))
            else:
                var.set(str(value))
        except (TypeError, ValueError, tk.TclError):
            pass

    def _apply_saved_settings(self):
        s = self._settings
        if not s:
            return
        bs = s.get("board_size")
        try:
            bs = int(bs)
        except (TypeError, ValueError):
            bs = None
        if bs in self.ui.board_sizes:
            self.board_size_var.set(bs)
            if bs != self.board_size:
                self.board_size = bs
                self.hole_mask = shapes.solid(bs)
                self._resize_board_canvas()
        if bs in self.ui.board_sizes and self.ui.allow_shape:
            try:
                mask, stones = shapes.from_text_board(s.get("shape", ""),
                                                      board_size=bs)
            except (TypeError, ValueError, AttributeError):
                mask, stones = shapes.solid(bs), None
            self.hole_mask = mask
            self.start_stones = stones
            self.shape_name = "整盘" if (not mask.any() and stones is None) else "自定义"
            self.shape_btn.config(text=self.shape_name)
            self._resize_board_canvas()
        self._set_var(self.device_var, s.get("device", "cuda"), "str")
        if self.device_var.get() not in ("cuda", "cpu"):
            self.device_var.set("cuda")
        for key, var, kind in (
                ("simulations", self.simulations_var, "int"),
                ("mcts_batch", self.mcts_batch_var, "int"),
                ("c_puct", self.c_puct_var, "float"),
                ("temperature", self.temperature_var, "float"),
                ("alpha", self.alpha_var, "float"),
                ("vloss", self.vloss_var, "int"),
                ("pcr", self.pcr_var, "bool"),
                ("pcr_prob", self.pcr_prob_var, "float"),
                ("dir_alpha", self.dir_alpha_var, "float"),
                ("dir_eps", self.dir_eps_var, "float"),
                ("resign", self.resign_var, "float"),
                ("train_only", self.train_only_var, "bool"),
                ("games", self.total_var, "int"),
                ("batch", self.batch_size_var, "int"),
                ("steps", self.steps_var, "int"),
                ("save_batch", self.save_batch_var, "int"),
                ("freeze", self.freeze_var, "str"),
                ("block_drop", self.block_drop_var, "str"),
                ("drop_blocks", self.drop_blocks_var, "str"),
                ("optimizer", self.optimizer_var, "str"),
                ("lr", self.lr_var, "float"),
                ("momentum", self.momentum_var, "float"),
                ("wd", self.wd_var, "float"),
                ("load_data", self.load_data_var, "bool"),
                ("save_data", self.save_data_var, "bool"),
                ("tensorrt", self.tensorrt_var, "bool"),
                ("load_mode", self.load_mode_var, "str")):
            if key in s:
                self._set_var(var, s[key], kind)
        if hasattr(self, "komi_var") and "komi" in s:
            self._set_var(self.komi_var, s["komi"], "float")
        if hasattr(self, "komi_range_var") and "komi_range" in s:
            self._set_var(self.komi_range_var, snap_half_step(s["komi_range"]), "float")
        if self.optimizer_var.get() not in ("Adam", "SGD"):
            self.optimizer_var.set("Adam")
        if self.load_mode_var.get() not in ("newest", "random"):
            self.load_mode_var.set("newest")
        dd = s.get("data_dir")
        if isinstance(dd, str) and dd.strip():
            self.data_dir_var.set(dd.strip())
        elif bs in self.ui.board_sizes and bs != self.ui.default_size:
            self.data_dir_var.set(self._default_data_dir(bs))

    def _create_model_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="模型设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="当前权重:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value=self.ui.model_path)
        ttk.Label(frame, textvariable=self.model_path_var, foreground="#1a5fb4").grid(
            row=0, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(frame, text="设备:").grid(row=2, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(frame, textvariable=self.device_var, values=["cuda", "cpu"],
                     state="readonly", width=8).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(frame, text="棋盘:").grid(row=3, column=0, sticky=tk.W)
        self.board_size_var = tk.IntVar(value=self.board_size)
        self.board_combo = ttk.Combobox(frame, textvariable=self.board_size_var,
                                        values=self.ui.board_sizes, state="readonly", width=6)
        self.board_combo.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.board_combo.bind("<<ComboboxSelected>>", self._on_board_size_change)
        if self.ui.allow_komi:
            ttk.Label(frame, text="贴目:").grid(row=4, column=0, sticky=tk.W)
            self.komi_var = tk.DoubleVar(value=KOMI)
            ttk.Spinbox(frame, from_=-25.0, to=25.0, increment=0.5,
                        textvariable=self.komi_var, width=8).grid(
                            row=4, column=1, sticky=tk.W, padx=5)
            ttk.Label(frame, text="贴目浮动范围:").grid(row=5, column=0, sticky=tk.W)
            self.komi_range_var = tk.DoubleVar(value=0.0)
            ttk.Spinbox(frame, from_=0.0, to=25.0, increment=0.5,
                        textvariable=self.komi_range_var, width=8).grid(
                            row=5, column=1, sticky=tk.W, padx=5)
        if self.ui.allow_shape:
            ttk.Label(frame, text="形状:").grid(row=6, column=0, sticky=tk.W)
            self.shape_name = "整盘"
            self.shape_btn = ttk.Button(frame, text="编辑形状",
                                        command=self._on_shape_change, width=12)
            self.shape_btn.grid(row=6, column=1, sticky=tk.W, padx=5)
        self.tensorrt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="TensorRT REFIT（实验，需预建引擎）",
            variable=self.tensorrt_var).grid(row=7, column=0, columnspan=3,
                                             sticky=tk.W, pady=(4, 0))

    def _create_mcts_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="MCTS", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="模拟次数:").grid(row=0, column=0, sticky=tk.W)
        self.simulations_var = tk.IntVar(value=NUM_SIMULATIONS)
        ttk.Spinbox(frame, from_=20, to=GUI_MAX_SIMS, textvariable=self.simulations_var,
                    width=10).grid(row=0, column=1, padx=5)
        # 推理 batch / 虚拟损失 不再暴露到界面（固定取超参），但变量保留：
        # 设置文件往返与 MCTS 构造都还在读它们。
        self.mcts_batch_var = tk.IntVar(value=BATCH_SIZE_MCTS)
        ttk.Label(frame, text="c_puct:").grid(row=2, column=0, sticky=tk.W)
        self.c_puct_var = tk.DoubleVar(value=C_PUCT)
        ttk.Spinbox(frame, from_=0.0, to=10.0, increment=0.05,
                    textvariable=self.c_puct_var, width=10).grid(row=2, column=1, padx=5)
        ttk.Label(frame, text="温度:").grid(row=3, column=0, sticky=tk.W)
        self.temperature_var = tk.DoubleVar(value=TEMPERATURE)
        ttk.Spinbox(frame, from_=0.1, to=2.0, increment=0.05,
                    textvariable=self.temperature_var, width=10).grid(row=3, column=1, padx=5)
        ttk.Label(frame, text="α 混合:").grid(row=4, column=0, sticky=tk.W)
        self.alpha_var = tk.DoubleVar(value=ALPHA)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.alpha_var, width=10).grid(row=4, column=1, padx=5)
        self.vloss_var = tk.IntVar(value=VIRTUAL_LOSS)
        self.pcr_var = tk.BooleanVar(value=PCR_ENABLED)
        ttk.Checkbutton(frame, text="快速走子（多数手少搜，不进训练数据）",
                        variable=self.pcr_var).grid(
            row=6, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(frame, text="全搜索比例:").grid(row=7, column=0, sticky=tk.W)
        self.pcr_prob_var = tk.DoubleVar(value=PCR_PROB)
        ttk.Spinbox(frame, from_=0.05, to=1.0, increment=0.05,
                    textvariable=self.pcr_prob_var, width=10).grid(row=7, column=1, padx=5)
        ttk.Label(frame, text="噪声 α:").grid(row=8, column=0, sticky=tk.W)
        self.dir_alpha_var = tk.DoubleVar(value=DIRICHLET_ALPHA)
        ttk.Spinbox(frame, from_=0.001, to=1.0, increment=0.005,
                    textvariable=self.dir_alpha_var, width=10).grid(row=8, column=1, padx=5)
        ttk.Label(frame, text="噪声 ε:").grid(row=9, column=0, sticky=tk.W)
        self.dir_eps_var = tk.DoubleVar(value=DIRICHLET_EPSILON)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.dir_eps_var, width=10).grid(row=9, column=1, padx=5)
        ttk.Label(frame, text="认输阈值:").grid(row=10, column=0, sticky=tk.W)
        self.resign_var = tk.DoubleVar(value=RESIGN_THRESHOLD)
        ttk.Spinbox(frame, from_=0.0, to=0.5, increment=0.01,
                    textvariable=self.resign_var, width=10).grid(row=10, column=1, padx=5)

    def _create_training_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="训练", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.train_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="仅训练模式（不生成新对局）",
                        variable=self.train_only_var).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(frame, text="对局数:").grid(row=1, column=0, sticky=tk.W)
        self.total_var = tk.IntVar(value=TRAIN_GAMES)
        ttk.Spinbox(frame, from_=10, to=100000, textvariable=self.total_var,
                    width=12).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="训练 batch:").grid(row=2, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=BATCH_SIZE)
        ttk.Spinbox(frame, from_=8, to=256, textvariable=self.batch_size_var,
                    width=12).grid(row=2, column=1, padx=5)
        ttk.Label(frame, text="每局训练步数:").grid(row=3, column=0, sticky=tk.W)
        self.steps_var = tk.IntVar(value=TRAIN_STEPS_PER_GAME)
        ttk.Spinbox(frame, from_=1, to=128, textvariable=self.steps_var,
                    width=12).grid(row=3, column=1, padx=5)
        ttk.Label(frame, text="保存间隔(batch):").grid(row=4, column=0, sticky=tk.W)
        self.save_batch_var = tk.IntVar(value=SAVE_INTERVAL_BATCHES)
        ttk.Spinbox(frame, from_=1, to=10000, textvariable=self.save_batch_var,
                    width=12).grid(row=4, column=1, padx=5)
        ttk.Label(frame, text="冻结参数(可空):").grid(row=5, column=0, sticky=tk.W)
        self.freeze_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.freeze_var, width=24).grid(row=5, column=1, padx=5)
        ttk.Label(frame, text="块dropout(p):").grid(row=6, column=0, sticky=tk.W)
        self.block_drop_var = tk.StringVar(value="0")
        ttk.Entry(frame, textvariable=self.block_drop_var, width=24).grid(row=6, column=1, padx=5)
        ttk.Label(frame, text="drop块(空=全部):").grid(row=7, column=0, sticky=tk.W)
        self.drop_blocks_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.drop_blocks_var, width=24).grid(row=7, column=1, padx=5)

    def _create_optimizer_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="优化器", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="优化器:").grid(row=0, column=0, sticky=tk.W)
        self.optimizer_var = tk.StringVar(value=OPTIMIZER_NAME)
        ttk.Combobox(frame, textvariable=self.optimizer_var, values=["Adam", "SGD"],
                     state="readonly", width=8).grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="学习率:").grid(row=1, column=0, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=LEARNING_RATE)
        ttk.Spinbox(frame, from_=0.00001, to=0.02, increment=0.0005,
                    textvariable=self.lr_var, width=12).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="动量(SGD):").grid(row=2, column=0, sticky=tk.W)
        self.momentum_var = tk.DoubleVar(value=MOMENTUM)
        ttk.Spinbox(frame, from_=0.0, to=0.999, increment=0.05,
                    textvariable=self.momentum_var, width=12).grid(row=2, column=1, padx=5)
        ttk.Label(frame, text="权重衰减:").grid(row=3, column=0, sticky=tk.W)
        self.wd_var = tk.DoubleVar(value=WEIGHT_DECAY)
        ttk.Spinbox(frame, from_=0.0, to=0.001, increment=0.00005,
                    textvariable=self.wd_var, width=12).grid(row=3, column=1, padx=5)

    def _create_data_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="数据", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.load_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="启动时加载已有数据", variable=self.load_data_var).pack(anchor=tk.W)
        self.save_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="保存新对局（.npz）", variable=self.save_data_var).pack(anchor=tk.W)
        mode_row = ttk.Frame(frame)
        mode_row.pack(anchor=tk.W, pady=(2, 2))
        ttk.Label(mode_row, text="加载方式:").pack(side=tk.LEFT)
        self.load_mode_var = tk.StringVar(value="newest")
        ttk.Radiobutton(mode_row, text="最新", value="newest",
                        variable=self.load_mode_var).pack(side=tk.LEFT, padx=4)
        ttk.Radiobutton(mode_row, text="随机", value="random",
                        variable=self.load_mode_var).pack(side=tk.LEFT, padx=4)
        ttk.Label(frame, text="数据目录:").pack(anchor=tk.W)
        self.data_dir_var = tk.StringVar(value=DATA_DIR)
        ttk.Entry(frame, textvariable=self.data_dir_var, width=28).pack(fill=tk.X, pady=2)

    def _create_buttons(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=8)
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        self.start_btn = ttk.Button(row1, text="开始训练", command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=4)
        self.stop_btn = ttk.Button(row1, text="停止", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=4)
        ttk.Button(row1, text="应用超参数", command=self._apply_hyperparams).pack(side=tk.LEFT, padx=4)
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        ttk.Button(row2, text="手动保存", command=self._save_model).pack(side=tk.LEFT, padx=4)
        ttk.Button(row2, text="加载数据文件", command=self._load_data).pack(side=tk.LEFT, padx=4)

    def _default_data_dir(self, board_size):
        return self.ui.data_dir(int(board_size))

    @staticmethod
    def _board_metrics(board_size):
        cell = 56 if board_size < 19 else 44
        margin = 44
        width = (board_size - 1) * cell + 2 * margin
        return cell, margin, width

    def _create_board_area(self, parent):
        frame = ttk.LabelFrame(parent, text="对局回放", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.cell_size, self.margin, self.canvas_size = self._board_metrics(self.board_size)
        self.board_width = (self.board_size - 1) * self.cell_size
        self.canvas = tk.Canvas(frame, width=self.canvas_size, height=self.canvas_size,
                                bg="#D1D1D1", highlightthickness=0)
        self.canvas.pack(pady=8)
        self.game_board = GameBoard(self.canvas, self.board_size, self.cell_size,
                                    self.margin, hole_mask=self.hole_mask)
        control = ttk.Frame(frame)
        control.pack(pady=4)
        ttk.Button(control, text="◀ 上一局", command=self._prev_game, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(control, text="下一局 ▶", command=self._next_game, width=8).pack(side=tk.LEFT, padx=4)
        ttk.Button(control, text="⏮", command=self._first_step, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(control, text="◀", command=self._prev_step, width=5).pack(side=tk.LEFT, padx=4)
        ttk.Button(control, text="▶", command=self._next_step, width=5).pack(side=tk.LEFT, padx=4)
        self.step_label = ttk.Label(control, text="步数: 0/0")
        self.step_label.pack(side=tk.LEFT, padx=10)
        self.game_info_label = ttk.Label(frame, text="")
        self.game_info_label.pack(pady=4)

    def _on_board_size_change(self, event=None):
        if self.is_training:
            self.board_size_var.set(self.board_size)
            messagebox.showwarning("提示", "训练中不能切换棋盘，请先停止训练")
            return
        n = int(self.board_size_var.get())
        if n == self.board_size:
            return
        old_defaults = {self._default_data_dir(b) for b in self.ui.board_sizes}
        if self.data_dir_var.get() in old_defaults:
            self.data_dir_var.set(self._default_data_dir(n))
        self.board_size = n
        self.hole_mask = shapes.solid(n)          # 换尺寸后旧形状对不上，退回整盘
        self.start_stones = None
        self.shape_name = "整盘"
        if self.ui.allow_shape:
            self.shape_btn.config(text=self.shape_name)
        self._resize_board_canvas()
        self._reset_replay_state()
        self._save_settings()

    def _on_shape_change(self, event=None):
        if not self.ui.allow_shape:
            return
        if self.is_training:
            messagebox.showwarning("提示", "训练中不能切换形状，请先停止训练")
            return
        try:
            import shape_editor
            mask, stones, _name = shape_editor.edit_shape(
                self.root, self.board_size, self.hole_mask, self.start_stones)
        except Exception as exc:
            messagebox.showerror("形状编辑器", "%s: %s" % (type(exc).__name__, exc))
            return
        if mask is None:
            return
        self.hole_mask = shapes.validate_mask(mask, self.board_size)
        self.start_stones = shapes.validate_stones(stones, self.hole_mask,
                                                   self.board_size)
        self.shape_name = "整盘" if (not self.hole_mask.any()
                                  and self.start_stones is None) else "自定义"
        self.shape_btn.config(text=self.shape_name)
        self._resize_board_canvas()
        self._reset_replay_state()
        self._save_settings()

    def _reset_replay_state(self):
        """换棋盘/换形状后清掉回放状态（原来这段被复制了两份）。"""
        self.game_history.clear()
        self.current_history_index = -1
        self.current_game_moves = []
        self.current_game_players = []
        self.current_step = 0
        self.step_label.config(text="步数: 0/0")
        self.game_info_label.config(text="")

    def _resize_board_canvas(self):
        self.cell_size, self.margin, self.canvas_size = self._board_metrics(self.board_size)
        self.board_width = (self.board_size - 1) * self.cell_size
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)
        self.game_board = GameBoard(self.canvas, self.board_size, self.cell_size,
                                    self.margin, hole_mask=self.hole_mask)
        self._draw_empty_board(self.start_stones)

    def _create_status_area(self, parent):
        frame = ttk.LabelFrame(parent, text="训练状态", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=(8, 0))
        self.status_text = tk.Text(frame, height=12, font=("Consolas", 9), wrap=tk.WORD)
        sb = ttk.Scrollbar(frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=sb.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_panel = TrainingStatsPanel(frame)
        self.stats_panel.create()

    # ---------------- 棋盘 ----------------
    def _draw_empty_board(self, stones=None):
        """画格线/星位，可选再叠一层开局棋子。

        注意：stones 的默认值是 None（**不画子**），而不是 self.start_stones。
        回放/样本渲染会在这层之上按真实盘面逐个画子；如果这里先画了开局子，
        被提掉的开局子就没有机会被擦掉（220 行外的调用则显式传 self.start_stones）。
        """
        self.canvas.delete("all")
        self.game_board.draw_grid()
        self.game_board.draw_stars()
        if stones is not None:
            for r in range(self.board_size):
                for c in range(self.board_size):
                    if stones[r, c] != 0:
                        self.game_board.draw_piece(r, c, int(stones[r, c]))

    def _replay_game(self, moves, step, hole_mask=None, players=None,
                     start_stones=None):
        """按当前棋种的规则重放前 step 手。

        - 必须用 rules.new_game：早先这里写死 GoGame，五子棋回放会把「没气」的棋子
          按围棋规则提掉，界面上就出现「五子棋吃子」。
        - 旧数据里可能有让子开局（开局前 k 手全是黑子），必须一次性摆上
          （setup_handicap），不能走 make_move 的黑白轮转 —— 否则 4 子让子会
          显示成 2 黑 2 白。新数据不再产生让子开局，这里只服务历史文件。
        - 自定义开局（start_stones）是局面的一部分，重放前先摆上；它不算手数，
          所以回放的 moves 仍然从黑棋第一手开始。
        """
        mask = self.hole_mask if hole_mask is None else hole_mask
        komi = float(self.komi_var.get()) if hasattr(self, "komi_var") else KOMI
        stones = start_stones
        game = self.rules.new_game(self.board_size, mask, komi,
                                   start_board=stones)
        limit = min(step, len(moves))
        start = 0
        if (start_stones is None and players is not None
                and hasattr(game, "setup_handicap")):
            while start < limit and start < len(players) and int(players[start]) == 1:
                start += 1
            # 普通对局的首手也是黑，k=1 时与逐手重放等价，不特判
            if start:
                game.setup_handicap([(int(moves[i][0]), int(moves[i][1]))
                                     for i in range(start)])
        for i in range(start, limit):
            r, c = moves[i]
            if (r, c) == PASS_MOVE:
                game.make_move(-1, -1)
            else:
                game.make_move(r, c)
        return game

    def _render_replay(self, moves, step, hole_mask=None, players=None,
                       start_stones=None):
        game = self._replay_game(moves, step, hole_mask, players,
                                 start_stones=start_stones)
        self._draw_empty_board()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if game.is_hole(r, c):
                    continue
                if game.board[r, c] != 0:
                    self.game_board.draw_piece(r, c, int(game.board[r, c]))
        if game.last_move and game.last_move != PASS_MOVE:
            self.game_board.highlight_move(*game.last_move)
        self.step_label.config(text=f"步数: {step}/{len(moves)}")

    def _display_game(self, game_result):
        self.current_game_moves = game_result.get_moves()
        self.current_game_players = list(getattr(game_result, "players", []) or [])
        self.current_game_stones = getattr(game_result, "start_stones", None)
        self.current_step = len(self.current_game_moves)
        if not np.array_equal(game_result.hole_mask, self.game_board.hole_mask):
            self.game_board.set_hole(game_result.hole_mask)
            self._draw_empty_board()
        self._render_replay(self.current_game_moves, self.current_step,
                            hole_mask=game_result.hole_mask,
                            players=self.current_game_players,
                            start_stones=game_result.start_stones)
        w = game_result.winner
        shape = "" if not game_result.hole_mask.any() else " 自定义形状"
        self.game_info_label.config(text=f"第 {game_result.game_id} 局 ({game_result.board_size}路{shape}) | "
                                         + result_text(w, game_result.score_diff,
                                                       getattr(game_result, "resigned", False)))

    def _prev_game(self):
        if not self.game_history:
            return
        self.current_history_index = max(0, self.current_history_index - 1)
        self._display_game(self.game_history[self.current_history_index])

    def _next_game(self):
        if not self.game_history:
            return
        self.current_history_index = min(len(self.game_history) - 1, self.current_history_index + 1)
        self._display_game(self.game_history[self.current_history_index])

    def _first_step(self):
        if self.current_game_moves:
            self.current_step = 1
            self._render_replay(self.current_game_moves, self.current_step,
                            players=getattr(self, "current_game_players", None),
                            start_stones=getattr(self, "current_game_stones", None))

    def _prev_step(self):
        if self.current_game_moves and self.current_step > 0:
            self.current_step -= 1
            self._render_replay(self.current_game_moves, self.current_step,
                            players=getattr(self, "current_game_players", None),
                            start_stones=getattr(self, "current_game_stones", None))

    def _next_step(self):
        if self.current_game_moves and self.current_step < len(self.current_game_moves):
            self.current_step += 1
            self._render_replay(self.current_game_moves, self.current_step,
                            players=getattr(self, "current_game_players", None),
                            start_stones=getattr(self, "current_game_stones", None))

    # ---------------- 按钮 ----------------
    def _apply_hyperparams(self):
        self._save_settings()
        if self.trainer is None:
            self._log_message("[超参数] 训练器尚未启动")
            return
        if self.is_training:
            # 训练线程可能正在 backward；这里只改标量/param_group，不重建优化器。
            self.trainer.learning_rate = self.lr_var.get()
            self.trainer.weight_decay = self.wd_var.get()
            for g in self.trainer.optimizer.param_groups:
                g["lr"] = self.trainer.learning_rate
                if g.get("wd_enabled", True):
                    g["weight_decay"] = self.trainer.weight_decay
            self.trainer.c_puct = self.c_puct_var.get()
            self.trainer.num_simulations = self.simulations_var.get()
            cfg = getattr(self, "_active_cfg", None)
            if cfg is not None:
                cfg["dir_alpha"] = self.dir_alpha_var.get()
                cfg["dir_eps"] = self.dir_eps_var.get()
                cfg["resign"] = float(self.resign_var.get())
            self._log_message("[超参数] 训练中已更新 lr/wd；模拟数/c_puct/噪声/认输阈值从下一局生效；"
                              "优化器类型/动量需停止后应用")
            return
        self.trainer.update_hyperparameters(
            learning_rate=self.lr_var.get(),
            weight_decay=self.wd_var.get(),
            optimizer_name=self.optimizer_var.get(),
            momentum=self.momentum_var.get(),
            c_puct=self.c_puct_var.get(),
            num_simulations=self.simulations_var.get())
        self._log_message("[超参数] 已更新")

    def _start_training(self):
        if self.is_training:
            return
        self._save_settings()
        model_path = self.model_path_var.get().strip()
        if not os.path.exists(model_path):
            messagebox.showerror("错误", "起始权重不存在：%s" % model_path)
            return
        # 先把 cfg 构造完再动状态：Tk 的 IntVar/DoubleVar.get() 在输入框里是
        # 非法文本时会抛 TclError。以前这段在 is_training/按钮状态之后，异常
        # 一逃出按钮回调，worker 就没启动、"finished" 永不到达，开始按钮
        # 永久变灰，只能重启程序。
        try:
            cfg = {
                "model_path": model_path,
                "board_size": self.board_size,
                "hole_mask": self.hole_mask,
                "start_stones": self.start_stones,
                "device": self.device_var.get(),
                "sims": self.simulations_var.get(),
                "mcts_batch": self.mcts_batch_var.get(),
                "c_puct": self.c_puct_var.get(),
                "temperature": self.temperature_var.get(),
                "alpha": self.alpha_var.get(),
                "vloss": self.vloss_var.get(),
                "pcr": bool(self.pcr_var.get()),
                "pcr_prob": float(self.pcr_prob_var.get()),
                "dir_alpha": self.dir_alpha_var.get(),
                "dir_eps": self.dir_eps_var.get(),
                "resign": float(self.resign_var.get()),
                "komi": float(self.komi_var.get()) if hasattr(self, "komi_var") else KOMI,
                # 每局从这个半宽里按 0.5 格点均匀抽贴目；0 = 固定贴目（旧行为）
                "komi_range": (snap_half_step(self.komi_range_var.get())
                               if hasattr(self, "komi_range_var") else 0.0),
                "train_only": self.train_only_var.get(),
                "games": self.total_var.get(),
                "batch": self.batch_size_var.get(),
                "steps": self.steps_var.get(),
                "save_interval": int(SAVE_INTERVAL),
                "ui_interval": int(UI_UPDATE_BATCHES),
                "save_batch": self.save_batch_var.get(),
                "freeze": self.freeze_var.get().strip() or None,
                "block_drop": float(self.block_drop_var.get() or 0),
                "drop_blocks": self.drop_blocks_var.get().strip(),
                "lr": self.lr_var.get(),
                "wd": self.wd_var.get(),
                "optimizer": self.optimizer_var.get(),
                "momentum": self.momentum_var.get(),
                "load_data": self.load_data_var.get(),
                "save_data": self.save_data_var.get(),
                "tensorrt": bool(self.tensorrt_var.get()),
                "load_mode": self.load_mode_var.get(),
                "data_dir": self.data_dir_var.get().strip() or DATA_DIR,
            }
            # 把对齐后的浮动范围回写到控件：用户填 1.3 时界面显示 1.5，与实际一致
            if hasattr(self, "komi_range_var"):
                self.komi_range_var.set(cfg["komi_range"])
        except Exception as exc:
            messagebox.showerror("参数错误", "参数填写有误，训练未开始：\n%s: %s"
                                 % (type(exc).__name__, exc))
            return
        self.is_training = True
        self.stop_event.clear()
        self.trainer = None
        self.game_history.clear()
        self.current_history_index = -1
        self.current_game_moves = []
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.status_text.delete(1.0, tk.END)
        if self._log_file is None:
            self._log_file = open(os.path.join(LOG_DIR, "train_gui_%s.log" % datetime.now().strftime("%Y%m%d_%H%M%S")),
                                  "a", encoding="utf-8", buffering=1)
        self._log_message("=" * 60)
        self._log_message("训练开始: %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._active_cfg = cfg
        self.worker = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self.worker.start()

    def _run(self, cfg):
        try:
            from train import SelfPlayTrainer
            t0 = time.time()
            trainer = SelfPlayTrainer(
                model_path=cfg["model_path"], board_size=cfg["board_size"],
                device=cfg["device"], num_simulations=cfg["sims"],
                data_dir=cfg["data_dir"], c_puct=cfg["c_puct"],
                learning_rate=cfg["lr"], weight_decay=cfg["wd"],
                optimizer_name=cfg["optimizer"], momentum=cfg["momentum"],
                freeze_parts=cfg["freeze"],
                use_tensorrt=cfg["tensorrt"], rules=self.rules,
                trt_engine_dir=getattr(self.ui, "trt_engine_dir", None),
                log=lambda msg: self.update_queue.put(
                    {"type": "log", "message": msg}))
            self.trainer = trainer
            try:
                _p = float(cfg.get("block_drop") or 0.0)
                _bl = [int(x) for x in str(cfg.get("drop_blocks") or "")
                       .replace("，", ",").split(",") if x.strip()]
                if _p > 0:
                    trainer.model.set_block_drop(_p, _bl)
                    self.update_queue.put({"type": "log", "message":
                                           "[训练] 块 dropout: p=%.2f 作用块=%s"
                                           % (_p, _bl or "全部")})
            except Exception as exc:
                self.update_queue.put({"type": "log", "message":
                                       "[警告] 块 dropout 参数无效: %s" % exc})
            self.update_queue.put({"type": "log", "message":
                                   "[棋盘] %d 路 | 形状 %s（可下 %d 点）| 贴目 %s | 数据目录: %s | 温度 %.2f→0@%d手 | 噪声 α=%.3f ε=%.2f"
                                   % (cfg["board_size"],
                                      "整盘" if not cfg["hole_mask"].any() else "自定义",
                                      shapes.playable_area(cfg["hole_mask"]),
                                      ("%.1f±%.1f（每局按 0.5 均匀抽）" % (cfg["komi"], cfg["komi_range"])
                                       if cfg.get("komi_range", 0.0) > 0 else "%.1f" % cfg["komi"]),
                                      cfg["data_dir"],
                                      cfg["temperature"],
                                      board_temperature_zero_for_area(
                                          shapes.playable_area(cfg["hole_mask"])),
                                      cfg["dir_alpha"], cfg["dir_eps"])})
            self.update_queue.put({"type": "log", "message":
                                   "[模型] 已加载: %s（训练后原地更新）" % trainer.model_path})
            self.update_queue.put({"type": "log", "message":
                                   "[模型] 参数量 %.3f M" % (sum(p.numel() for p in trainer.model.parameters()) / 1e6)})
            if cfg["load_data"]:
                trainer.load_training_data(mode=cfg["load_mode"])
            db = trainer.data_buffer
            if getattr(db, "last_skipped", 0):
                self.update_queue.put({"type": "log", "message":
                                       "[数据] 跳过 %d 个旧版/损坏文件（支持 v3/v4/v5）"
                                       % db.last_skipped})
            self.update_queue.put({"type": "log", "message":
                                   "[数据] 缓冲 %d 条（%.1fs）" % (len(db), time.time() - t0)})
            if cfg["train_only"]:
                self._train_only_loop(trainer, cfg)
            else:
                self._selfplay_loop(trainer, cfg)
            trainer.flush_grad_accum()
            p = trainer.save_model("final")
            self.update_queue.put({"type": "log", "message": "[模型] 已更新唯一权重: %s" % p})
        except Exception as e:
            self.update_queue.put({"type": "error", "message": "%s: %s" % (type(e).__name__, e)})
            self.update_queue.put({"type": "log", "message": traceback.format_exc()})
            if self.trainer is not None:
                try:
                    self.trainer.flush_grad_accum()
                    p = self.trainer.save_model("error")
                    self.update_queue.put({"type": "log", "message": "[模型] 异常收尾已更新唯一权重: %s" % p})
                except Exception:
                    pass
        finally:
            self.is_training = False
            self.update_queue.put({"type": "finished"})

    def _train_only_loop(self, trainer, cfg):
        if len(trainer.data_buffer) < cfg["batch"]:
            self.update_queue.put({"type": "error", "message":
                                   "数据不足：%d < %d" % (len(trainer.data_buffer), cfg["batch"])})
            return
        batch = 0
        acc = [0.0] * trainer.LOSS_N
        acc_n = 0
        while self.is_training and not self.stop_event.is_set():
            loss = trainer.train_step(cfg["batch"])
            if loss[0] is None:
                time.sleep(0.1)
                continue
            trainer.train_count += 1
            batch += 1
            for i, v in enumerate(loss):
                acc[i] += v
            acc_n += 1
            if acc_n >= cfg["ui_interval"]:
                mean = tuple(a / acc_n for a in acc)
                acc = [0.0] * len(acc)
                acc_n = 0
                self.update_queue.put({"type": "train_progress", "loss": mean, "batch": batch})
            if batch % cfg["save_batch"] == 0:
                # 先把半截梯度累积兑现掉再存，否则存下来的权重比「当前训练状态」
                # 落后最多半个累积（1 次 optimizer.step）。
                trainer.flush_grad_accum()
                p = trainer.save_model("b%06d" % batch)
                self.update_queue.put({"type": "log", "message": "[模型] 已保存: %s" % p})
            if batch % 10 == 0:
                self.update_queue.put({"type": "stats"})
        if acc_n:
            mean = tuple(a / acc_n for a in acc)
            self.update_queue.put({"type": "train_progress", "loss": mean, "batch": batch})

    def _selfplay_loop(self, trainer, cfg):
        from mcts import MCTS

        def make_mcts():
            return MCTS(evaluator=trainer.evaluator,
                        num_simulations=trainer.num_simulations,
                        batch_size=cfg["mcts_batch"], c_puct=trainer.c_puct,
                        temperature=cfg["temperature"], alpha=cfg["alpha"],
                        dirichlet_alpha=cfg["dir_alpha"],
                        dirichlet_epsilon=cfg["dir_eps"],
                        board_size=cfg["board_size"], virtual_loss=cfg["vloss"],
                        rules=self.rules)

        mcts = make_mcts()
        pcr_prob = float(cfg.get("pcr_prob") or 0.0) if cfg.get("pcr") else 0.0
        if pcr_prob > 0.0:
            self.update_queue.put({"type": "log", "message":
                                   "[自对弈] 快速走子开启：全搜索比例 %.0f%%，"
                                   "快搜索访问数 = 模拟数/%d，只有全搜索局面进训练数据"
                                   % (pcr_prob * 100, max(1, int(PCR_FAST_DIV)))})
        games = 0
        while games < cfg["games"] and self.is_training and not self.stop_event.is_set():
            # 训练中改模拟数/c_puct 时，在局间重建搜索器；树在局间本来就会 reset。
            if (mcts.num_simulations != trainer.num_simulations
                    or abs(mcts.c_puct - trainer.c_puct) > 1e-9
                    or abs(mcts.dirichlet_alpha - cfg["dir_alpha"]) > 1e-12
                    or abs(mcts.dirichlet_epsilon - cfg["dir_eps"]) > 1e-12):
                mcts = make_mcts()
                self.update_queue.put({"type": "log", "message":
                                       "[MCTS] 已应用 模拟数=%d c_puct=%.3f 噪声α=%.3f ε=%.2f"
                                       % (trainer.num_simulations, trainer.c_puct,
                                          cfg["dir_alpha"], cfg["dir_eps"])})
            t0 = time.time()
            # 每局重新抽一次贴目：komi_range=0 时原样返回 cfg["komi"]（固定贴目）
            komi = sample_komi(cfg["komi"], cfg.get("komi_range", 0.0))
            result = play_one_game(
                mcts=mcts, board_size=cfg["board_size"],
                num_simulations=trainer.num_simulations,
                device=cfg["device"], c_puct=trainer.c_puct,
                temperature=cfg["temperature"],
                hole_mask=cfg["hole_mask"], rules=self.rules,
                komi=komi, start_stones=cfg["start_stones"],
                pcr_prob=pcr_prob, resign_threshold=cfg.get("resign", 0.0))
            if result["payload"] is None or not result["complete"]:
                self.update_queue.put({"type": "log", "message":
                                       "[自对弈] 对局未正常结束，丢弃本局数据"})
                continue
            saved_path = None
            if cfg["save_data"]:
                saved_path = trainer.save_training_data(result["payload"])
            trainer.data_buffer.add_game(result, source=saved_path)
            trainer.game_count += 1
            games += 1
            losses = []
            for _ in range(cfg["steps"]):
                loss = trainer.train_step(cfg["batch"])
                if loss[0] is None:
                    break
                trainer.train_count += 1
                losses.append(loss)
            if losses:
                loss = tuple(float(np.mean([x[i] for x in losses]))
                             for i in range(trainer.LOSS_N))
            else:
                loss = (None,) * trainer.LOSS_N
            gr = GameResult(games, result["moves"], result["winner"], cfg["board_size"],
                            hole_mask=cfg["hole_mask"],
                            score_diff=result.get("score_diff", 0.0),
                            start_stones=cfg["start_stones"],
                            resigned=result.get("resigned", False))
            self.update_queue.put({"type": "game", "game_result": gr,
                                   "game_count": games, "loss": loss,
                                   "seconds": time.time() - t0,
                                   "buffer": len(trainer.data_buffer),
                                   **({"komi": komi}
                                      if cfg.get("komi_range", 0.0) > 0 else {})})
            if games % cfg["save_interval"] == 0:
                p = trainer.save_model("g%06d" % games)
                self.update_queue.put({"type": "log", "message": "[模型] 已保存: %s" % p})

    def _stop_training(self):
        self.is_training = False
        self.stop_event.set()
        self._log_message("[用户] 正在停止（当前局结束后收尾）...")

    def _save_model(self):
        if self.trainer is None:
            self._log_message("[模型] 训练器尚未启动")
            return
        p = self.trainer.save_model("manual")
        self._log_message("[模型] 已保存: %s" % p)

    def _load_data(self):
        if self.trainer is None:
            self._log_message("[数据] 请先开始一次训练，再加载数据")
            return
        f = filedialog.askopenfilename(initialdir=self.data_dir_var.get(),
                                       filetypes=[("Go npz", "*.npz")])
        if f:
            self.trainer.load_training_data(f)
            self._log_message("[数据] 已加载: %s" % f)

    def _log_message(self, msg):
        line = "[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg)
        self.status_text.insert(tk.END, line + "\n")
        self.status_text.see(tk.END)
        if self._log_file is not None:
            try:
                self._log_file.write(line + "\n")
            except Exception:
                pass

    def _update_stats(self):
        if self.trainer is not None:
            self.stats_panel.update_stats(len(self.trainer.data_buffer),
                                          self.trainer.game_count,
                                          self.trainer.train_count,
                                          self.trainer.optimizer.param_groups[0]["lr"])

    def _process_queue(self):
        try:
            while not self.update_queue.empty():
                msg = self.update_queue.get_nowait()
                typ = msg.get("type")
                if typ == "game":
                    gr = msg["game_result"]
                    self.game_history.append(gr)
                    if self.current_history_index == -1:
                        self.current_history_index = 0
                        self._display_game(gr)
                    if msg["loss"][0] is not None:
                        self.stats_panel.update_loss(*msg["loss"])
                    w = gr.winner
                    res = result_text(w, gr.score_diff, gr.resigned)
                    if msg["loss"][0] is not None:
                        self._log_message(
                            "[游戏 %d] %s | 手数 %d | %.1fs | 缓冲 %d%s | 平均 p=%.4f own=%.4f win=%.4f total=%.4f"
                            % (msg["game_count"], res, len(gr.moves),
                               msg["seconds"], msg["buffer"],
                               " | 贴目 %.1f" % msg["komi"] if "komi" in msg else "",
                               msg["loss"][0], msg["loss"][1],
                               msg["loss"][3], msg["loss"][4]))
                    else:
                        self._log_message("[游戏 %d] %s | 手数 %d | %.1fs | 缓冲 %d"
                                          % (msg["game_count"], res, len(gr.moves),
                                             msg["seconds"], msg["buffer"]))
                elif typ == "train_progress":
                    self.stats_panel.update_loss(*msg["loss"])
                    self._log_message("[训练] batch %d | 平均 p=%.4f own=%.4f win=%.4f total=%.4f"
                                      % (msg["batch"], msg["loss"][0], msg["loss"][1],
                                         msg["loss"][3], msg["loss"][4]))
                elif typ == "log":
                    self._log_message(msg["message"])
                elif typ == "stats":
                    self._update_stats()
                elif typ == "error":
                    self._log_message("[错误] " + msg["message"])
                elif typ == "finished":
                    self._log_message("[完成] 训练结束")
                    self.is_training = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
            self._update_stats()
        except Exception as e:
            self._log_message("[GUI 错误] %s" % e)
        finally:
            self.root.after(100, self._process_queue)

    def _on_close(self):
        self._save_settings()
        self.is_training = False
        self.stop_event.set()
        if self.worker is not None and self.worker.is_alive():
            self.worker.join(timeout=5.0)
        try:
            if self._log_file is not None:
                self._log_file.close()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    TrainingGUI()
