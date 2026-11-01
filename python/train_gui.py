"""本机围棋训练界面：单进程自对弈 + 训练 + 回放。

不再包含 --headless / 多 worker / ONNX 评估子进程——这些已归档到 legacy/server。
自对弈在后台线程运行，Tk 主线程只处理界面与队列；模型与搜索共用同一份权重。
"""
import os
import sys
import json
import queue
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

from game import GoGame, PASS_MOVE
from hyperparams import (BOARD_SIZE, BOARD_SIZES, MODEL_PATH, DATA_DIR, NUM_SIMULATIONS,
                         C_PUCT, TEMPERATURE, EXPLORATION_MODE, BATCH_SIZE,
                         SAVE_INTERVAL, TRAIN_GAMES, LEARNING_RATE,
                         WEIGHT_DECAY, MOMENTUM, OPTIMIZER_NAME,
                         UI_UPDATE_BATCHES, SAVE_INTERVAL_BATCHES, ALPHA,
                         TRAIN_STEPS_PER_GAME, BATCH_SIZE_MCTS, VIRTUAL_LOSS,
                         DIRICHLET_ALPHA, DIRICHLET_EPSILON, GUI_MAX_SIMS,
                         CANDIDATE_MASS, MIN_CANDIDATES, board_temperature_zero, LOG_DIR,
                         validate_hollow, hollow_choices)
from ui_utils import GameBoard, TrainingStatsPanel
from selfplay import play_one_game
from replay import (DATA_VERSION, SUPPORTED_DATA_VERSIONS, load_npz_arrays,
                    policy_segment, set_policy_onehot)

if sys.platform == "win32":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass

MAX_REPLAY_GAMES = 200
SETTINGS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "train_gui_settings.json")


class GameResult:
    __slots__ = ("game_id", "moves", "winner", "board_size", "hollow", "label_game")

    def __init__(self, game_id, moves, winner, board_size=BOARD_SIZE, hollow=0, label_game=None):
        self.game_id = game_id
        self.winner = winner
        self.board_size = int(board_size)
        self.hollow = int(hollow)
        self.label_game = label_game
        n = self.board_size + 1
        arr = array("h")
        for m in moves:
            r, c = int(m[0]), int(m[1])
            arr.append(0 if (r < 0 or c < 0) else (r + 1) * n + (c + 1))
        self.moves = arr

    def get_moves(self):
        n = self.board_size + 1
        return [((-1, -1) if code == 0 else (code // n - 1, code % n - 1))
                for code in self.moves]


class LabelGame:
    """从磁盘 .npz 读出的可标注对局。

    棋盘用 samples 里的 boards 重建，所以显示的是「该样本落子前」的局面；
    原来的目标着法不会以棋子形式出现在盘上。
    """
    __slots__ = ("path", "arrays", "board_size", "hollow", "count", "winner")

    def __init__(self, path, arrays=None):
        self.path = None if path is None else os.path.abspath(path)
        if arrays is None:
            if self.path is None:
                raise ValueError("没有数据文件")
            arrays = load_npz_arrays(self.path)
        self.arrays = arrays
        a = self.arrays
        for key in ("version", "board_size", "boards", "players", "lasts",
                    "legals", "moves", "pol_len", "pol_idx", "pol_p"):
            if key not in a:
                raise ValueError("缺少键 %s" % key)
        if int(a["version"]) not in SUPPORTED_DATA_VERSIONS:
            raise ValueError("数据版本 %d 不受支持（支持 %s）"
                             % (int(a["version"]), SUPPORTED_DATA_VERSIONS))
        self.board_size = int(a["board_size"])
        self.hollow = int(a["hollow"]) if "hollow" in a else 0
        self.hollow = validate_hollow(self.board_size, self.hollow)
        self.count = len(a["boards"])
        if self.count <= 0:
            raise ValueError("空对局")
        for key in ("players", "lasts", "legals", "moves", "pol_len"):
            if len(a[key]) != self.count:
                raise ValueError("%s 长度 %d != 样本数 %d" % (key, len(a[key]), self.count))
        if (int(np.asarray(a["pol_len"]).sum()) != len(a["pol_idx"])
                or len(a["pol_idx"]) != len(a["pol_p"])):
            raise ValueError("策略稀疏数组长度与 pol_len 不一致")
        self.winner = int(a["winner"]) if "winner" in a else 0

    def board(self, i):
        n = self.board_size
        rel = np.asarray(self.arrays["boards"][i], dtype=np.int16).reshape(n, n)
        return rel * int(self.arrays["players"][i])

    def legal(self, i):
        n = self.board_size
        bits = np.unpackbits(self.arrays["legals"][i], count=n * n)
        return bits.reshape(n, n).astype(bool)

    def last_move(self, i):
        v = int(self.arrays["lasts"][i])
        n = self.board_size
        if 0 <= v < n * n:
            return v // n, v % n
        return None

    def player(self, i):
        return int(self.arrays["players"][i])

    def target_move(self, i):
        """原实战目标着法；pass 返回 None。"""
        code = int(self.arrays["moves"][i])
        if code == 0:
            return None
        n1 = self.board_size + 1
        return code // n1 - 1, code % n1 - 1

    def move_index(self, r, c):
        return int(r) * self.board_size + int(c)

    def move_text(self, index):
        n = self.board_size
        i = int(index)
        if i == n * n:
            return "Pass"
        return "(%d,%d)" % (i // n, i % n)

    def top_policy(self, i):
        idx, pp = policy_segment(self.arrays, i)
        if len(idx) == 0:
            return None, 0.0
        j = int(np.argmax(pp))
        return int(idx[j]), float(pp[j])

    def set_onehot(self, i, r, c):
        if self.path is None:
            raise ValueError("当前对局未保存为 .npz，无法改硬盘标签")
        if self.hollow:
            lo = (self.board_size - self.hollow) // 2
            hi = lo + self.hollow
            if lo <= r < hi and lo <= c < hi:
                raise ValueError("洞内不能落子")
        board = self.board(i)
        if board[r, c] != 0:
            raise ValueError("该点已有棋子")
        if not self.legal(i)[r, c]:
            raise ValueError("该点不合法（劫/自杀）")
        move = self.move_index(r, c)
        self.arrays = set_policy_onehot(self.path, i, move)
        return move


class TrainingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("围棋 AI 训练（本机单进程）")
        self.root.geometry("2000x1300")
        self.board_size = BOARD_SIZE
        self.hollow = 0
        self.trainer = None
        self.is_training = False
        self.worker = None
        self.update_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.game_history = deque(maxlen=MAX_REPLAY_GAMES)
        self.current_history_index = -1
        self.current_game_moves = []
        self.current_step = 0
        self.view_mode = "live"
        self.label_game = None
        self.label_index = 0
        self.live_label_game = None
        self.live_label_opening = 0
        self._log_file = None
        self._settings = self._read_settings()
        self._create_widgets()
        self._apply_saved_settings()
        self._draw_empty_board()
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
            "hollow": int(self.hollow_var.get()),
            "device": self.device_var.get(),
            "simulations": int(self.simulations_var.get()),
            "mcts_batch": int(self.mcts_batch_var.get()),
            "c_puct": float(self.c_puct_var.get()),
            "temperature": float(self.temperature_var.get()),
            "alpha": float(self.alpha_var.get()),
            "vloss": int(self.vloss_var.get()),
            "candidate_mass": float(self.candidate_mass_var.get()),
            "min_candidates": int(self.min_candidates_var.get()),
            "dir_alpha": float(self.dir_alpha_var.get()),
            "dir_eps": float(self.dir_eps_var.get()),
            "exploration": bool(self.exploration_var.get()),
            "train_only": bool(self.train_only_var.get()),
            "games": int(self.total_var.get()),
            "batch": int(self.batch_size_var.get()),
            "steps": int(self.steps_var.get()),
            "save_interval": int(self.save_interval_var.get()),
            "ui_interval": int(self.ui_interval_var.get()),
            "save_batch": int(self.save_batch_var.get()),
            "freeze": self.freeze_var.get(),
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
        if bs in BOARD_SIZES:
            self.board_size_var.set(bs)
            if bs != self.board_size:
                self.board_size = bs
                self.hollow_combo["values"] = hollow_choices(bs)
                self._resize_board_canvas()
        if bs in BOARD_SIZES:
            try:
                h = validate_hollow(bs, int(s.get("hollow", 0)))
            except (TypeError, ValueError):
                h = 0
            self.hollow = h
            self.hollow_var.set(h)
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
                ("candidate_mass", self.candidate_mass_var, "float"),
                ("min_candidates", self.min_candidates_var, "int"),
                ("dir_alpha", self.dir_alpha_var, "float"),
                ("dir_eps", self.dir_eps_var, "float"),
                ("exploration", self.exploration_var, "bool"),
                ("train_only", self.train_only_var, "bool"),
                ("games", self.total_var, "int"),
                ("batch", self.batch_size_var, "int"),
                ("steps", self.steps_var, "int"),
                ("save_interval", self.save_interval_var, "int"),
                ("ui_interval", self.ui_interval_var, "int"),
                ("save_batch", self.save_batch_var, "int"),
                ("freeze", self.freeze_var, "str"),
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
        if self.optimizer_var.get() not in ("Adam", "SGD"):
            self.optimizer_var.set("Adam")
        if self.load_mode_var.get() not in ("newest", "random"):
            self.load_mode_var.set("newest")
        dd = s.get("data_dir")
        if isinstance(dd, str) and dd.strip():
            self.data_dir_var.set(dd.strip())
        elif bs in BOARD_SIZES and bs != BOARD_SIZE:
            self.data_dir_var.set(self._default_data_dir(bs))

    def _create_model_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="模型设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="当前权重:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value=MODEL_PATH)
        ttk.Label(frame, textvariable=self.model_path_var, foreground="#1a5fb4").grid(
            row=0, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(frame, text="训练后原地原子更新，不生成其他权重文件",
                  foreground="gray").grid(row=1, column=1, columnspan=2, sticky=tk.W, padx=5)
        ttk.Label(frame, text="设备:").grid(row=2, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(frame, textvariable=self.device_var, values=["cuda", "cpu"],
                     state="readonly", width=8).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(frame, text="棋盘:").grid(row=3, column=0, sticky=tk.W)
        self.board_size_var = tk.IntVar(value=BOARD_SIZE)
        self.board_combo = ttk.Combobox(frame, textvariable=self.board_size_var,
                                        values=BOARD_SIZES, state="readonly", width=6)
        self.board_combo.grid(row=3, column=1, sticky=tk.W, padx=5)
        self.board_combo.bind("<<ComboboxSelected>>", self._on_board_size_change)
        ttk.Label(frame, text="镂空:").grid(row=4, column=0, sticky=tk.W)
        self.hollow_var = tk.IntVar(value=self.hollow)
        self.hollow_combo = ttk.Combobox(frame, textvariable=self.hollow_var,
                                         values=hollow_choices(self.board_size),
                                         state="readonly", width=6)
        self.hollow_combo.grid(row=4, column=1, sticky=tk.W, padx=5)
        self.hollow_combo.bind("<<ComboboxSelected>>", self._on_hollow_change)
        self.tensorrt_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            frame, text="TensorRT REFIT（实验，需预建引擎）",
            variable=self.tensorrt_var).grid(row=5, column=0, columnspan=3,
                                             sticky=tk.W, pady=(4, 0))
        ttk.Label(frame, text="不匹配/缺失自动回退 PyTorch",
                  foreground="gray").grid(row=6, column=0, columnspan=3,
                                          sticky=tk.W, padx=5)

    def _create_mcts_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="MCTS", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="模拟次数:").grid(row=0, column=0, sticky=tk.W)
        self.simulations_var = tk.IntVar(value=NUM_SIMULATIONS)
        ttk.Spinbox(frame, from_=20, to=GUI_MAX_SIMS, textvariable=self.simulations_var,
                    width=10).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="推理 batch:").grid(row=1, column=0, sticky=tk.W)
        self.mcts_batch_var = tk.IntVar(value=BATCH_SIZE_MCTS)
        ttk.Spinbox(frame, from_=1, to=64, textvariable=self.mcts_batch_var,
                    width=10).grid(row=1, column=1, padx=5)
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
        ttk.Label(frame, text="虚拟损失:").grid(row=5, column=0, sticky=tk.W)
        self.vloss_var = tk.IntVar(value=VIRTUAL_LOSS)
        ttk.Spinbox(frame, from_=0, to=20, textvariable=self.vloss_var,
                    width=10).grid(row=5, column=1, padx=5)
        ttk.Label(frame, text="候选质量:").grid(row=6, column=0, sticky=tk.W)
        self.candidate_mass_var = tk.DoubleVar(value=CANDIDATE_MASS)
        ttk.Spinbox(frame, from_=0.5, to=1.0, increment=0.005,
                    textvariable=self.candidate_mass_var, width=10).grid(row=6, column=1, padx=5)
        ttk.Label(frame, text="最少候选:").grid(row=7, column=0, sticky=tk.W)
        self.min_candidates_var = tk.IntVar(value=MIN_CANDIDATES)
        ttk.Spinbox(frame, from_=1, to=64, textvariable=self.min_candidates_var,
                    width=10).grid(row=7, column=1, padx=5)
        ttk.Label(frame, text="噪声 α:").grid(row=8, column=0, sticky=tk.W)
        self.dir_alpha_var = tk.DoubleVar(value=DIRICHLET_ALPHA)
        ttk.Spinbox(frame, from_=0.001, to=1.0, increment=0.005,
                    textvariable=self.dir_alpha_var, width=10).grid(row=8, column=1, padx=5)
        ttk.Label(frame, text="噪声 ε:").grid(row=9, column=0, sticky=tk.W)
        self.dir_eps_var = tk.DoubleVar(value=DIRICHLET_EPSILON)
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05,
                    textvariable=self.dir_eps_var, width=10).grid(row=9, column=1, padx=5)
        self.exploration_var = tk.BooleanVar(value=EXPLORATION_MODE)
        ttk.Checkbutton(frame, text="四角随机开局", variable=self.exploration_var).grid(
            row=10, column=0, columnspan=2, sticky=tk.W)

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
        ttk.Label(frame, text="保存间隔(局):").grid(row=4, column=0, sticky=tk.W)
        self.save_interval_var = tk.IntVar(value=SAVE_INTERVAL)
        ttk.Spinbox(frame, from_=1, to=10000, textvariable=self.save_interval_var,
                    width=12).grid(row=4, column=1, padx=5)
        ttk.Label(frame, text="UI 更新间隔(batch):").grid(row=5, column=0, sticky=tk.W)
        self.ui_interval_var = tk.IntVar(value=UI_UPDATE_BATCHES)
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.ui_interval_var,
                    width=12).grid(row=5, column=1, padx=5)
        ttk.Label(frame, text="保存间隔(batch):").grid(row=6, column=0, sticky=tk.W)
        self.save_batch_var = tk.IntVar(value=SAVE_INTERVAL_BATCHES)
        ttk.Spinbox(frame, from_=1, to=10000, textvariable=self.save_batch_var,
                    width=12).grid(row=6, column=1, padx=5)
        ttk.Label(frame, text="冻结参数(可空):").grid(row=7, column=0, sticky=tk.W)
        self.freeze_var = tk.StringVar(value="")
        ttk.Entry(frame, textvariable=self.freeze_var, width=24).grid(row=7, column=1, padx=5)

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

    @staticmethod
    def _default_data_dir(board_size):
        n = int(board_size)
        return "data/" if n == 19 else "data%d/" % n

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
                                    self.margin, hollow=self.hollow)
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
        label_bar = ttk.Frame(frame)
        label_bar.pack(pady=(2, 0))
        ttk.Button(label_bar, text="打开数据文件打标签",
                   command=self._open_label_file).pack(side=tk.LEFT, padx=4)
        ttk.Button(label_bar, text="关闭标注",
                   command=self._back_to_live).pack(side=tk.LEFT, padx=4)
        self.label_info = ttk.Label(frame, text="标注：未打开数据文件", foreground="#1a5fb4")
        self.label_info.pack(pady=2)
        self.canvas.bind("<Button-1>", self._on_board_click)

    def _on_board_size_change(self, event=None):
        if self.is_training:
            self.board_size_var.set(self.board_size)
            messagebox.showwarning("提示", "训练中不能切换棋盘，请先停止训练")
            return
        n = int(self.board_size_var.get())
        if n == self.board_size:
            return
        old_defaults = {self._default_data_dir(b) for b in BOARD_SIZES}
        if self.data_dir_var.get() in old_defaults:
            self.data_dir_var.set(self._default_data_dir(n))
        self.board_size = n
        h = int(self.hollow)
        if h > n - 4:
            h = 0
        self.hollow = h
        self.hollow_combo["values"] = hollow_choices(n)
        self.hollow_var.set(h)
        self._resize_board_canvas()
        self.game_history.clear()
        self.current_history_index = -1
        self.current_game_moves = []
        self.current_step = 0
        self.view_mode = "live"
        self.label_game = None
        self.label_index = 0
        self.live_label_game = None
        self.live_label_opening = 0
        if hasattr(self, "label_info"):
            self.label_info.config(text="标注：未打开数据文件", foreground="#1a5fb4")
        self.step_label.config(text="步数: 0/0")
        self.game_info_label.config(text="")
        self._save_settings()

    def _on_hollow_change(self, event=None):
        if self.is_training:
            self.hollow_var.set(self.hollow)
            messagebox.showwarning("提示", "训练中不能切换镂空，请先停止训练")
            return
        try:
            h = validate_hollow(self.board_size, int(self.hollow_var.get()))
        except ValueError as exc:
            messagebox.showerror("镂空非法", str(exc))
            self.hollow_var.set(self.hollow)
            return
        if h == self.hollow:
            return
        self.hollow = h
        self.game_history.clear()
        self.current_history_index = -1
        self.current_game_moves = []
        self.current_step = 0
        self.view_mode = "live"
        self.label_game = None
        self.label_index = 0
        self.live_label_game = None
        self.live_label_opening = 0
        if hasattr(self, "label_info"):
            self.label_info.config(text="标注：未打开数据文件", foreground="#1a5fb4")
        self.step_label.config(text="步数: 0/0")
        self.game_info_label.config(text="")
        self._resize_board_canvas()
        self._save_settings()

    def _resize_board_canvas(self):
        self.cell_size, self.margin, self.canvas_size = self._board_metrics(self.board_size)
        self.board_width = (self.board_size - 1) * self.cell_size
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)
        self.game_board = GameBoard(self.canvas, self.board_size, self.cell_size,
                                    self.margin, hollow=self.hollow)
        self._draw_empty_board()

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
    def _draw_empty_board(self):
        self.canvas.delete("all")
        self.game_board.draw_grid()
        self.game_board.draw_stars()

    def _render_replay(self, moves, step, hollow=None):
        h = self.game_board.hollow if hollow is None else hollow
        game = GoGame(self.board_size, hollow=h)
        for i in range(min(step, len(moves))):
            r, c = moves[i]
            if (r, c) == PASS_MOVE:
                game.make_move(-1, -1)
            else:
                game.make_move(r, c)
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
        self._update_live_label_info(step)

    # ---------------- 磁盘样本标注 ----------------
    def _open_label_file(self):
        f = filedialog.askopenfilename(initialdir=self.data_dir_var.get() or ".",
                                       filetypes=[("Go npz", "*.npz")])
        if not f:
            return
        try:
            game = LabelGame(f)
        except Exception as e:
            messagebox.showerror("无法打开数据文件", "%s\n\n%s" % (f, e))
            return
        if game.board_size != self.board_size:
            if self.is_training:
                messagebox.showerror("棋盘不一致",
                                     "文件是 %d 路，当前界面 %d 路；请先停止训练再切换棋盘"
                                     % (game.board_size, self.board_size))
                return
            self.board_size_var.set(game.board_size)
            self._on_board_size_change()
        if game.hollow != self.hollow:
            if self.is_training:
                messagebox.showerror("镂空不一致",
                                     "文件是镂空 %d，当前界面镂空 %d；请先停止训练再切换"
                                     % (game.hollow, self.hollow))
                return
            self.hollow_var.set(game.hollow)
            self._on_hollow_change()
        self.label_game = game
        self.label_index = 0
        self.view_mode = "label"
        self._render_label_sample()
        self._log_message("[标注] 已打开 %s（%d 条样本，%d 路）"
                          % (os.path.basename(game.path), game.count, game.board_size))

    def _render_label_sample(self):
        game = self.label_game
        if game is None or game.count == 0:
            return
        i = max(0, min(int(self.label_index), game.count - 1))
        self.label_index = i
        board = game.board(i)
        if self.game_board.hollow != game.hollow:
            self.game_board.set_hollow(game.hollow)
        self._draw_empty_board()
        for r in range(self.board_size):
            for c in range(self.board_size):
                if self.game_board.is_hole(r, c):
                    continue
                if board[r, c] != 0:
                    self.game_board.draw_piece(r, c, int(board[r, c]))
        last = game.last_move(i)
        if last is not None:
            self.game_board.highlight_move(*last)
        self.step_label.config(text="样本: %d/%d" % (i + 1, game.count))
        w = game.winner
        hollow = (" 镂空%d" % game.hollow) if game.hollow else ""
        self.game_info_label.config(
            text="%s | %d路%s | %s" % (os.path.basename(game.path), game.board_size,
                                        hollow,
                                        "黑胜" if w == 1 else ("白胜" if w == -1 else "平局")))
        top, prob = game.top_policy(i)
        old = game.move_text(top) if top is not None else "--"
        target = game.target_move(i)
        played = "Pass" if target is None else "(%d,%d)" % target
        player = "黑" if game.player(i) == 1 else "白"
        self.label_info.config(
            text="样本 %d/%d | 行棋方 %s | 原实战 %s | 当前标签 %s p=%.3f | 盘面为该样本落子前；点击空点=改为 one-hot"
                 % (i + 1, game.count, player, played, old, prob),
            foreground="#1a5fb4")

    def _prev_label_sample(self):
        if self.label_game is None:
            return
        self.label_index = max(0, self.label_index - 1)
        self._render_label_sample()

    def _next_label_sample(self):
        if self.label_game is None:
            return
        self.label_index = min(self.label_game.count - 1, self.label_index + 1)
        self._render_label_sample()

    def _back_to_live(self):
        self.view_mode = "live"
        self.label_game = None
        self.label_index = 0
        if hasattr(self, "label_info"):
            self.label_info.config(text="标注：未打开数据文件", foreground="#1a5fb4")
        if self.game_history and 0 <= self.current_history_index < len(self.game_history):
            self._display_game(self.game_history[self.current_history_index])
        else:
            self.live_label_game = None
            self.live_label_opening = 0
            self.game_board.set_hollow(self.hollow)
            self.hollow_var.set(self.hollow)
            self._draw_empty_board()
            self.step_label.config(text="步数: 0/0")
            self.game_info_label.config(text="")

    def _sync_buffer_label(self, path, sample_index, move):
        """把内存回放缓冲里的同一条样本也改掉；训练中下一 batch 即生效。"""
        if self.trainer is None:
            return
        path = os.path.normcase(os.path.abspath(path))
        hit = 0
        for s in list(self.trainer.data_buffer.data):
            if (getattr(s, "source", None) == path
                    and getattr(s, "index", -1) == int(sample_index)):
                s.pidx = np.array([int(move)], dtype=np.uint16)
                s.pp = np.array([1.0], dtype=np.float16)
                hit += 1
        if hit:
            self._log_message("[标注] 内存缓冲同步 %d 条样本" % hit)
        else:
            self._log_message("[标注] 当前缓冲没有该样本；磁盘已改，重启训练后生效")

    def _update_live_label_info(self, step):
        """对局回放模式下，显示「当前第 n-1 手后，点击=第 n 手标签」。"""
        if self.view_mode != "live":
            return
        lg = self.live_label_game
        if lg is None:
            self.label_info.config(
                text="标注：当前对局未保存为 .npz（未勾选保存/未完成），无法改硬盘标签",
                foreground="#1a5fb4")
            return
        i = int(step) - self.live_label_opening
        if not 0 <= i < lg.count:
            self.label_info.config(
                text="标注：当前第 %d 手不是样本落子前；请调到第 %d..%d 手之间"
                     % (step, self.live_label_opening,
                        self.live_label_opening + lg.count - 1),
                foreground="#1a5fb4")
            return
        top, prob = lg.top_policy(i)
        old = lg.move_text(top) if top is not None else "--"
        target = lg.target_move(i)
        played = "Pass" if target is None else "(%d,%d)" % target
        suffix = "" if lg.path is not None else " | 未保存，不能写盘"
        self.label_info.config(
            text="第 %d 手后 | 目标第 %d 手 | 样本 %d/%d | 原实战 %s | 当前标签 %s p=%.3f | 点击空点=one-hot%s"
                 % (step, step + 1, i + 1, lg.count, played, old, prob, suffix),
            foreground="#1a5fb4")

    def _on_live_board_click(self, event):
        lg = self.live_label_game
        if lg is None:
            return
        if lg.path is None:
            self.label_info.config(text="标注失败: 当前对局没有保存为 .npz", foreground="red")
            return
        i = self.current_step - self.live_label_opening
        if not 0 <= i < lg.count:
            self.label_info.config(
                text="标注失败: 请先把步数调到样本落子前（第 %d..%d 手）"
                     % (self.live_label_opening,
                        self.live_label_opening + lg.count - 1),
                foreground="red")
            return
        pos = self.game_board.coord_to_index(event.x, event.y)
        if pos is None:
            return
        r, c = pos
        try:
            move = lg.set_onehot(i, r, c)
        except Exception as e:
            self.label_info.config(text="标注失败: %s" % e, foreground="red")
            self._log_message("[标注] 失败: %s" % e)
            return
        self._sync_buffer_label(lg.path, i, move)
        self._render_replay(self.current_game_moves, self.current_step)
        self.label_info.config(
            text="已写入第 %d 手（样本 %d/%d）：%s one-hot"
                 % (self.current_step + 1, i + 1, lg.count, lg.move_text(move)),
            foreground="green")
        self._log_message("[标注] 第 %d 手（样本 %d/%d）-> %s one-hot"
                          % (self.current_step + 1, i + 1, lg.count, lg.move_text(move)))

    def _on_board_click(self, event):
        if self.view_mode == "label":
            self._on_label_board_click(event)
        elif self.view_mode == "live":
            self._on_live_board_click(event)

    def _on_label_board_click(self, event):
        if self.label_game is None:
            return
        pos = self.game_board.coord_to_index(event.x, event.y)
        if pos is None:
            return
        r, c = pos
        i = self.label_index
        try:
            move = self.label_game.set_onehot(i, r, c)
        except Exception as e:
            self.label_info.config(text="标注失败: %s" % e, foreground="red")
            self._log_message("[标注] 失败: %s" % e)
            return
        self._sync_buffer_label(self.label_game.path, i, move)
        self._render_label_sample()
        self.label_info.config(
            text="已写入样本 %d/%d: %s one-hot"
                 % (i + 1, self.label_game.count,
                    self.label_game.move_text(move)),
            foreground="green")
        self._log_message("[标注] 样本 %d/%d -> %s one-hot"
                          % (i + 1, self.label_game.count, self.label_game.move_text(move)))

    def _display_game(self, game_result):
        self.view_mode = "live"
        self.current_game_moves = game_result.get_moves()
        self.current_step = len(self.current_game_moves)
        self.live_label_game = getattr(game_result, "label_game", None)
        if self.live_label_game is not None and len(self.current_game_moves) >= self.live_label_game.count:
            self.live_label_opening = len(self.current_game_moves) - self.live_label_game.count
        else:
            self.live_label_opening = 0
        if game_result.hollow != self.game_board.hollow:
            self.game_board.set_hollow(game_result.hollow)
            self.hollow_var.set(game_result.hollow)
            self._draw_empty_board()
        self._render_replay(self.current_game_moves, self.current_step,
                            hollow=game_result.hollow)
        w = game_result.winner
        hollow = (" 镂空%d" % game_result.hollow) if game_result.hollow else ""
        self.game_info_label.config(text=f"第 {game_result.game_id} 局 ({game_result.board_size}路{hollow}) | "
                                         + ("黑胜" if w == 1 else ("白胜" if w == -1 else "平局")))

    def _prev_game(self):
        if self.view_mode == "label":
            self._back_to_live()
            return
        if not self.game_history:
            return
        self.current_history_index = max(0, self.current_history_index - 1)
        self._display_game(self.game_history[self.current_history_index])

    def _next_game(self):
        if self.view_mode == "label":
            self._back_to_live()
            return
        if not self.game_history:
            return
        self.current_history_index = min(len(self.game_history) - 1, self.current_history_index + 1)
        self._display_game(self.game_history[self.current_history_index])

    def _first_step(self):
        if self.view_mode == "label":
            self.label_index = 0
            self._render_label_sample()
            return
        if self.current_game_moves:
            self.current_step = 1
            self._render_replay(self.current_game_moves, self.current_step)

    def _prev_step(self):
        if self.view_mode == "label":
            self._prev_label_sample()
            return
        if self.current_game_moves and self.current_step > 0:
            self.current_step -= 1
            self._render_replay(self.current_game_moves, self.current_step)

    def _next_step(self):
        if self.view_mode == "label":
            self._next_label_sample()
            return
        if self.current_game_moves and self.current_step < len(self.current_game_moves):
            self.current_step += 1
            self._render_replay(self.current_game_moves, self.current_step)

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
                cfg["candidate_mass"] = self.candidate_mass_var.get()
                cfg["min_candidates"] = int(self.min_candidates_var.get())
            self._log_message("[超参数] 训练中已更新 lr/wd；模拟数/c_puct/噪声/候选从下一局生效；"
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
        cfg = {
            "model_path": model_path,
            "board_size": self.board_size,
            "hollow": self.hollow,
            "device": self.device_var.get(),
            "sims": self.simulations_var.get(),
            "mcts_batch": self.mcts_batch_var.get(),
            "c_puct": self.c_puct_var.get(),
            "temperature": self.temperature_var.get(),
            "alpha": self.alpha_var.get(),
            "vloss": self.vloss_var.get(),
            "dir_alpha": self.dir_alpha_var.get(),
            "dir_eps": self.dir_eps_var.get(),
            "candidate_mass": self.candidate_mass_var.get(),
            "min_candidates": self.min_candidates_var.get(),
            "exploration": self.exploration_var.get(),
            "train_only": self.train_only_var.get(),
            "games": self.total_var.get(),
            "batch": self.batch_size_var.get(),
            "steps": self.steps_var.get(),
            "save_interval": self.save_interval_var.get(),
            "ui_interval": self.ui_interval_var.get(),
            "save_batch": self.save_batch_var.get(),
            "freeze": self.freeze_var.get().strip() or None,
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
                use_tensorrt=cfg["tensorrt"],
                log=lambda msg: self.update_queue.put(
                    {"type": "log", "message": msg}))
            self.trainer = trainer
            self.update_queue.put({"type": "log", "message":
                                   "[棋盘] %d 路 | 镂空 %s | 数据目录: %s | 温度 %.2f→0@%d手 | 噪声 α=%.3f ε=%.2f | 候选 %.3f/%d"
                                   % (cfg["board_size"],
                                      ("%dx%d" % (cfg["hollow"], cfg["hollow"])) if cfg["hollow"] else "无",
                                      cfg["data_dir"],
                                      cfg["temperature"],
                                      board_temperature_zero(cfg["board_size"], cfg["hollow"]),
                                      cfg["dir_alpha"], cfg["dir_eps"],
                                      cfg["candidate_mass"], cfg["min_candidates"])})
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
                        candidate_mass=cfg["candidate_mass"],
                        min_candidates=cfg["min_candidates"],
                        board_size=cfg["board_size"], virtual_loss=cfg["vloss"])

        mcts = make_mcts()
        games = 0
        while games < cfg["games"] and self.is_training and not self.stop_event.is_set():
            # 训练中改模拟数/c_puct 时，在局间重建搜索器；树在局间本来就会 reset。
            if (mcts.num_simulations != trainer.num_simulations
                    or abs(mcts.c_puct - trainer.c_puct) > 1e-9
                    or abs(mcts.dirichlet_alpha - cfg["dir_alpha"]) > 1e-12
                    or abs(mcts.dirichlet_epsilon - cfg["dir_eps"]) > 1e-12
                    or abs(mcts.candidate_mass - cfg["candidate_mass"]) > 1e-12
                    or mcts.min_candidates != int(cfg["min_candidates"])):
                mcts = make_mcts()
                self.update_queue.put({"type": "log", "message":
                                       "[MCTS] 已应用 模拟数=%d c_puct=%.3f 噪声α=%.3f ε=%.2f "
                                       "候选质量=%.3f 最少候选=%d"
                                       % (trainer.num_simulations, trainer.c_puct,
                                          cfg["dir_alpha"], cfg["dir_eps"],
                                          cfg["candidate_mass"], cfg["min_candidates"])})
            t0 = time.time()
            result = play_one_game(
                mcts=mcts, board_size=cfg["board_size"],
                num_simulations=trainer.num_simulations,
                device=cfg["device"], c_puct=trainer.c_puct,
                temperature=cfg["temperature"], exploration=cfg["exploration"],
                on_move=lambda played, ns: self.update_queue.put(
                    {"type": "search_progress", "played": played, "samples": ns}),
                hollow=cfg["hollow"])
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
            label_game = None
            if result["payload"] is not None:
                try:
                    arr = dict(result["payload"])
                    arr["version"] = np.int16(DATA_VERSION)
                    arr["board_size"] = np.int16(cfg["board_size"])
                    label_game = LabelGame(saved_path, arr)
                except Exception as e:
                    self.update_queue.put({"type": "log", "message":
                                           "[标注] 本局无法建立标注视图: %s" % e})
            gr = GameResult(games, result["moves"], result["winner"], cfg["board_size"],
                            hollow=cfg["hollow"], label_game=label_game)
            self.update_queue.put({"type": "game", "game_result": gr,
                                   "game_count": games, "loss": loss,
                                   "seconds": time.time() - t0,
                                   "buffer": len(trainer.data_buffer)})
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
                        # 正在标注磁盘样本时不要被新对局顶掉视角；关闭标注后再看。
                        if self.view_mode != "label":
                            self._display_game(gr)
                    if msg["loss"][0] is not None:
                        self.stats_panel.update_loss(*msg["loss"])
                    w = gr.winner
                    res = "黑胜" if w == 1 else ("白胜" if w == -1 else "平局")
                    if msg["loss"][0] is not None:
                        self._log_message(
                            "[游戏 %d] %s | 手数 %d | %.1fs | 缓冲 %d | 平均 p=%.4f own=%.4f win=%.4f total=%.4f"
                            % (msg["game_count"], res, len(gr.moves),
                               msg["seconds"], msg["buffer"],
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
                elif typ == "search_progress":
                    if msg["played"] % 10 == 0:
                        self._log_message("[搜索] 第 %d 手（样本 %d）"
                                          % (msg["played"], msg["samples"]))
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