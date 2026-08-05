import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import re
import multiprocessing as mp
import os
import subprocess
import sys
import pickle
import struct
from datetime import datetime
from selfplay import worker_process   # 数据生成 worker（纯生成，不加载 torch）
from ui_utils import GameBoard, TrainingStatsPanel
import numpy as np
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)  # 让Tkinter使用DirectX渲染

# 超参数（统一配置见 hyperparams.py）
from hyperparams import (BOARD_SIZE, ONNX_PATH, MODEL_PATH, DATA_DIR,
                         NUM_SIMULATIONS, C_PUCT, TEMPERATURE,
                         EXPLORATION_MODE, BATCH_SIZE, SAVE_INTERVAL,
                         TRAIN_GAMES, LEARNING_RATE,
                         WEIGHT_DECAY, MOMENTUM, OPTIMIZER_NAME, EVAL_GAMES,
                         UI_UPDATE_BATCHES, SAVE_INTERVAL_BATCHES, ALPHA)

class GameResult:
    def __init__(self, game_id, moves, winner):
        self.game_id = game_id
        self.moves = moves 
        self.winner = winner

class TrainingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("围棋AI训练")
        self.root.geometry("2000x1300")

        self.trainer = None
        self.training_processes = []
        self.is_training = False
        self.update_queue = queue.Queue()
        self.stop_event = mp.Event()
        self.game_history = []
        self.current_history_index = -1
        self.current_game_moves = []
        self.current_winner = None
        self.current_step = 0

        # 初始化模型并导出ONNX（若不存在）
        if not os.path.exists(ONNX_PATH):
            from model import PolicyValueNet   # 延迟导入：worker 子进程不加载 torch
            if os.path.exists(MODEL_PATH):
                model = PolicyValueNet.load_model(MODEL_PATH, device='cuda')
            else:
                model = PolicyValueNet()
                model.save_model(MODEL_PATH)
            model.export_onnx(ONNX_PATH)

        self._create_widgets()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)   # 关闭窗口时顺带结束评估进程
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

    def _create_model_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="模型设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="模型路径:").grid(row=0, column=0, sticky=tk.W)
        self.model_path_var = tk.StringVar(value=MODEL_PATH)
        ttk.Entry(frame, textvariable=self.model_path_var, width=30).grid(row=0, column=1, padx=5)
        ttk.Button(frame, text="浏览", command=self._browse_model).grid(row=0, column=2)
        ttk.Label(frame, text="设备:").grid(row=1, column=0, sticky=tk.W)
        self.device_var = tk.StringVar(value="cuda")
        ttk.Combobox(frame, textvariable=self.device_var, values=["cuda", "cpu"], state="readonly").grid(row=1, column=1, sticky=tk.W, padx=5)

    def _create_mcts_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="MCTS设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="模拟次数:").grid(row=0, column=0, sticky=tk.W)
        self.simulations_var = tk.IntVar(value=NUM_SIMULATIONS)
        ttk.Spinbox(frame, from_=20, to=500, textvariable=self.simulations_var, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(frame, text="c_puct:").grid(row=1, column=0, sticky=tk.W)
        self.c_puct_var = tk.DoubleVar(value=C_PUCT)
        ttk.Scale(frame, from_=0.0, to=10.0, variable=self.c_puct_var, orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=5)
        self.c_puct_label = ttk.Label(frame, text=str(C_PUCT))
        self.c_puct_label.grid(row=1, column=2)
        self.c_puct_var.trace('w', lambda *a: self.c_puct_label.configure(text=f"{self.c_puct_var.get():.1f}"))
        ttk.Label(frame, text="温度:").grid(row=2, column=0, sticky=tk.W)
        self.temperature_var = tk.DoubleVar(value=TEMPERATURE)
        ttk.Scale(frame, from_=0.1, to=2.0, variable=self.temperature_var, orient=tk.HORIZONTAL, length=150).grid(row=2, column=1, padx=5)
        self.temp_label = ttk.Label(frame, text=str(TEMPERATURE))
        self.temp_label.grid(row=2, column=2)
        self.temperature_var.trace('w', lambda *a: self.temp_label.configure(text=f"{self.temperature_var.get():.2f}"))

    def _create_training_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="训练设置", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.train_only_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(frame, text="仅训练模式", variable=self.train_only_var).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(frame, text="训练局数:").grid(row=1, column=0, sticky=tk.W)
        self.total_var = tk.IntVar(value=TRAIN_GAMES)
        ttk.Spinbox(frame, from_=10, to=10000, textvariable=self.total_var, width=15).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="进程数:").grid(row=2, column=0, sticky=tk.W)
        self.threads_var = tk.IntVar(value=1)   # 线程数（每线程独立计数评估，1=最稳定）
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.threads_var, width=10).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="批大小:").grid(row=3, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=BATCH_SIZE)
        ttk.Spinbox(frame, from_=32, to=4096, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="保存间隔(局,自对弈):").grid(row=4, column=0, sticky=tk.W)
        self.save_interval_var = tk.IntVar(value=SAVE_INTERVAL)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.save_interval_var, width=10).grid(row=4, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="更新UI间隔(Batch):").grid(row=5, column=0, sticky=tk.W)
        self.ui_update_interval_var = tk.IntVar(value=UI_UPDATE_BATCHES)
        ttk.Spinbox(frame, from_=1, to=100, textvariable=self.ui_update_interval_var, width=10).grid(row=5, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="保存模型间隔(Batch):").grid(row=6, column=0, sticky=tk.W)
        self.save_interval_batch_var = tk.IntVar(value=SAVE_INTERVAL_BATCHES)
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.save_interval_batch_var, width=10).grid(row=6, column=1, padx=5, sticky=tk.W)
        self.exploration_var = tk.BooleanVar(value=EXPLORATION_MODE)
        ttk.Checkbutton(frame, text="探索模式", variable=self.exploration_var).grid(row=7, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(frame, text="冻结参数(可空):").grid(row=8, column=0, sticky=tk.W)
        self.freeze_var = tk.StringVar(value='')
        ttk.Entry(frame, textvariable=self.freeze_var, width=24).grid(row=8, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="例: 0,1,2,3,5,7 冻结这些残差块；0,own,policy 冻结块0+归属头+策略头",
                  foreground="gray").grid(row=9, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(frame, text="α混合(0~1):").grid(row=10, column=0, sticky=tk.W)
        self.alpha_var = tk.DoubleVar(value=ALPHA)   # MCTS价值混合：0=纯胜率logit，1=纯目差
        ttk.Spinbox(frame, from_=0.0, to=1.0, increment=0.05, textvariable=self.alpha_var, width=10).grid(row=10, column=1, padx=5, sticky=tk.W)

    def _create_optimizer_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="优化器", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="优化器:").grid(row=0, column=0, sticky=tk.W)
        self.optimizer_name_var = tk.StringVar(value=OPTIMIZER_NAME)
        ttk.Combobox(frame, textvariable=self.optimizer_name_var, values=["Adam", "SGD"],
                     state="readonly", width=8).grid(row=0, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="学习率:").grid(row=1, column=0, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=LEARNING_RATE)
        ttk.Spinbox(frame, from_=0.00001, to=0.02, increment=0.0005,
                    textvariable=self.lr_var, width=12).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="动量(仅SGD):").grid(row=2, column=0, sticky=tk.W)
        self.momentum_var = tk.DoubleVar(value=MOMENTUM)
        ttk.Spinbox(frame, from_=0.0, to=0.999, increment=0.05,
                    textvariable=self.momentum_var, width=12).grid(row=2, column=1, padx=5)
        ttk.Label(frame, text="权重衰减:").grid(row=3, column=0, sticky=tk.W)
        self.wd_var = tk.DoubleVar(value=WEIGHT_DECAY)
        ttk.Spinbox(frame, from_=0.0, to=0.001, increment=0.00005,
                    textvariable=self.wd_var, width=12).grid(row=3, column=1, padx=5)

    def _create_data_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="数据管理", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        self.load_data_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(frame, text="加载已有数据", variable=self.load_data_var).pack(anchor=tk.W)
        self.data_dir_var = tk.StringVar(value=DATA_DIR)
        ttk.Entry(frame, textvariable=self.data_dir_var, width=25).pack(fill=tk.X, pady=2)

    def _create_buttons(self, parent):
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=10)
        
        # 第一行：训练控制按钮
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X, pady=2)
        self.start_btn = ttk.Button(row1, text="开始训练", command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(row1, text="停止训练", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.apply_btn = ttk.Button(row1, text="应用超参数", command=self._apply_hyperparams)
        self.apply_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="保存模型", command=self._save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(row1, text="加载数据", command=self._load_data).pack(side=tk.LEFT, padx=5)
        
        # 第二行：评估控制
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, pady=2)
        self.eval_games_var = tk.IntVar(value=EVAL_GAMES)
        ttk.Label(row2, text="评估局数:").pack(side=tk.LEFT, padx=(8, 0))
        ttk.Spinbox(row2, from_=1, to=20, width=3, textvariable=self.eval_games_var).pack(side=tk.LEFT)
        self.eval_btn = ttk.Button(row2, text="评估N次", command=self._start_eval)
        self.eval_btn.pack(side=tk.LEFT, padx=5)

    def _create_board_area(self, parent):
        frame = ttk.LabelFrame(parent, text="对局回放", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.board_size = BOARD_SIZE
        self.cell_size = 50
        self.margin = 50
        self.board_width = (self.board_size - 1) * self.cell_size
        self.canvas_size = self.board_width + 2 * self.margin
        self.canvas = tk.Canvas(frame, width=self.canvas_size, height=self.canvas_size,
                                bg='#D1D1D1', highlightthickness=0)
        self.canvas.pack(pady=10)
        self.game_board = GameBoard(self.canvas, self.board_size, self.cell_size, self.margin)

        control = ttk.Frame(frame)
        control.pack(pady=5)
        ttk.Button(control, text="◀ 上一局", command=self._prev_game, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="下一局 ▶", command=self._next_game, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Separator(control, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(control, text="⏮ 第一步", command=self._first_step, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="◀ 上一步", command=self._prev_step, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="下一步 ▶", command=self._next_step, width=8).pack(side=tk.LEFT, padx=5)
        self.step_label = ttk.Label(control, text="步数: 0/0", font=('微软雅黑', 10))
        self.step_label.pack(side=tk.LEFT, padx=10)
        self.game_info_label = ttk.Label(frame, text="", font=('微软雅黑', 10))
        self.game_info_label.pack(pady=5)

    def _create_status_area(self, parent):
        frame = ttk.LabelFrame(parent, text="训练状态", padding="10")
        frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.status_text = tk.Text(frame, height=10, font=('Consolas', 9), wrap=tk.WORD)
        sb = ttk.Scrollbar(frame, command=self.status_text.yview)
        self.status_text.configure(yscrollcommand=sb.set)
        self.status_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self.stats_panel = TrainingStatsPanel(frame)
        self.stats_panel.create()

    # ---------- 棋盘绘制 ----------
    def _draw_empty_board(self):
        self.canvas.delete("all")
        self.game_board.draw_grid()
        self.game_board.draw_stars()

    def _draw_board_from_moves(self, moves, step):
        import game
        # 临时解除 Pass 限制（回放时允许任意步 Pass）
        old_min = game.MIN_MOVES_BEFORE_PASS
        game.MIN_MOVES_BEFORE_PASS = 0
        try:
            temp_game = game.GoGame()
            # 按顺序落子到 step 步
            for i in range(step):
                  r, c, _ = moves[i]
                  if (r, c) == (-1, -1):
                        temp_game.make_move(-1, -1)
                  else:
                        temp_game.make_move(r, c)
                        # 绘制最终棋盘
                  self._draw_empty_board()
                  for r in range(self.board_size):
                        for c in range(self.board_size):
                              p = temp_game.board[r, c]
                              if p:
                                    self.game_board.draw_piece(r, c, p)
                  if step > 0:
                        last_r, last_c, _ = moves[step - 1]
                        if last_r != -1:
                              self.game_board.highlight_move(last_r, last_c)
                  self.step_label.config(text=f"步数: {step}/{len(moves)}")
        finally:
            # 恢复原限制
            game.MIN_MOVES_BEFORE_PASS = old_min

    def _display_game(self, game_result):
        if game_result is None:
            self._draw_empty_board()
            self.game_info_label.config(text="")
            self.step_label.config(text="步数: 0/0")
            return
        self.current_game_moves = game_result.moves
        self.current_winner = game_result.winner
        self.current_step = len(self.current_game_moves)
        self._draw_board_from_moves(self.current_game_moves, self.current_step)
        winner_text = "黑胜" if self.current_winner == 1 else "白胜" if self.current_winner == -1 else "平局"
        self.game_info_label.config(
            text=f"游戏 #{game_result.game_id} | 结果: {winner_text} | 手数: {len(self.current_game_moves)}"
        )
        self.stats_panel.update_current_game(game_result.game_id)

    def _prev_game(self):
        if self.game_history and self.current_history_index > 0:
            self.current_history_index -= 1
            self._display_game(self.game_history[self.current_history_index])

    def _next_game(self):
        if self.game_history and self.current_history_index < len(self.game_history) - 1:
            self.current_history_index += 1
            self._display_game(self.game_history[self.current_history_index])
    def _first_step(self):
        if self.current_game_moves:
            self.current_step = 1
            self._draw_board_from_moves(self.current_game_moves, self.current_step)

    def _prev_step(self):
        if self.current_game_moves and self.current_step > 0:
            self.current_step -= 1
            self._draw_board_from_moves(self.current_game_moves, self.current_step)

    def _next_step(self):
        if self.current_game_moves and self.current_step < len(self.current_game_moves):
            self.current_step += 1
            self._draw_board_from_moves(self.current_game_moves, self.current_step)

    # ---------- 按钮回调 ----------
    def _browse_model(self):
        f = filedialog.askopenfilename(title="选择模型文件", filetypes=[("PyTorch模型", "*.pt")])
        if f:
            self.model_path_var.set(f)
            # 换模型后立即重新导出 ONNX：不同 channels/blocks 的 .pt 其 ONNX 也各不同，
            # 不刷新会让 worker/评估继续用旧的 model.onnx（静默评估错模型）。
            try:
                from model import PolicyValueNet   # 延迟导入（worker 子进程不加载 torch）
                m = PolicyValueNet.load_model(f, device='cpu')
                m.export_onnx(ONNX_PATH)
                self._log_message('[模型] 已按所选模型重新导出 ONNX: %s' % ONNX_PATH)
            except Exception as e:
                self._log_message('[错误] 导出 ONNX 失败: %s' % e)

    def _apply_hyperparams(self):
        if self.is_training:
            messagebox.showwarning("警告", "训练中请先停止")
            return
        if self.trainer:
            self.trainer.update_hyperparameters(
                learning_rate=self.lr_var.get(),
                weight_decay=self.wd_var.get(),
                optimizer_name=self.optimizer_name_var.get(),
                momentum=self.momentum_var.get(),
                c_puct=self.c_puct_var.get(),
                num_simulations=self.simulations_var.get()
            )
            self._log_message("[超参数] 已更新")

    def _start_training(self):
        if self.is_training:
            return
        if getattr(self, '_eval_proc', None) is not None and self._eval_proc.poll() is None:
            self._log_message("[训练] 评估进行中，请先等评估结束")
            return
        freeze_spec = self.freeze_var.get().strip()
        from train import SelfPlayTrainer   # 延迟导入：worker 子进程不加载 torch
        try:
            self.trainer = SelfPlayTrainer(
                model_path=self.model_path_var.get(),
                board_size=BOARD_SIZE,
                device=self.device_var.get(),
                num_simulations=self.simulations_var.get(),
                data_dir=self.data_dir_var.get(),
                c_puct=self.c_puct_var.get(),
                learning_rate=self.lr_var.get(),
                weight_decay=self.wd_var.get(),
                optimizer_name=self.optimizer_name_var.get(),
                momentum=self.momentum_var.get(),
                freeze_parts=freeze_spec or None
            )
        except Exception as e:
            self._log_message('[错误] 创建训练器失败（检查冻结参数格式）: %s' % e)
            return
        if freeze_spec:
            self._log_message('[训练] 冻结参数: %s（%d 个参数）'
                              % (freeze_spec, self.trainer.frozen_params_count))
        if not os.path.exists(ONNX_PATH):
            self.trainer.model.export_onnx(ONNX_PATH)

        self.game_history = []
        self.current_history_index = -1
        self.status_text.delete(1.0, tk.END)
        self._log_message("=" * 60)
        self._log_message(f"训练开始: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        self.is_training = True
        self.stop_event.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)

        if self.train_only_var.get():
            thread = threading.Thread(
                target=self._train_only_loop,
                args=(self.batch_size_var.get(),
                      self.ui_update_interval_var.get(),
                      self.save_interval_batch_var.get(),
                      self.load_data_var.get()),
                daemon=True
            )
            thread.start()
        else:
            params = {
                'board_size': BOARD_SIZE,
                'num_simulations': self.simulations_var.get(),
                'c_puct': self.c_puct_var.get(),
                'temperature': self.temperature_var.get(),
                'exploration_mode': self.exploration_var.get(),
                'alpha': self.alpha_var.get()
            }
            result_queue = mp.Queue()
            self.training_processes = []
            for _ in range(self.threads_var.get()):
                p = mp.Process(
                    target=worker_process,
                    args=(params, self.model_path_var.get(), result_queue, self.stop_event, self.device_var.get())
                )
                p.start()
                self.training_processes.append(p)
            thread = threading.Thread(
                target=self._training_loop,
                args=(result_queue, self.total_var.get(), self.batch_size_var.get(),
                      self.save_interval_var.get(), self.load_data_var.get(),
                      (params, self.model_path_var.get(), self.device_var.get())),
                daemon=True
            )
            thread.start()

    def _train_only_loop(self, batch_size, update_interval, save_interval, load_data):
        """仅训练模式：无限循环，只用 Batch 作单位。
        每 update_interval 个 Batch 向界面推送一次损失均值（显示该区间均值）；
        每 save_interval 个 Batch 保存一次模型；停止时补存一次并收尾。"""
        try:
            if load_data:
                self.trainer.load_training_data()
            if len(self.trainer.data_buffer) < batch_size:
                self._log_message(f"[错误] 数据不足 ({len(self.trainer.data_buffer)} < {batch_size})")
                self.update_queue.put({'type': 'finished'})
                return
            batch_count = 0
            acc = [0.0] * 5              # policy / own / entropy / unc / total 累加
            while self.is_training:
                loss = self.trainer.train_step(batch_size)
                if loss[0] is None:
                    continue   # 数据不足时跳过（已加载数据，正常情况下不会发生）
                self.trainer.train_count += 1
                batch_count += 1
                for i in range(5):
                    acc[i] += loss[i]
                if batch_count % update_interval == 0:
                    mean = tuple(a / update_interval for a in acc)
                    acc = [0.0] * 5
                    self.update_queue.put({
                        'type': 'train_progress',
                        'loss': mean,
                        'batch': batch_count,
                    })
                if batch_count % save_interval == 0:
                    self.trainer.save_model()
                    self.update_queue.put({'type': 'log',
                                           'message': f"[模型] 已保存 (Batch {batch_count})"})
            if batch_count % save_interval != 0:
                self.trainer.save_model()   # 停止时补存一次
            self.update_queue.put({'type': 'finished', 'batches': batch_count})
        except Exception as e:
            self.update_queue.put({'type': 'error', 'message': str(e)})

    def _training_loop(self, result_queue, total_games, batch_size, save_interval, load_data,
                       worker_args=None):
        try:
            if load_data:
                self.trainer.load_training_data()
            games = 0
            while games < total_games and self.is_training:
                # worker看门狗：进程意外退出（显存不足/TRT原生崩溃等）→自动重启，训练不终止
                if worker_args is not None:
                    for i, p in enumerate(self.training_processes):
                        if not p.is_alive():
                            params, model_path, device = worker_args
                            np_ = mp.Process(target=worker_process,
                                             args=(params, model_path, result_queue,
                                                   self.stop_event, device))
                            np_.start()
                            self.training_processes[i] = np_
                            self.update_queue.put({'type': 'log',
                                                   'message': f"[Worker]进程{i}意外退出，已自动重启"})
                try:
                    msg = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                if isinstance(msg, tuple) and len(msg) == 2 and msg[0] == 'eval_log':
                    self.update_queue.put({'type': 'log', 'message': msg[1]})   # 评估日志走UI日志区
                    continue
                states, policies, players, moves, winner, score_diff, ownership = msg
                games += 1
                self.trainer.data_buffer.add_game(
                    states, policies, players, winner, score_diff, ownership)
                self.trainer.game_count += 1
                self.trainer.save_training_data(
                    (states, policies, players, winner, score_diff, ownership))
                loss = self.trainer.train_step(batch_size)
                if loss[0] is not None:
                    self.trainer.train_count += 1

                game_result = GameResult(games, moves, winner)
                self.update_queue.put({
                    'type': 'game',
                    'game_result': game_result,
                    'game_count': games,
                    'score_diff': score_diff,
                    'loss': loss
                })

                if games % save_interval == 0:
                    self.trainer.save_model()
                    self.update_queue.put({'type': 'log', 'message': f"[模型] 已保存 (游戏 {games})"})
            self.stop_event.set()
            for p in self.training_processes:
                p.join(timeout=2)
            self.trainer.save_model()
            self.update_queue.put({'type': 'finished'})
        except Exception as e:
            self.update_queue.put({'type': 'error', 'message': str(e)})

    def _stop_training(self):
        self.is_training = False
        self.stop_event.set()
        self._log_message("[用户] 停止训练...")

    def _save_model(self):
        if self.trainer:
            self.trainer.save_model()
            self._log_message("[模型] 已保存")

    def _load_data(self):
        if self.trainer:
            f = filedialog.askopenfilename(initialdir=self.data_dir_var.get())
            if f:
                self.trainer.load_training_data(f)
                self._log_message(f"[数据] 已加载 {f}")

    def _start_eval(self):
        """一次性评估 N 局（独立进程），对局实时回传 GUI 回放；无循环、无日志文件。
        评估与训练互斥：训练中不可评估，评估中不可开始训练。"""
        if self.is_training:
            self._log_message("训练进行中，请先停止训练再评估")
            return
        if getattr(self, '_eval_proc', None) is not None and self._eval_proc.poll() is None:
            self._log_message("已有评估在运行，请等它结束")
            return
        n = self.eval_games_var.get()
        cmd = [sys.executable, '-u', self._eval_script(),
               '--games', str(n),
               '--sims', str(self.simulations_var.get()),
               '--data-dir', self.data_dir_var.get() or 'data/',
               '--device', self.device_var.get()]
        try:
            self._eval_proc = subprocess.Popen(
                cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self._log_message("开始评估 %d 局: %s" % (n, ' '.join(cmd)))
            threading.Thread(target=self._eval_stdout_reader,
                             args=(self._eval_proc,), daemon=True).start()
            threading.Thread(target=self._eval_stderr_reader,
                             args=(self._eval_proc,), daemon=True).start()
        except Exception as e:
            self._log_message("启动失败: %s" % e)

    def _eval_stdout_reader(self, proc):
        """读取评估对局帧（4字节长度 + pickle 局数据）→ 送入对局回放。"""
        game_count = 0
        try:
            while True:
                header = proc.stdout.read(4)
                if not header:
                    break
                size = struct.unpack('<I', header)[0]
                payload = proc.stdout.read(size)
                if len(payload) != size:
                    break
                data = pickle.loads(payload)
                _, _, _, moves, winner, score_diff, _ = data
                game_count += 1
                self.update_queue.put({
                    'type': 'game',
                    'game_result': GameResult(game_count, moves, winner),
                    'game_count': game_count,
                    'score_diff': score_diff,
                    'loss': (None, None, None, None, None)
                })
        except Exception as e:
            self.update_queue.put({'type': 'log', 'message': "回放读取异常: %s" % e})
        finally:
            rc = proc.wait()
            if rc != 0:
                self.update_queue.put({'type': 'log',
                                       'message': "异常退出 code=%d（完成 %d 局）" % (rc, game_count)})
            else:
                self.update_queue.put({'type': 'log',
                                       'message': "评估结束（%d 局）" % game_count})
            self._eval_proc = None

    def _eval_stderr_reader(self, proc):
        """评估进度（stderr）→ 日志区。按 \r/\n 都切分，避免构建进度裸 \r 刷屏。"""
        buf = b''
        try:
            while True:
                chunk = proc.stderr.read(4096)
                if not chunk:
                    break
                buf += chunk
                while True:
                    nl = -1
                    for i in range(len(buf)):
                        if buf[i] in (10, 13):          # \n 或 \r
                            nl = i
                            break
                    if nl == -1:
                        break
                    line = buf[:nl]
                    buf = buf[nl + 1:]
                    text = line.decode('utf-8', errors='replace').strip()
                    text = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', text).strip()
                    if text:
                        self.update_queue.put({'type': 'log', 'message': text})
            tail = buf.decode('utf-8', errors='replace').strip()
            tail = re.sub(r'\x1b\[[0-9;]*[A-Za-z]', '', tail).strip()
            if tail:
                self.update_queue.put({'type': 'log', 'message': tail})
        except Exception:
            pass

    def _on_close(self):
        """关闭 GUI 时顺带结束评估子进程，避免留下孤儿进程。"""
        proc = getattr(self, '_eval_proc', None)
        if proc is not None and proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self.root.destroy()

    def _eval_script(self):
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval.py')

    def _log_message(self, msg):
        self.status_text.insert(tk.END, msg + "\n")
        self.status_text.see(tk.END)

    def _update_stats(self):
        if self.trainer:
            self.stats_panel.update_stats(
                len(self.trainer.data_buffer),
                self.trainer.game_count,
                self.trainer.train_count,
                self.trainer.optimizer.param_groups[0]['lr']
            )

    def _process_queue(self):
        try:
            while not self.update_queue.empty():
                msg = self.update_queue.get_nowait()
                if msg['type'] == 'game':
                        game_result = msg['game_result']
                        self.game_history.append(game_result)
                        if msg['loss'][0] is not None:
                              self.stats_panel.update_loss(*msg['loss'])
                        sd = msg.get('score_diff')
                        if game_result.winner == 0:
                            result_text = "平局"
                        elif sd is None:
                            result_text = "黑胜" if game_result.winner == 1 else "白胜"
                        else:
                            pts = '%.1f' % abs(sd)
                            if pts.endswith('.0'):
                                pts = pts[:-2]
                            result_text = ("黑胜%s目" if game_result.winner == 1 else "白胜%s目") % pts
                        self._log_message(f"[游戏 {msg['game_count']}] {result_text} | 手数: {len(game_result.moves)}")
                        if self.current_history_index == -1 and self.game_history:
                              self.current_history_index = 0
                              self._display_game(self.game_history[0])
                elif msg['type'] == 'train_progress':
                    p, v, e, w, t = msg['loss']
                    self.stats_panel.update_loss(p, v, e, w, t)
                    self._log_message(f"[训练进度] Batch {msg['batch']} | 损失均值 p={p:.4f} v={v:.4f} e={e:.4f} w={w:.4f} t={t:.4f}")
                elif msg['type'] == 'stats':
                    self._update_stats()
                elif msg['type'] == 'log':
                    self._log_message(msg['message'])
                elif msg['type'] == 'error':
                    self._log_message(f"[错误] {msg['message']}")
                elif msg['type'] == 'finished':
                    if 'batches' in msg:
                        self._log_message(f"[完成] 仅训练停止，共 {msg['batches']} 个Batch")
                    else:
                        self._log_message("[完成] 训练结束")
                    self.is_training = False
                    self.start_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
            self._update_stats()
        except Exception as e:
            self._log_message(f"[GUI处理错误] {e}")
        finally:
            self.root.after(100, self._process_queue)


if __name__ == "__main__":
    mp.freeze_support()
    app = TrainingGUI()
