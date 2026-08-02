import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
import queue
import multiprocessing as mp
import os
from datetime import datetime
from train import SelfPlayTrainer, play_one_game
from ui_utils import GameBoard, TrainingStatsPanel
from model import PolicyValueNet
import numpy as np
# 超参数
BOARD_SIZE = 13
ONNX_PATH = 'model.onnx'
MODEL_PATH = 'model.pt'
DATA_DIR = 'data/'
NUM_SIMULATIONS = 200
C_PUCT = 1.5
TEMPERATURE = 0.8
EXPLORATION_MODE = False
BATCH_SIZE = 384
SAVE_INTERVAL = 10
TRAIN_EPOCHS = 10

def worker_process(trainer_params, model_path, result_queue, stop_event, device='cuda'):
    onnx_path = ONNX_PATH
    if not os.path.exists(onnx_path):
        print(f"[Worker] ONNX文件不存在: {onnx_path}")
        return
    board_size = trainer_params['board_size']
    num_simulations = trainer_params['num_simulations']
    c_puct = trainer_params['c_puct']
    temperature = trainer_params['temperature']
    exploration = trainer_params['exploration_mode']
    np.random.seed()
    while not stop_event.is_set():
        states, policies, players, moves, winner,score_diff,ownership = play_one_game(
            board_size=board_size,
            num_simulations=num_simulations,
            device=device,
            c_puct=c_puct,
            temperature=temperature,
            exploration=exploration,
            onnx_path=onnx_path
        )
        result_queue.put((states, policies, players, moves, winner,score_diff,ownership))

class GameResult:
    def __init__(self, game_id, moves, winner):
        self.game_id = game_id
        self.moves = moves 
        self.winner = winner

class TrainingGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("围棋AI训练")
        self.root.geometry("1500x900")
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
            if os.path.exists(MODEL_PATH):
                model = PolicyValueNet.load_model(MODEL_PATH, device='cuda')
            else:
                model = PolicyValueNet()
                model.save_model(MODEL_PATH)
            model.export_onnx(ONNX_PATH)

        self._create_widgets()
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
        self.total_var = tk.IntVar(value=1000)
        ttk.Spinbox(frame, from_=10, to=10000, textvariable=self.total_var, width=15).grid(row=1, column=1, padx=5)
        ttk.Label(frame, text="进程数:").grid(row=2, column=0, sticky=tk.W)
        self.threads_var = tk.IntVar(value=2)
        ttk.Spinbox(frame, from_=1, to=16, textvariable=self.threads_var, width=10).grid(row=2, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="批大小:").grid(row=3, column=0, sticky=tk.W)
        self.batch_size_var = tk.IntVar(value=BATCH_SIZE)
        ttk.Spinbox(frame, from_=32, to=4096, textvariable=self.batch_size_var, width=10).grid(row=3, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="保存间隔(局):").grid(row=4, column=0, sticky=tk.W)
        self.save_interval_var = tk.IntVar(value=SAVE_INTERVAL)
        ttk.Spinbox(frame, from_=1, to=20, textvariable=self.save_interval_var, width=10).grid(row=4, column=1, padx=5, sticky=tk.W)
        ttk.Label(frame, text="训练轮数(仅训练):").grid(row=5, column=0, sticky=tk.W)
        self.train_epochs_var = tk.IntVar(value=TRAIN_EPOCHS)
        ttk.Spinbox(frame, from_=1, to=1000, textvariable=self.train_epochs_var, width=15).grid(row=5, column=1, padx=5)
        self.exploration_var = tk.BooleanVar(value=EXPLORATION_MODE)
        ttk.Checkbutton(frame, text="探索模式", variable=self.exploration_var).grid(row=6, column=0, columnspan=2, sticky=tk.W)

    def _create_optimizer_settings(self, parent):
        frame = ttk.LabelFrame(parent, text="优化器", padding="10")
        frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(frame, text="学习率:").grid(row=0, column=0, sticky=tk.W)
        self.lr_var = tk.DoubleVar(value=0.0001)
        ttk.Scale(frame, from_=0.0, to=0.0005, variable=self.lr_var, orient=tk.HORIZONTAL, length=150).grid(row=0, column=1, padx=5)
        self.lr_label = ttk.Label(frame, text="0.0001")
        self.lr_label.grid(row=0, column=2)
        self.lr_var.trace('w', lambda *a: self.lr_label.configure(text=f"{self.lr_var.get():.6f}"))
        ttk.Label(frame, text="权重衰减:").grid(row=1, column=0, sticky=tk.W)
        self.wd_var = tk.DoubleVar(value=0.0001)
        ttk.Scale(frame, from_=0.0, to=0.001, variable=self.wd_var, orient=tk.HORIZONTAL, length=150).grid(row=1, column=1, padx=5)
        self.wd_label = ttk.Label(frame, text="0.0001")
        self.wd_label.grid(row=1, column=2)
        self.wd_var.trace('w', lambda *a: self.wd_label.configure(text=f"{self.wd_var.get():.6f}"))

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
        self.start_btn = ttk.Button(frame, text="开始训练", command=self._start_training)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(frame, text="停止训练", command=self._stop_training, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        self.apply_btn = ttk.Button(frame, text="应用超参数", command=self._apply_hyperparams)
        self.apply_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="保存模型", command=self._save_model).pack(side=tk.LEFT, padx=5)
        ttk.Button(frame, text="加载数据", command=self._load_data).pack(side=tk.LEFT, padx=5)

    def _create_board_area(self, parent):
        frame = ttk.LabelFrame(parent, text="对局回放", padding="10")
        frame.pack(fill=tk.BOTH, expand=True)
        self.board_size = BOARD_SIZE
        self.cell_size = 35
        self.margin = 35
        self.board_width = (self.board_size - 1) * self.cell_size
        self.canvas_size = self.board_width + 2 * self.margin
        self.canvas = tk.Canvas(frame, width=self.canvas_size, height=self.canvas_size,
                                bg='#DCB35C', highlightthickness=0)
        self.canvas.pack(pady=10)
        self.game_board = GameBoard(self.canvas, self.board_size, self.cell_size, self.margin)

        control = ttk.Frame(frame)
        control.pack(pady=5)
        ttk.Button(control, text="◀ 上一局", command=self._prev_game, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="下一局 ▶", command=self._next_game, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Separator(control, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5)
        ttk.Button(control, text="◀ 上一步", command=self._prev_step, width=8).pack(side=tk.LEFT, padx=5)
        ttk.Button(control, text="下一步 ▶", command=self._next_step, width=8).pack(side=tk.LEFT, padx=5)
        self.step_label = ttk.Label(control, text="步数: 0/0", font=('微软雅黑', 9))
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

    def _apply_hyperparams(self):
        if self.is_training:
            messagebox.showwarning("警告", "训练中请先停止")
            return
        if self.trainer:
            self.trainer.update_hyperparameters(
                learning_rate=self.lr_var.get(),
                weight_decay=self.wd_var.get(),
                c_puct=self.c_puct_var.get(),
                num_simulations=self.simulations_var.get()
            )
            self._log_message("[超参数] 已更新")

    def _start_training(self):
        if self.is_training:
            return
        self.trainer = SelfPlayTrainer(
            model_path=self.model_path_var.get(),
            board_size=BOARD_SIZE,
            device=self.device_var.get(),
            num_simulations=self.simulations_var.get(),
            data_dir=self.data_dir_var.get(),
            c_puct=self.c_puct_var.get(),
            learning_rate=self.lr_var.get(),
            weight_decay=self.wd_var.get()
        )
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
                args=(self.total_var.get(), self.batch_size_var.get(),
                      self.save_interval_var.get(), self.load_data_var.get(),
                      self.train_epochs_var.get()),
                daemon=True
            )
            thread.start()
        else:
            params = {
                'board_size': BOARD_SIZE,
                'num_simulations': self.simulations_var.get(),
                'c_puct': self.c_puct_var.get(),
                'temperature': self.temperature_var.get(),
                'exploration_mode': self.exploration_var.get()
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
                      self.save_interval_var.get(), self.load_data_var.get()),
                daemon=True
            )
            thread.start()

    def _train_only_loop(self, total_epochs, batch_size, save_interval, load_data, train_epochs):
        try:
            if load_data:
                self.trainer.load_training_data()
            if len(self.trainer.data_buffer) < batch_size:
                self._log_message(f"[错误] 数据不足 ({len(self.trainer.data_buffer)} < {batch_size})")
                self.update_queue.put({'type': 'finished'})
                return
            for epoch in range(total_epochs):
                if not self.is_training:
                    break
                for step in range(train_epochs):
                    if not self.is_training:
                        break
                    loss = self.trainer.train_step(batch_size)
                    if loss[0] is not None:
                        self.trainer.train_count += 1
                        if step % 10 == 0:
                            self.update_queue.put({
                                'type': 'train_progress',
                                'loss': loss,
                                'epoch': epoch + 1,
                                'step': step + 1,
                                'total_steps': train_epochs
                            })
                if (epoch + 1) % save_interval == 0:
                    self.trainer.save_model()
                    self.update_queue.put({'type': 'log', 'message': f"[模型] 已保存 (轮次 {epoch+1})"})
            self.trainer.save_model()
            self.update_queue.put({'type': 'finished'})
        except Exception as e:
            self.update_queue.put({'type': 'error', 'message': str(e)})

    def _training_loop(self, result_queue, total_games, batch_size, save_interval, load_data):
        try:
            if load_data:
                self.trainer.load_training_data()
            games = 0
            while games < total_games and self.is_training:
                try:
                    states, policies, players, moves, winner,score_diff,ownership = result_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                games += 1
                self.trainer.data_buffer.add_game(states, policies, players, winner,score_diff,ownership)
                self.trainer.game_count += 1
                self.trainer.save_training_data((states, policies, players, winner,score_diff,ownership))
                loss = self.trainer.train_step(batch_size)
                if loss[0] is not None:
                    self.trainer.train_count += 1

                game_result = GameResult(games, moves, winner)
                self.update_queue.put({
                    'type': 'game',
                    'game_result': game_result,
                    'game_count': games,
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
                        winner_text = "黑胜" if game_result.winner == 1 else "白胜" if game_result.winner == -1 else "平局"
                        self._log_message(f"[游戏 {msg['game_count']}] {winner_text} | 手数: {len(game_result.moves)}")
                        if self.current_history_index == -1 and self.game_history:
                              self.current_history_index = 0
                              self._display_game(self.game_history[0])
                elif msg['type'] == 'train_progress':
                    p, v, e, t = msg['loss']
                    self.stats_panel.update_loss(p, v, e, t)
                    self._log_message(f"[训练进度] 轮次 {msg['epoch']}/{msg['total_steps']} | 步骤 {msg['step']}")
                elif msg['type'] == 'stats':
                    self._update_stats()
                elif msg['type'] == 'log':
                    self._log_message(msg['message'])
                elif msg['type'] == 'error':
                    self._log_message(f"[错误] {msg['message']}")
                elif msg['type'] == 'finished':
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