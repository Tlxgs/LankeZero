"""
GUI工具模块 - 提供通用的UI组件和绘制函数
"""
import tkinter as tk
from tkinter import ttk
import numpy as np


class GameBoard:
    def __init__(self, canvas, board_size, cell_size, margin):
        self.canvas = canvas
        self.board_size = board_size
        self.cell_size = cell_size
        self.margin = margin
        self.board_width = (board_size - 1) * cell_size

    def draw_grid(self):
        for i in range(self.board_size):
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y, self.margin + self.board_width, y,
                                    fill='black', width=1)
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.margin + self.board_width,
                                    fill='black', width=1)

    def draw_stars(self):
        # 9路星位：天元 (4,4) 及四隅 (2,2),(6,2),(2,6),(6,6)
        if self.board_size == 9:
            positions = [(2,2), (6,2), (2,6), (6,6), (4,4)]
        elif self.board_size == 13:
            positions = [(3,3), (9,3), (6,6), (3,9), (9,9)]
        else:
            positions = []
        for r, c in positions:
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='black', outline='black')

    def draw_piece(self, row, col, piece):
        if piece == 0:
            return
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2
        color = 'black' if piece == 1 else 'white'
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline='gray')

    def highlight_move(self, row, col):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill='red', outline='red')

    def draw_stat_circle(self, row, col, value, max_value, text):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2.5
        norm_value = max(0.0, min(1.0, value))
        if norm_value <= 0.5:
            r = 255
            g = int(100 + 75 * (norm_value / 0.5))
            b = 100
        else:
            r = int(255 - 75 * ((norm_value - 0.5) / 0.5))
            g = 255
            b = 100
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline='')
        self.canvas.create_text(x, y, text=text, font=('Arial', 9, 'bold'),
                                fill='black', justify='center')

    def draw_best_circle(self, row, col):
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2 + 4
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                outline='green', width=2, fill='')

    def coord_to_index(self, x, y):
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        if 0 <= row < self.board_size and 0 <= col < self.board_size:
            return (row, col)
        return None


class TrainingStatsPanel:
    def __init__(self, parent):
        self.parent = parent
        self.stats_label = None
        self.current_game_label = None
        self.policy_loss_label = None
        self.value_loss_label = None
        self.entropy_loss_label = None
        self.total_loss_label = None
        self.lr_display_label = None

    def create(self):
        stats_frame = tk.Frame(self.parent)
        stats_frame.pack(fill=tk.X, pady=5)
        self.stats_label = tk.Label(stats_frame, text="等待开始...", font=('微软雅黑', 10))
        self.stats_label.pack(side=tk.LEFT)
        self.current_game_label = tk.Label(stats_frame, text="", font=('微软雅黑', 10), foreground='blue')
        self.current_game_label.pack(side=tk.RIGHT)

        loss_frame = tk.Frame(self.parent)
        loss_frame.pack(fill=tk.X, pady=5)
        self.policy_loss_label = tk.Label(loss_frame, text="策略损失: --", font=('微软雅黑', 9))
        self.policy_loss_label.pack(side=tk.LEFT, padx=5)
        self.value_loss_label = tk.Label(loss_frame, text="价值损失: --", font=('微软雅黑', 9))
        self.value_loss_label.pack(side=tk.LEFT, padx=5)
        self.entropy_loss_label = tk.Label(loss_frame, text="熵损失: --", font=('微软雅黑', 9))
        self.entropy_loss_label.pack(side=tk.LEFT, padx=5)
        self.total_loss_label = tk.Label(loss_frame, text="总损失: --", font=('微软雅黑', 9))
        self.total_loss_label.pack(side=tk.LEFT, padx=5)
        self.lr_display_label = tk.Label(loss_frame, text="学习率: --", font=('微软雅黑', 9))
        self.lr_display_label.pack(side=tk.RIGHT, padx=5)

    def update_stats(self, data_size, game_count, train_count, lr):
        self.stats_label.config(text=f"数据量: {data_size} 条 | 对局数: {game_count} | 训练步数: {train_count}")
        self.lr_display_label.config(text=f"学习率: {lr:.6f}")

    def update_loss(self, policy_loss, value_loss, entropy_loss, total_loss):
        self.policy_loss_label.config(text=f"策略损失: {policy_loss:.4f}")
        self.value_loss_label.config(text=f"价值损失: {value_loss:.4f}")
        self.entropy_loss_label.config(text=f"熵损失: {entropy_loss:.4f}")
        self.total_loss_label.config(text=f"总损失: {total_loss:.4f}")

    def update_current_game(self, game_id):
        self.current_game_label.config(text=f"当前: #{game_id}")