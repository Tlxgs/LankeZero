"""
GUI工具模块 - 提供通用的UI组件和绘制函数
"""
import tkinter as tk
import numpy as np

from hyperparams import HOLE


class GameBoard:
    def __init__(self, canvas, board_size, cell_size, margin, hole_mask=None):
        self.canvas = canvas
        self.board_size = int(board_size)
        self.cell_size = cell_size
        self.margin = margin
        self.board_width = (board_size - 1) * cell_size
        self.set_hole(hole_mask)

    def set_hole(self, hole_mask):
        """棋盘形状掩码（True = 洞）；None 表示整盘。"""
        n = self.board_size
        self.hole_mask = (np.zeros((n, n), dtype=bool) if hole_mask is None
                          else np.asarray(hole_mask, dtype=bool).reshape(n, n))

    def is_hole(self, row, col):
        n = self.board_size
        if not (0 <= row < n and 0 <= col < n):
            return False
        return bool(self.hole_mask[row, col])

    def draw_grid(self):
        for i in range(self.board_size):
            y = self.margin + i * self.cell_size
            self.canvas.create_line(self.margin, y, self.margin + self.board_width, y,
                                    fill='black', width=1)
            x = self.margin + i * self.cell_size
            self.canvas.create_line(x, self.margin, x, self.margin + self.board_width,
                                    fill='black', width=1)
        # 网格先画完，再用背景色盖住洞；洞边那一圈网格线会保留下来成为内边界。
        self.draw_hole()

    def draw_hole(self):
        """按格涂灰：心形、圆环、多个洞、贴边缺口都是同一段循环。"""
        if not self.hole_mask.any():
            return
        half = 0.5 * self.cell_size
        for r, c in np.argwhere(self.hole_mask):
            x = self.margin + int(c) * self.cell_size
            y = self.margin + int(r) * self.cell_size
            self.canvas.create_rectangle(x - half, y - half, x + half, y + half,
                                         fill='#A9A9A9', outline='')

    def draw_stars(self):
        """按棋盘大小生成星位；7/9/11 用二线，13+ 用三线。

        偶数路没有天元（正中心根本没有交叉点），也不画边星 —— 只画四个角。
        否则 (n-1)//2 会画出偏半格的一个点：既不是天元，也不在星位上。
        """
        n = self.board_size
        if n < 7:
            return
        off = 2 if n <= 11 else 3
        pts = [(off, off), (off, n - 1 - off),
               (n - 1 - off, off), (n - 1 - off, n - 1 - off)]
        if n % 2 == 1:
            mid = (n - 1) // 2
            pts.append((mid, mid))
            if n >= 15:
                pts.extend([(off, mid), (mid, off),
                            (n - 1 - off, mid), (mid, n - 1 - off)])
        for r, c in dict.fromkeys(pts):
            if self.is_hole(r, c):
                continue
            x = self.margin + c * self.cell_size
            y = self.margin + r * self.cell_size
            self.canvas.create_oval(x-4, y-4, x+4, y+4, fill='black', outline='black')

    def draw_piece(self, row, col, piece):
        if piece == 0 or piece == HOLE or self.is_hole(row, col):
            return None
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2
        color = 'black' if piece == 1 else 'white'
        return self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                       fill=color, outline='gray')

    def highlight_move(self, row, col):
        if self.is_hole(row, col):
            return None
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        return self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, fill='red', outline='red')

    def draw_stat_circle(self, row, col, value, text):
        """在交叉点画统计圆；value 必须是已归一化到 0..1 的量。

        归一化由调用方做：胜率本身就是 0..1，策略先验要先除以当前盘面最大先验
        （否则 0.05~0.3 的先验全挤在色带同一端，棋盘上看不出热点）。
        以前这里收了个 max_value 参数却从没用过，两种模式共用一套绝对映射。
        """
        if self.is_hole(row, col):
            return None
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2.2
        norm_value = float(np.nan_to_num(value, nan=0.0))
        norm_value = max(0.0, min(1.0, norm_value))
        if norm_value <= 0.5:
            r = 255
            g = int(100 + 155 * (norm_value / 0.5))
            b = 100
        else:
            r = int(255 - 155 * ((norm_value - 0.5) / 0.5))
            g = 255
            b = 100
        color = f'#{r:02x}{g:02x}{b:02x}'
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                fill=color, outline='')
        self.canvas.create_text(x, y, text=text, font=('Arial', 10, 'bold'),
                                fill='black', justify='center')

    def draw_best_circle(self, row, col):
        if self.is_hole(row, col):
            return None
        x = self.margin + col * self.cell_size
        y = self.margin + row * self.cell_size
        radius = self.cell_size // 2 - 2 + 4
        self.canvas.create_oval(x - radius, y - radius, x + radius, y + radius,
                                outline='green', width=4, fill='')

    def coord_to_index(self, x, y):
        col = round((x - self.margin) / self.cell_size)
        row = round((y - self.margin) / self.cell_size)
        if (0 <= row < self.board_size and 0 <= col < self.board_size
                and not self.is_hole(row, col)):
            return (row, col)
        return None


class TrainingStatsPanel:
    def __init__(self, parent):
        self.parent = parent
        self.stats_label = None
        self.policy_loss_label = None
        self.own_loss_label = None
        self.win_loss_label = None
        self.total_loss_label = None
        self.lr_display_label = None

    def create(self):
        stats_frame = tk.Frame(self.parent)
        stats_frame.pack(fill=tk.X, pady=5)
        self.stats_label = tk.Label(stats_frame, text="等待开始...", font=('微软雅黑', 12))
        self.stats_label.pack(side=tk.LEFT)

        loss_frame = tk.Frame(self.parent)
        loss_frame.pack(fill=tk.X, pady=5)
        
        # 第一行：策略损失 + 领地损失（后者出自 own 头的逐点归属）
        row1 = tk.Frame(loss_frame)
        row1.pack(fill=tk.X)
        self.policy_loss_label = tk.Label(row1, text="平均策略损失: --", font=('微软雅黑', 12))
        self.policy_loss_label.pack(side=tk.LEFT, padx=5)
        self.own_loss_label = tk.Label(row1, text="平均领地损失: --", font=('微软雅黑', 12))
        self.own_loss_label.pack(side=tk.LEFT, padx=5)
        
        # 第二行：胜率损失 + 分数损失 + 总损失
        # 熵约束关闭后不再显示「熵损失」——它显示的是网络输出熵 H(q)，只有开正则时才有判断价值。
        row2 = tk.Frame(loss_frame)
        row2.pack(fill=tk.X)
        self.win_loss_label = tk.Label(row2, text="平均胜率损失: --", font=('微软雅黑', 12))
        self.win_loss_label.pack(side=tk.LEFT, padx=5)
        self.total_loss_label = tk.Label(row2, text="平均总损失: --", font=('微软雅黑', 12))
        self.total_loss_label.pack(side=tk.LEFT, padx=5)
        
        # 第三行：学习率（单独一行）
        row3 = tk.Frame(loss_frame)
        row3.pack(fill=tk.X)
        self.lr_display_label = tk.Label(row3, text="学习率: --", font=('微软雅黑', 12))
        self.lr_display_label.pack(side=tk.LEFT, padx=5)

    def update_stats(self, data_size, game_count, train_count, lr):
        self.stats_label.config(text=f"数据量: {data_size} 条 | 对局数: {game_count} | 训练Batch: {train_count}")
        self.lr_display_label.config(text=f"学习率: {lr:.6f}")

    def update_loss(self, policy_loss, own_loss, entropy_loss, win_loss,
                    total_loss):
        """形参顺序即 SelfPlayTrainer.train_step 的返回顺序（分数头已删除）。
        entropy_loss 仍收下但不显示 —— 熵约束关闭后 H(q) 没有判断对象；
        要开正则时在这里加回一行即可。"""
        self.policy_loss_label.config(text=f"平均策略损失: {policy_loss:.4f}")
        self.own_loss_label.config(text=f"平均领地损失: {own_loss:.4f}")
        self.win_loss_label.config(text=f"平均胜率损失: {win_loss:.4f}")
        self.total_loss_label.config(text=f"平均总损失: {total_loss:.4f}")
