import tkinter as tk
from tkinter import ttk, messagebox
import math
import numpy as np

from game import GoGame, PASS_MOVE, SCALE
from mcts import MCTS
from ui_utils import GameBoard

# ==================== 超参数 ====================
BOARD_SIZE = 13
ONNX_PATH = 'model.pt'
NUM_SIMULATIONS = 400
C_PUCT = 1.5
TEMPERATURE = 0.0
BATCH = 16
STEP_MS = 10
DEBUG = True
KOMI = 7.5
# ===============================================


class GomokuGUI:
    def __init__(self, device='cuda'):
        self.board_size = BOARD_SIZE
        self.komi = KOMI
        self.game = GoGame(board_size=BOARD_SIZE, komi=self.komi)
        self.mcts = MCTS(c_puct=C_PUCT, num_simulations=NUM_SIMULATIONS,
                         temperature=TEMPERATURE, onnx_path=ONNX_PATH,
                         device=device, board_size=BOARD_SIZE,dirichlet_epsilon=0)

        self.cell_size, self.margin = 50, 50
        self.canvas_size = (self.board_size - 1) * self.cell_size + 2 * self.margin

        self.root = tk.Tk()
        self.root.title("围棋 AI - 13路")
        self.root.resizable(False, False)

        # 左侧控制面板
        ctrl = ttk.LabelFrame(self.root, text="设置", padding=8)
        ctrl.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        self.sim_var = tk.IntVar(value=NUM_SIMULATIONS)
        self.cpuct_var = tk.DoubleVar(value=C_PUCT)
        self.temp_var = tk.DoubleVar(value=TEMPERATURE)
        self.color_var = tk.StringVar(value="黑棋 (先手)")
        self.mode_var = tk.StringVar(value="对弈棋盘")

        params = [
            ("模拟次数:", self.sim_var, 10, 500, 1),
            ("c_puct:", self.cpuct_var, 0.0, 10.0, 0.1),
            ("温度:", self.temp_var, 0.0, 2.0, 0.1),
        ]

        for i, (txt, var, lo, hi, inc) in enumerate(params):
            ttk.Label(ctrl, text=txt).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Spinbox(ctrl, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=8).grid(row=i, column=1, pady=3)

        row = len(params)  # 现有参数行数
        ttk.Label(ctrl, text="贴目:").grid(row=row, column=0, sticky="w", pady=3)
        self.komi_var = tk.DoubleVar(value=KOMI)
        ttk.Spinbox(ctrl, from_=-21, to=21.0, increment=0.5,
                    textvariable=self.komi_var, width=8).grid(row=row, column=1, pady=3)

        row += 1
        ttk.Label(ctrl, text="执子:").grid(row=row, column=0, sticky="w", pady=3)
        ttk.Combobox(ctrl, textvariable=self.color_var,
                     values=["黑棋 (先手)", "白棋 (后手)"],
                     state="readonly", width=12).grid(row=row, column=1, pady=3)

        row += 1
        ttk.Label(ctrl, text="显示:").grid(row=row, column=0, sticky="w", pady=3)
        mode_cb = ttk.Combobox(ctrl, textvariable=self.mode_var,
                               values=["对弈棋盘", "MCTS统计", "策略网络概率", "领地归属"],
                               state="readonly", width=12)
        mode_cb.grid(row=row, column=1, pady=3)
        mode_cb.bind("<<ComboboxSelected>>", lambda _: self._draw())

        row += 1
        ttk.Button(ctrl, text="应用参数", command=self._apply_params) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(ctrl, text="重置游戏", command=self._reset) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(ctrl, text="弃权 (Pass)", command=self._human_pass) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        row += 1
        ttk.Button(ctrl, text="退出", command=self.root.quit) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)

        # 胜率条
        row += 1
        ttk.Separator(ctrl, orient="horizontal") \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=6)
        row += 1
        wr_frame = ttk.LabelFrame(ctrl, text="综合估值", padding=5)
        wr_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.win_canvas = tk.Canvas(wr_frame, width=200, height=30,
                                    bg='white', highlightthickness=0)
        self.win_canvas.pack(pady=(0, 2))
        self.win_label = ttk.Label(wr_frame, text="",
                                   font=('微软雅黑', 10))
        self.win_label.pack()

        # 目差显示
        row += 1
        score_frame = ttk.LabelFrame(ctrl, text="启发式目差估值", padding=5)
        score_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.score_label = ttk.Label(score_frame, text="",
                                     font=('微软雅黑', 10))
        self.score_label.pack()

        # === Value Net 估值显示 ===
        row += 1
        value_frame = ttk.LabelFrame(ctrl, text="价值网络估值", padding=5)
        value_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.value_net_label = ttk.Label(value_frame, text="0.000", font=('微软雅黑', 10))
        self.value_net_label.pack()

        # === 领地目差显示 (Value Net 归属头) ===
        row += 1
        territory_frame = ttk.LabelFrame(ctrl, text="领地网络估值", padding=5)
        territory_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.territory_label = ttk.Label(territory_frame, text="",
                                         font=('微软雅黑', 10))
        self.territory_label.pack()

        # 棋盘
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size,
                                bg='#DCB35C', highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=5, pady=10)
        self.board = GameBoard(self.canvas, self.board_size, self.cell_size, self.margin)
        self.canvas.bind("<Button-1>", self._on_click)

        # 搜索状态（单线程，无锁）
        self._search_gen = 0
        self._sim_count = 0

        # === 当前 Value Net 输出缓存 ===
        self.current_value = None
        # === 当前领地归属头输出缓存 ===
        self.current_ownership = None

        self.black_weighted = 0
        self.white_weighted = 0

        self._init_mcts()
        self._update_score_diff()
        self._update_current_value()   # 初始估值 + 归属
        self._draw()
        self._start_search()
        self.root.mainloop()

    # ---------------- 初始化 / 判断 ----------------

    def _init_mcts(self):
        self.mcts.root = None
        self.mcts.init_root(self.game)
        self._update_current_value()   # 根节点初始化后更新估值

    def _is_human_turn(self):
        return (self.game.current_player == 1) == (self.color_var.get() == "黑棋 (先手)")

    # ---------------- 绘制 ----------------

    def _draw(self):
        root = self.mcts.root
        children = root.children if root else {}
        self.canvas.delete("all")
        self.board.draw_grid()
        self.board.draw_stars()
        mode = self.mode_var.get()
        if mode == "对弈棋盘":
            self._draw_pieces()
        elif mode == "领地归属":
            self._draw_ownership()
        else:
            self._draw_stats(mode == "策略网络概率", children)
        self._draw_pass_info(children)
        self._draw_winrate()
        # === 更新 Value Net 显示 ===
        if self.current_value is not None:
            value = SCALE * math.atanh(self.current_value) * self.game.current_player
            self.value_net_label.config(text=f"黑棋: {value:+.2f}")
        else:
            self.value_net_label.config(text="--")

        bw = self.black_weighted if hasattr(self, 'black_weighted') else 0
        ww = self.white_weighted if hasattr(self, 'white_weighted') else 0
        self.territory_label.config(
            text=f"黑棋: {bw-ww:+.2f}"
        )

    def _draw_ownership(self):
        """可视化领地归属：黑白灰小正方形叠加在棋子上方"""
        own = self.current_ownership
        if own is None:
            return

        # 1. 先绘制当前棋盘上的棋子（活子/死子保持原样）
        self._draw_pieces()

        # 2. 在每格中心画小正方形，颜色由 ownership 值映射为灰度
        for r in range(self.board_size):
            for c in range(self.board_size):
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                d = float(own[r, c])                     # ∈[-1, 1]
                # 将 d 映射到灰度 0(黑) ~ 255(白)
                gray = int(128 - 127 * d)                # d=1 -> 0, d=0 -> 128, d=-1 -> 255
                gray = max(0, min(255, gray))
                color = f'#{gray:02x}{gray:02x}{gray:02x}'
                # 正方形边长约为棋子直径的 60%
                size = int(self.cell_size * 0.5)
                self.canvas.create_rectangle(
                    x - size//2, y - size//2,
                    x + size//2, y + size//2,
                    fill=color, outline='', width=0
                )


    def _draw_pieces(self):
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.game.board[r, c]
                if p:
                    self.board.draw_piece(r, c, p)
        if self.game.last_move and self.game.last_move != PASS_MOVE:
            self.board.highlight_move(*self.game.last_move)

    def _draw_stats(self, is_policy, children):
        if not children:
            return
        for r in range(self.board_size):
            for c in range(self.board_size):
                p = self.game.board[r, c]
                if p:
                    x = self.margin + c * self.cell_size
                    y = self.margin + r * self.cell_size
                    rad = self.cell_size // 2 - 2
                    self.canvas.create_oval(x - rad, y - rad, x + rad, y + rad,
                                            fill='gray' if p == 1 else 'lightgray',
                                            outline='gray', stipple='gray50')
        stats = []
        for m, ch in children.items():
            if m == PASS_MOVE:
                continue
            r, c = m
            if self.game.board[r, c] != 0:
                continue
            vis = ch.visit_count
            wr = (1 - ch.get_value()) / 2 if vis > 0 else 0.5
            stats.append((m, vis, wr, ch.prior_prob))
        if not stats:
            return
        max_val = max(s[3] if is_policy else s[2] for s in stats)
        best = max(stats, key=lambda s: s[3] if is_policy else s[1])[0]
        for m, vis, wr, prob in stats:
            if is_policy:
                self.board.draw_stat_circle(m[0], m[1], prob, max_val,
                                            f"{prob*100:.1f}%")
            elif vis:
                self.board.draw_stat_circle(m[0], m[1], wr, max_val,
                                            f"{vis}\n{wr*100:.0f}%")
        self.board.draw_best_circle(*best)

    def _draw_pass_info(self, children):
        """在棋盘下方显示 Pass 的先验概率与搜索次数"""
        if PASS_MOVE in children:
            ch = children[PASS_MOVE]
            info = (f"Pass: 先验 {ch.prior_prob*100:.1f}%"
                    f"  |  搜索 {ch.visit_count}次")
        else:
            info = "Pass: --"
        y = self.canvas_size - 6
        self.canvas.create_text(self.margin, y, text=info, anchor='sw',
                                font=('微软雅黑', 12), fill='#333333')

    def _draw_winrate(self):
        self.win_canvas.delete("all")
        root = self.mcts.root
        if not root or not root.children:
            self.win_label.config(text="黑棋")
            return
        total = sum(c.visit_count for c in root.children.values())
        if total == 0:
            self.win_label.config(text="黑棋")
            return
        wv = sum(c.get_value() * c.visit_count for c in root.children.values()) / total
        if self.game.current_player == 1:
            black_win = (1 - wv) / 2
        else:
            black_win = (wv + 1) / 2

        W, H = 200, 30
        bw = int(W * black_win)
        if bw > 0:
            self.win_canvas.create_rectangle(0, 0, bw, H, fill='black', outline='')
        if bw < W:
            self.win_canvas.create_rectangle(bw, 0, W, H, fill='white', outline='')
        self.win_canvas.create_rectangle(0, 0, W, H, outline='gray')
        clamped = max(-0.9999, min(0.9999, black_win * 2 - 1))
        self.win_label.config(text=f"黑棋: {SCALE*math.atanh(clamped):+.2f}")

    # ---------------- 目差 ----------------

    def _update_score_diff(self):
        """每次落子后计算并显示盘面目差（调用 game.compute_score）"""
        black_score, white_score, diff = self.game.compute_score()
        self.score_label.config(text=f"黑棋: {diff:+.1f} ")

    # === 更新 Value Net 估值 + 归属图 + 目差 ===
    def _update_current_value(self):
        """更新 Value Net 估值 + 归属图 + 目差估计"""
        try:
            state = self.game.get_canonical_state()
            legal_moves, legal_mask = self.game.get_legal_moves_and_mask()
            policy, value, ownership_view = self.mcts.model.get_policy_value_ownership(
                state, legal_mask, device=self.mcts.device
            )
            self.current_value = value
            # 转换为绝对归属（黑=1，白=-1）
            self.current_ownership = ownership_view * self.game.current_player
            
            # 计算领地目差
            if self.current_ownership is not None:
                own = self.current_ownership
                # 正=黑，负=白
                self.black_weighted = float(np.sum(own[own > 0]))
                self.white_weighted = float(-np.sum(own[own < 0]))
            else:
                self.black_weighted = 0
                self.white_weighted = 0
                
        except Exception as e:
            self.current_value = None
            self.current_ownership = None
            self.black_weighted = 0
            self.white_weighted = 0

    # ---------------- 落子 ----------------

    def _on_click(self, event):
        if self.game.game_over or not self._is_human_turn():
            return
        pos = self.board.coord_to_index(event.x, event.y)
        if pos and self.game.is_legal_move(pos[0], pos[1]):
            self._do_move(pos[0], pos[1])

    def _human_pass(self):
        if self.game.game_over or not self._is_human_turn():
            return
        self._do_move(-1, -1)

    def _do_move(self, r, c):
        if self.game.game_over:
            return
        if not self.game.is_legal_move(r, c) and (r, c) != PASS_MOVE:
            return
        self.game.make_move(r, c)
        self._update_score_diff()
        self.mcts.update_root(self.game, (r, c) if (r, c) != PASS_MOVE else PASS_MOVE)
        self._update_current_value()   # 落子后更新估值+归属
        self._draw()
        if self.game.game_over:
            self._search_gen += 1
            msg = "平局！" if self.game.winner == 0 else f"{'黑' if self.game.winner == 1 else '白'}棋获胜！"
            messagebox.showinfo("游戏结束", msg)
        else:
            self._start_search()

    def _select_move(self):
        root = self.mcts.root
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visit_count for m in moves])
        if np.sum(visits) == 0:
            probs = np.array([root.children[m].prior_prob for m in moves])
            probs = probs / (probs.sum() + 1e-8)
            return moves[np.random.choice(len(moves), p=probs)]
        temp = self.mcts.temperature
        if temp <= 0.05:
            return moves[np.argmax(visits)]
        probs = visits ** (1.0 / max(temp, 0.01))
        probs = probs / (probs.sum() + 1e-8)
        return moves[np.random.choice(len(moves), p=probs)]

    # ---------------- 单线程协作式搜索 ----------------

    def _start_search(self):
        if self.game.game_over:
            return
        self._search_gen += 1
        self._sim_count = 0
        gen = self._search_gen
        self.root.after(STEP_MS, lambda: self._search_step(gen))

    def _search_step(self, gen):
        # 旧搜索作废（点击了新点/重置/改参数）
        if gen != self._search_gen or self.game.game_over:
            return
        n = self.mcts.num_simulations
        if self._sim_count >= n:
            # 搜索完成：轮到 AI 才落子；轮到人类则等待
            if not self._is_human_turn():
                best = self._select_move()
                self._do_move(best[0], best[1])
            return
        batch_n = min(BATCH, n - self._sim_count)
        self.mcts.simulate_batch(self.game.copy(), batch_n)
        self._sim_count += batch_n
        self._draw()
        self.root.after(STEP_MS, lambda: self._search_step(gen))

    # ---------------- 参数 / 重置 ----------------

    def _apply_params(self):
        self._search_gen += 1
        self.mcts.num_simulations = self.sim_var.get()
        self.mcts.c_puct = self.cpuct_var.get()
        self.mcts.temperature = self.temp_var.get()
        new_komi = self.komi_var.get()
        if new_komi != self.komi:
            self.komi = new_komi
            self.game.komi = new_komi  # 更新当前游戏的贴目
            self._update_score_diff()   # 更新显示
        self.mcts.root = None
        self.mcts.init_root(self.game)
        self._update_current_value()   # 参数改变，重新估值
        self._draw()
        self._start_search()

    def _reset(self):
        self._search_gen += 1
        self.game.reset()
        self.game.komi = self.komi
        self.mcts.root = None
        self.mcts.init_root(self.game)
        self._update_score_diff()
        self._update_current_value()   # 重置后更新估值+归属
        self._draw()
        self._start_search()


if __name__ == "__main__":
    GomokuGUI()