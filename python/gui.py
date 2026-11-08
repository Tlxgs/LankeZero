import os
import traceback
import time
import faulthandler
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import ctypes
ctypes.windll.shcore.SetProcessDpiAwareness(1)
from game import PASS_MOVE
from mcts import MCTS
from ui_utils import GameBoard
from rules_adapter import GO_RULES

from hyperparams import (MCTS_CAP,
                         GUI_MAX_SIMS, NUM_SIMULATIONS, C_PUCT,
                         BATCH_SIZE_MCTS,
                         GUI_TEMPERATURE as TEMPERATURE,
                         GUI_SIMS_PER_TICK as SIMS_PER_TICK,
                         GUI_STEP_MS as STEP_MS,
                         GUI_DEBUG as DEBUG, GUI_CRASH_LOG as CRASH_LOG,
                         GUI_CRASH_NATIVE_LOG)

import shapes

try:
    faulthandler.enable(open(GUI_CRASH_NATIVE_LOG, 'a', buffering=1))
except Exception:
    pass
def _gui_cap(board_size):
    return max(MCTS_CAP, int(GUI_MAX_SIMS * board_size * board_size * 1.2))


def _board_cell(board_size):
    """大棋盘缩小格距，避免 21/23 路窗口超出屏幕。"""
    return 60 if int(board_size) <= 19 else 48


class GameGUI:
    """对弈界面。棋种差异全部由 rules.ui 的能力开关决定（见 rules_adapter.UISpec）。"""
    def __init__(self, device='cuda', rules=None):
        self.rules = rules or GO_RULES
        self.ui = self.rules.ui
        self.board_size = self.ui.default_size
        self.shape_name = "整盘"
        self.hole_mask = shapes.solid(self.board_size)
        self.start_stones = None          # 自定义开局的棋子（None = 空盘）
        self.komi = self.rules.komi
        self.device = device
        self.game = self.rules.new_game(self.board_size, self.hole_mask, self.komi)
        self.model_path = self.ui.model_path
        print('[GUI] 正在初始化本地推理引擎: %s' % self.model_path)
        try:
            from model import PolicyValueNet
            from trt_engine import make_evaluator
            model = PolicyValueNet.load_model(self.model_path, device=device)
            evaluator = make_evaluator(
                model, board_size=self.board_size, device=device,
                max_batch=BATCH_SIZE_MCTS, use_tensorrt=True,
                trt_engine_dir=self.ui.trt_engine_dir,
                log=lambda msg: print(msg, flush=True))
            self.model = model
            self.evaluator = evaluator
            self.mcts = MCTS(evaluator=evaluator, c_puct=C_PUCT,
                             num_simulations=NUM_SIMULATIONS,
                             temperature=TEMPERATURE,
                             device=device, board_size=self.board_size,
                             alpha=self.ui.default_alpha,
                             dirichlet_epsilon=0, cap=_gui_cap(self.board_size),
                             rules=self.rules)
        except Exception as e:
            msg = (f"模型初始化失败，GUI 无法启动：\n{type(e).__name__}: {e}\n\n"
                   "请检查：\n"
                   "1) 权重是否存在，且是当前 KataGo-lite 架构（空间 11 通道 + 全局 1 维）\n"
                   f"   当前路径: {self.model_path}\n"
                   "2) PyTorch CUDA 是否可用（也可用 --device cpu）\n"
                   "3) 显存是否被其他进程占用")
            try:
                messagebox.showerror("初始化失败", msg)
            except Exception:
                print(msg)
            raise SystemExit(1)

        self.cell_size, self.margin = _board_cell(self.board_size), 60
        self.canvas_size = (self.board_size - 1) * self.cell_size + 2 * self.margin

        self.root = tk.Tk()
        self._setup_fonts()
        self.root.title(self._window_title())
        self.root.resizable(True, True)
        self.root.bind("<space>", self._on_space_key)

        # 左侧控制面板
        ctrl = ttk.LabelFrame(self.root, text="设置", padding=8)
        ctrl.grid(row=0, column=0, sticky="ns", padx=(10, 5), pady=10)

        self.sim_var = tk.IntVar(value=NUM_SIMULATIONS)
        # 推理 batch：每次前向喂多少局面。初值直接取 BATCH_SIZE_MCTS —— 面板只改这一个
        # 变量，就可以在「同局面、同模拟数」下对比搜索分布怎么随批大小变（改动只影响
        # 每批装几个叶子，不改模拟总数、不改树的走向规则）。
        self.batch_var = tk.IntVar(value=BATCH_SIZE_MCTS)
        self.cpuct_var = tk.DoubleVar(value=C_PUCT)
        self.temp_var = tk.DoubleVar(value=TEMPERATURE)
        self.alpha_var = tk.DoubleVar(value=self.mcts.alpha)   # MCTS保守系数（可运行时调整）
        # 虚拟损失：并行搜索时给"这一批正在走的路径"临时加访问罚，避免一批模拟全挤同一分支
        self.vloss_var = tk.IntVar(value=self.mcts.vloss)
        # 根节点 Dirichlet 噪声：ε 是混合比例、α 是噪声浓度（两者不是一回事）。默认 ε=0
        # ＝不加噪声、维持最强着法；想复刻训练口径就填 ε=0.20 / α=0.03。
        self.dir_eps_var = tk.DoubleVar(value=self.mcts.dirichlet_epsilon)
        self.dir_alpha_var = tk.DoubleVar(value=self.mcts.dirichlet_alpha)
        self.mode_var = tk.StringVar(value="对弈棋盘")
        self.black_auto_var = tk.BooleanVar(value=False)   # 黑棋自动落子：搜索完成后自动落子
        self.white_auto_var = tk.BooleanVar(value=False)   # 白棋自动落子

        params = [
            ("模拟次数:", self.sim_var, 10, 500, 1),
            ("推理batch:", self.batch_var, 1, 128, 1),
            ("c_puct:", self.cpuct_var, 0.0, 10.0, 0.1),
            ("虚拟损失:", self.vloss_var, 0, 20, 1),
            ("温度:", self.temp_var, 0.0, 2.0, 0.1),
            ("α:", self.alpha_var, 0.0, 1.0, 0.05),
            ("噪声 ε:", self.dir_eps_var, 0.0, 1.0, 0.05),
            ("噪声 α:", self.dir_alpha_var, 0.001, 1.0, 0.01),
        ]

        for i, (txt, var, lo, hi, inc) in enumerate(params):
            ttk.Label(ctrl, text=txt).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Spinbox(ctrl, from_=lo, to=hi, increment=inc,
                        textvariable=var, width=8).grid(row=i, column=1, pady=3)

        row = len(params)  # 现有参数行数
        if self.ui.allow_komi:
            ttk.Label(ctrl, text="贴目:").grid(row=row, column=0, sticky="w", pady=3)
            self.komi_var = tk.DoubleVar(value=self.komi)
            ttk.Spinbox(ctrl, from_=-21, to=21.0, increment=0.5,
                        textvariable=self.komi_var, width=8).grid(row=row, column=1, pady=3)
            row += 1

        ttk.Label(ctrl, text="棋盘:").grid(row=row, column=0, sticky="w", pady=3)
        self.board_size_var = tk.IntVar(value=self.board_size)
        board_cb = ttk.Combobox(ctrl, textvariable=self.board_size_var,
                                values=self.ui.board_sizes, state="readonly", width=12)
        board_cb.grid(row=row, column=1, pady=3)
        board_cb.bind("<<ComboboxSelected>>", self._on_board_size_change)

        if self.ui.allow_shape:
            row += 1
            ttk.Label(ctrl, text="形状:").grid(row=row, column=0, sticky="w", pady=3)
            self.shape_label = ttk.Label(ctrl, text="整盘", width=12)
            self.shape_label.grid(row=row, column=1, sticky="w", pady=3)
            row += 1
            ttk.Button(ctrl, text="编辑形状", command=self._on_shape_change).grid(
                row=row, column=0, columnspan=2, sticky="ew", pady=2)

        row += 1
        ttk.Label(ctrl, text="显示:").grid(row=row, column=0, sticky="w", pady=3)
        modes = ["对弈棋盘", "MCTS统计", "策略网络概率"]
        if self.ui.show_ownership:
            modes.append("领地归属")
        mode_cb = ttk.Combobox(ctrl, textvariable=self.mode_var,
                               values=modes, state="readonly", width=12)
        mode_cb.grid(row=row, column=1, pady=3)
        mode_cb.bind("<<ComboboxSelected>>", lambda _: self._draw())

        row += 1
        auto_frame = ttk.Frame(ctrl)
        auto_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        ttk.Checkbutton(auto_frame, text="黑棋自动落子", variable=self.black_auto_var,
                        command=self._on_auto_toggle).pack(side=tk.LEFT, padx=(4, 10))
        ttk.Checkbutton(auto_frame, text="白棋自动落子", variable=self.white_auto_var,
                        command=self._on_auto_toggle).pack(side=tk.LEFT)
        row += 1
        ttk.Button(ctrl, text="AI 落子", command=self._ai_move) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(ctrl, text="悔棋", command=self._undo) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(ctrl, text="应用参数", command=self._apply_params) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=4)
        row += 1
        ttk.Button(ctrl, text="重置游戏", command=self._reset) \
            .grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        if self.ui.show_pass:
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
        self.win_label = ttk.Label(wr_frame, text="黑棋",
                                   font=('微软雅黑', 12))
        self.win_label.pack()

        # 目差显示（围棋专有）
        self.score_label = None
        if self.ui.show_score:
            row += 1
            score_frame = ttk.LabelFrame(ctrl, text="启发式目差估值", padding=5)
            score_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            self.score_label = ttk.Label(score_frame, text="",
                                         font=('微软雅黑', 12))
            self.score_label.pack()

        # === 领地目差显示 (Value Net 归属头) ===
        self.territory_label = None
        if self.ui.show_ownership:
            row += 1
            territory_frame = ttk.LabelFrame(ctrl, text="领地网络估值", padding=5)
            territory_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
            self.territory_label = ttk.Label(territory_frame, text="",
                                             font=('微软雅黑', 12))
            self.territory_label.pack()

        # === 胜率头估值（网络直接输出；与"综合估值"不同：后者是 MCTS 搜索值）===
        row += 1
        winhead_frame = ttk.LabelFrame(ctrl, text="胜率头估值", padding=5)
        winhead_frame.grid(row=row, column=0, columnspan=2, sticky="ew", pady=2)
        self.winhead_label = tk.Label(winhead_frame, text="",
                                      font=('微软雅黑', 12))
        self.winhead_label.pack()

        # 棋盘
        self.canvas = tk.Canvas(self.root, width=self.canvas_size, height=self.canvas_size,
                                bg="#D1D1D1", highlightthickness=0)
        self.canvas.grid(row=0, column=1, padx=5, pady=10)
        self.board = GameBoard(self.canvas, self.board_size, self.cell_size,
                               self.margin, hole_mask=self.hole_mask)
        self.canvas.bind("<Button-1>", self._on_click)

        # 搜索状态（单线程，无锁）
        self._search_gen = 0
        self._sim_count = 0
        self._history = []                # 悔棋：落子前局面快照栈

        # === 当前领地归属头输出缓存 ===
        self.current_ownership = None

        # 当前胜率头输出（当前玩家视角胜率 ∈[0,1]；None = 该模型无胜率头/输出异常）
        self.current_win_prob = None

        # 分数头目差（当前执棋方视角的期望目差与标准差）

        self.black_weighted = 0
        self.white_weighted = 0

        # 全局异常兜底：tkinter 回调/after 中的异常写日志并继续，不闪退
        self.root.report_callback_exception = self._log_exception
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        try:
            self._init_mcts()
            self._update_score_diff()
            self._update_current_value()   # 初始估值 + 归属
            self._draw()
            self._start_search()
        except Exception as e:
            self._log_exception(type(e), e, e.__traceback__)
            print(f"[GUI] 初始化阶段异常（已忽略，继续运行）: {type(e).__name__}: {e}")
        self.root.mainloop()

    # ---------------- 初始化 / 判断 ----------------

    def _setup_fonts(self):
        """全局字体：所有 Tk/ttk 控件（Label/LabelFrame标题/Spinbox/Combobox/Button等）
        统一 12 号，替代系统默认小字体。圆圈内胜率文字不受影响（ui_utils 单独设为 10）。"""
        try:
            import tkinter.font as tkfont
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont",
                         "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
                         "TkIconFont", "TkTooltipFont"):
                try:
                    tkfont.nametofont(name).configure(size=12)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            style = ttk.Style()
            style.configure(".", font=('微软雅黑', 12))
            for st in ("TLabel", "TButton", "TSpinbox", "TCombobox",
                       "TCheckbutton", "TRadiobutton", "TLabelframe.Label",
                       "TNotebook", "TNotebook.Tab", "TEntry", "TMenubutton"):
                try:
                    style.configure(st, font=('微软雅黑', 12))
                except Exception:
                    pass
        except Exception:
            pass

    def _on_space_key(self, event):
        """空格键：AI 落子。焦点在文本输入类控件（Spinbox/Combobox/Entry）时不触发。"""
        w = self.root.focus_get()
        if w is not None and w.winfo_class() in ('TSpinbox', 'TCombobox', 'TEntry',
                                                 'Spinbox', 'Combobox', 'Entry'):
            return None
        self._ai_move()
        return "break"

    def _log_exception(self, exc_type, exc_value, exc_tb):
        """全局异常日志：写 gui_crash.log（诊断闪退用），不中断 GUI。"""
        ts = time.strftime('%Y-%m-%d %H:%M:%S')
        lines = [f"[{ts}] {exc_type.__name__}: {exc_value}",
                 ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))]
        try:
            with open(CRASH_LOG, 'a', encoding='utf-8') as f:
                f.write('\n'.join(lines) + '\n' + '-' * 60 + '\n')
        except Exception:
            pass
        print('[GUI] 捕获异常，已写入 gui_crash.log:', exc_type.__name__, exc_value)

    def _on_close(self):
        """关闭窗口：作废进行中的搜索，避免退出时 after 回调报 TclError。"""
        try:
            self._search_gen += 1
        except Exception:
            pass
        try:
            self.root.destroy()
        except Exception:
            pass

    def _init_mcts(self):
        try:
            self.mcts.root = None
            self.mcts.init_root(self.game)
            self._update_current_value()   # 根节点初始化后更新估值
        except Exception as e:
            self._log_exception(type(e), e, e.__traceback__)

    def _rebuild_tree_and_search(self):
        """重建搜索根 → 刷新估值 → 重画 → 重启搜索。

        悔棋、重置、改参数、换棋盘全都以这四步收尾（原来这段被复制了 5 份，
        每份的异常路径各写各的，于是有几处改完参数后忘了重启搜索，界面就
        停在「参数已改、树已废、搜索链断」的状态里）。
        """
        self._init_mcts()
        self._draw()
        self._start_search()

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
        bw = self.black_weighted if hasattr(self, 'black_weighted') else 0
        ww = self.white_weighted if hasattr(self, 'white_weighted') else 0
        # 领地网络（归属头逐点求和）就是目差估计：黑白点数差再减掉贴目
        if self.territory_label is not None:
            self.territory_label.config(
                text=f"黑棋: {bw-ww-float(self.game.komi):+.1f} 目"
            )
        # 胜率头：网络直接输出（未过搜索）。模型给的是"当前执棋方"胜率，要换算成黑棋视角
        p_cur = getattr(self, 'current_win_prob', None)
        if p_cur is None:
            self.winhead_label.config(text="黑棋胜率: --")
        else:
            black_p = p_cur if self.game.current_player == 1 else 1.0 - p_cur
            self.winhead_label.config(text=f"黑棋胜率: {black_p*100:.1f}%")

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
                if self.game.is_hole(r, c):
                    continue
                x = self.margin + c * self.cell_size
                y = self.margin + r * self.cell_size
                d = float(np.nan_to_num(own[r, c], nan=0.0))  # ∈[-1, 1]
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
                if self.game.is_hole(r, c):
                    continue
                p = self.game.board[r, c]
                if p:
                    self.board.draw_piece(r, c, p)
        if self.game.last_move and self.game.last_move != PASS_MOVE:
            self.board.highlight_move(*self.game.last_move)

    @staticmethod
    def _mixed_to_node_winrate(v):
        """MCTS utility（[-1,1] 的胜负式混合值）→ 该节点行棋方的胜率。"""
        return (float(v) + 1.0) / 2.0

    def _draw_stats(self, is_policy, children):
        # 棋子与对弈棋盘完全一致（黑白实心 + 最近落子红点）
        self._draw_pieces()
        if not children:
            return
        stats = []
        for m, ch in children.items():
            if m == PASS_MOVE:
                continue
            r, c = m
            if self.game.is_hole(r, c) or self.game.board[r, c] != 0:
                continue
            vis = ch.visit_count
            # 子节点存的是「轮到对手走」那一方的价值 ⇒ 取反才是我方胜率。
            wr = (1.0 - self._mixed_to_node_winrate(ch.get_value())
                  if vis > 0 else 0.5)
            stats.append((m, vis, wr, ch.prior_prob))
        if not stats:
            return
        max_val = max(s[3] if is_policy else s[2] for s in stats)
        best = max(stats, key=lambda s: s[3] if is_policy else s[1])[0]
        for m, vis, wr, prob in stats:
            if is_policy:
                # 概率按当前盘面最大值归一化后再上色，否则低先验全落在色带同一端
                rel = prob / max_val if max_val > 0 else 0.0
                self.board.draw_stat_circle(m[0], m[1], rel,
                                            f"{prob*100:.1f}%")
            elif vis:
                # 胜率本身就是 0..1，直接上色（不能再除以 max，否则 50% 会被画成满格）
                self.board.draw_stat_circle(m[0], m[1], wr,
                                            f"{wr*100:.1f}\n{vis}")
        self.board.draw_best_circle(*best)

    def _draw_pass_info(self, children):
        """在棋盘下方显示 Pass 的先验概率与搜索次数"""
        if not self.ui.show_pass:
            return
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
            self.win_label.config(text="黑胜率:50.0%|白胜率:50.0%")
            return
        total = sum(c.visit_count for c in root.children.values())
        if total == 0:
            self.win_label.config(text="黑胜率:50.0%|白胜率:50.0%")
            return
        # 综合估值胜率：先把每个子节点的混合价值换算成「我方胜率」，再按访问量加权 ——
        # 平均要在**胜率空间**做：在效用空间平均再换算不是同一件事（非线性）。
        wv = sum((1.0 - self._mixed_to_node_winrate(c.get_value())) * c.visit_count
                 for c in root.children.values()) / total
        wv = float(np.nan_to_num(wv, nan=0.5, posinf=1.0, neginf=0.0))
        if self.game.current_player == 1:
            black_win = wv
        else:
            black_win = 1.0 - wv

        W, H = 200, 30
        bw = int(W * black_win)
        if bw > 0:
            self.win_canvas.create_rectangle(0, 0, bw, H, fill='black', outline='')
        if bw < W:
            self.win_canvas.create_rectangle(bw, 0, W, H, fill='white', outline='')
        self.win_canvas.create_rectangle(0, 0, W, H, outline='gray')
        self.win_label.config(text=f"黑胜率: {black_win*100:.1f}%|白胜率: {(1-black_win)*100:.1f}%")

    # ---------------- 目差 ----------------

    def _update_score_diff(self):
        """每次落子后计算并显示盘面目差（调用 game.compute_score）"""
        if self.score_label is None:
            return
        black_score, white_score, diff = self.game.compute_score()
        self.score_label.config(text=f"黑棋: {diff:+.1f} ")

    # === 更新 Value Net 估值 + 归属图 + 目差 ===
    def _update_current_value(self):
        """更新 Value Net 估值 + 归属图 + 目差估计"""
        try:
            _, legal_mask = self.game.get_legal_moves_and_mask()
            state, gfeat = self.game.get_state_bundle(legal_mask=legal_mask)
            policy, value_view, ownership_view, win_logit_view = \
                self.mcts.model.get_policy_value_ownership(
                    state, legal_mask, device=self.mcts.device, global_features=gfeat
                )
            # 转换为绝对归属（黑=1，白=-1）；NaN/Inf 清洗 + 形状校验
            if ownership_view is not None and ownership_view.shape == (self.board_size, self.board_size):
                own = np.asarray(ownership_view, dtype=np.float64)
                own = np.nan_to_num(own, nan=0.0, posinf=0.0, neginf=0.0)
                self.current_ownership = own * self.game.current_player
            else:
                self.current_ownership = None

            # 胜率头 logit → 当前玩家视角胜率：tanh(z) 是 ±1 的胜负量，+1 映射到 100%
            if win_logit_view is not None and np.isfinite(win_logit_view):
                self.current_win_prob = (1.0 + float(np.tanh(win_logit_view))) / 2.0
            else:
                self.current_win_prob = None

            # 计算领地目差（归属头逐点求和；贴目在黑棋视角里扣掉，见 _draw）
            if self.current_ownership is not None:
                own = self.current_ownership
                # 正=黑，负=白
                self.black_weighted = float(np.sum(own[own > 0]))
                self.white_weighted = float(-np.sum(own[own < 0]))
            else:
                self.black_weighted = 0
                self.white_weighted = 0
                
        except Exception as e:
            if DEBUG:
                print(f"[GUI] _update_current_value 失败: {type(e).__name__}: {e}")
            self.current_ownership = None
            self.current_win_prob = None
            self.black_weighted = 0
            self.white_weighted = 0

    # ---------------- 落子 ----------------

    def _on_click(self, event):
        #任何人随时可落子：点击即当前执棋方落子（双人/人机/双AI通用）
        if self.game.game_over:
            return
        pos = self.board.coord_to_index(event.x, event.y)
        if pos and self.game.is_legal_move(pos[0], pos[1]):
            self._do_move(pos[0], pos[1])

    def _human_pass(self):
        #任意一方都可弃权（双人局/人机局均可用）
        if self.game.game_over:
            return
        self._do_move(-1, -1)

    def _auto_move_enabled(self):
        """当前执棋方是否勾选自动落子（黑/白独立开关）"""
        if self.game.current_player == 1:
            return self.black_auto_var.get()
        return self.white_auto_var.get()

    def _on_auto_toggle(self):
        """勾选自动落子：若搜索已完成且轮到对应方，立即触发一次 AI 落子；
        搜索未完成则由 _search_step 在完成时自动触发。

        必须再查一次 _auto_move_enabled()：两个复选框共用一个回调，轮到白棋时
        勾「黑棋自动落子」不该让白棋立刻自己走一手。
        """
        if self.game.game_over:
            return
        if self._sim_count < self.mcts.num_simulations:
            return
        if not self._auto_move_enabled():
            return
        self._ai_move()

    def _ai_move(self):
        """AI落子：当前执棋方立即落子（取当前搜索树最优；树为空则先补一轮）"""
        if self.game.game_over:
            return
        try:
            root = self.mcts.root
            if root is None or not root.children:
                self.mcts.simulate_batch(self.game.copy(),
                                         min(SIMS_PER_TICK, self.mcts.num_simulations))
            best = self._select_move()
            self._do_move(best[0], best[1])
        except Exception as e:
            self._log_exception(type(e), e, e.__traceback__)

    def _undo(self):
        """悔棋：回退最后一步（可连续点击悔多步，终局后也可悔）"""
        if not self._history:
            if DEBUG:
                print("[GUI] 无棋可悔")
            return
        self._search_gen += 1                 # 作废进行中的搜索
        self.game = self._history.pop()       # 恢复落子前局面
        self.game.komi = self.komi            # 保持当前贴目设置一致
        self._update_score_diff()
        self._rebuild_tree_and_search()       # 悔棋后重建树根 + 刷估值 + 重启搜索

    def _do_move(self, r, c):
        if self.game.game_over:
            return
        if not self.game.is_legal_move(r, c) and (r, c) != PASS_MOVE:
            return
        self._history.append(self.game.copy())   # 悔棋：落子前快照入栈
        self.game.make_move(r, c)
        self._update_score_diff()
        try:
            self.mcts.update_root(self.game, (r, c) if (r, c) != PASS_MOVE else PASS_MOVE)
        except Exception:
            # 树推进失败则整树重建，保证树与局面一致
            self.mcts.root = None
            self.mcts.init_root(self.game)
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
        if root is None or not root.children:
            return PASS_MOVE
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
        # GUI 的搜索直接调 init_root/simulate_batch，不经过 get_move_probs，所以噪声要在这里接。
        # 放在这里是因为**所有**改根路径（启动 init_root、落子 update_root、悔棋/重置/应用参数）
        # 都以 _start_search 收尾；mcts 内部按「待混标记」保证一个根只混一次，故无脑调即可 ——
        # 根没变时是空操作，ε=0 时直接短路。
        self.mcts.mix_root_noise()
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
            # 搜索完成：勾选了自动落子的执棋方由 AI 自动落子；
            # 未勾选则保持等待（AI落子按钮随时取当前最优，双人/人机/双AI通用）
            if self._auto_move_enabled():
                self._ai_move()
            return
        # 每拍至少喂满一个推理 batch —— 否则 SIMS_PER_TICK 会在 simulate_batch 内部被
        # min(batch_size, remaining) 截回小批，「推理batch」控件就成了摆设（改了没反应）。
        batch_n = min(max(SIMS_PER_TICK, self.mcts.batch_size), n - self._sim_count)
        try:
            done = self.mcts.simulate_batch(self.game.copy(), batch_n)
            if not done:
                # 根不存在（init_root 失败过）或一次都没跑成。绝不能照加
                # _sim_count —— 那会让搜索「瞬间完成」，然后拿空树选点，
                # AI 连走两个 Pass 直接认输，而且只留一行日志。
                if DEBUG:
                    print("[GUI] 搜索未能推进（搜索树为空），本手停止搜索")
                return
            self._sim_count += done
            self._draw()
        except Exception as e:
            # 搜索异常不崩：记录日志并停止本轮搜索（避免每 10ms 重复刷屏）
            self._log_exception(type(e), e, e.__traceback__)
            return
        self.root.after(STEP_MS, lambda: self._search_step(gen))

    # ---------------- 参数 / 重置 ----------------

    def _restart_board(self, size, hole_mask, shape_name, komi, stones=None):
        """按 (尺寸, 形状/开局, 贴目) 重建对局、推理器、搜索树，并重启搜索。

        切换尺寸和切换形状共用这一份逻辑。关键是「先全部构造成功、再提交」：
        原来是边改 self.* 边构造，make_evaluator / MCTS 抛异常就直接 return，
        留下「棋盘已换、mcts 还是旧尺寸、搜索链已作废」的半初始化状态，
        之后坐标取点和悔棋都会错位。

        stones 是自定义开局的棋子（(n,n) int8，1 黑 / -1 白）；None = 空盘。
        开局摆子**永远黑先**，与子数多少无关（见 GoGame._apply_start）。
        """
        hole_mask = shapes.validate_mask(hole_mask, size)
        stones = shapes.validate_stones(stones, hole_mask, size)
        try:
            game = self.rules.new_game(size, hole_mask, komi, start_board=stones)
            from trt_engine import make_evaluator
            evaluator = make_evaluator(
                self.model, board_size=size, device=self.device,
                max_batch=BATCH_SIZE_MCTS, use_tensorrt=True,
                trt_engine_dir=self.ui.trt_engine_dir,
                log=lambda msg: print(msg, flush=True))
            old = self.mcts
            mcts = MCTS(evaluator=evaluator,
                        c_puct=old.c_puct, num_simulations=old.num_simulations,
                        temperature=old.temperature, alpha=old.alpha,
                        dirichlet_alpha=old.dirichlet_alpha,
                        dirichlet_epsilon=old.dirichlet_epsilon,
                        board_size=size, batch_size=old.batch_size,
                        cap=_gui_cap(size),
                        fpu_reduction=old.fpu_reduction,
                        virtual_loss=old.vloss, rules=self.rules)
        except Exception as e:
            messagebox.showerror("切换失败", "%s: %s" % (type(e).__name__, e))
            return False
        self._search_gen += 1
        self.board_size, self.hole_mask, self.shape_name, self.komi = \
            size, hole_mask, shape_name, komi
        self.start_stones = stones
        self.game = game
        self.evaluator = evaluator
        self.mcts = mcts
        self.cell_size, self.margin = _board_cell(size), 60
        self.canvas_size = (size - 1) * self.cell_size + 2 * self.margin
        self.canvas.config(width=self.canvas_size, height=self.canvas_size)
        self.board = GameBoard(self.canvas, size, self.cell_size, self.margin,
                               hole_mask=hole_mask)
        self.shape_label.config(text=shape_name)
        self.root.title(self._window_title())
        self._history.clear()
        self._update_score_diff()
        self._rebuild_tree_and_search()
        return True

    def _on_board_size_change(self, event=None):
        n = int(self.board_size_var.get())
        if n == self.board_size:
            return
        if self.game.move_count > 0:
            if not messagebox.askyesno("切换棋盘", "切换棋盘会重置当前对局，继续？"):
                self.board_size_var.set(self.board_size)
                return
        try:
            komi = float(self.komi_var.get()) if self.ui.allow_komi else self.rules.komi
        except Exception:
            komi = self.rules.komi
        # 换尺寸后旧形状的掩码对不上，退回整盘
        if not self._restart_board(n, shapes.solid(n), "整盘", komi):
            self.board_size_var.set(self.board_size)

    def _on_shape_change(self, event=None):
        if self.game.move_count > 0:
            if not messagebox.askyesno("切换形状", "切换形状会重置当前对局，继续？"):
                return
        try:
            import shape_editor
            mask, stones, _name = shape_editor.edit_shape(
                self.root, self.board_size, self.hole_mask, self.start_stones)
        except Exception as e:
            self._log_exception(type(e), e, e.__traceback__)
            return
        if mask is None:
            return
        name = "整盘" if (not mask.any() and stones is None) else "自定义"
        self._restart_board(self.board_size, mask, name, self.komi, stones)

    def _window_title(self):
        shape = "" if self.shape_name in ("整盘", "") else " " + self.shape_name
        if not self.ui.allow_shape:
            shape = ""
        return "%s AI - %d路%s - %s" % (
            self.ui.title, self.board_size, shape, os.path.basename(self.model_path))

    def _apply_params(self):
        self._search_gen += 1
        # 校验 Spinbox 输入（手输非法文本会导致 TclError）
        try:
            self.mcts.num_simulations = int(self.sim_var.get())
        except Exception:
            self.sim_var.set(self.mcts.num_simulations)
        # 只改「每批几个叶子」：不改模拟总数、不改树的选点规则，所以下面 init_root 重建根
        # 之后就能在同一局面上重跑一次、逐点对比搜索分布（ε=0 时同局面同结果，可比）。
        try:
            self.mcts.batch_size = max(1, int(self.batch_var.get()))
        except Exception:
            self.batch_var.set(self.mcts.batch_size)
        try:
            self.mcts.c_puct = float(self.cpuct_var.get())
        except Exception:
            self.cpuct_var.set(self.mcts.c_puct)
        # 虚拟损失只影响并行搜索的去重强度，改完不用重建树（不像 Q 归一化会改 Q 量纲）
        try:
            self.mcts.virtual_loss = max(0, int(self.vloss_var.get()))
        except Exception:
            self.vloss_var.set(self.mcts.vloss)
        try:
            self.mcts.temperature = float(self.temp_var.get())
        except Exception:
            self.temp_var.set(self.mcts.temperature)
        try:
            self.mcts.alpha = float(self.alpha_var.get())
        except Exception:
            self.alpha_var.set(self.mcts.alpha)
        # 根噪声：下面 init_root 会把待混标记置位，收尾那次 _start_search 就按新值混入，
        # 所以改完 ε/α 不用手动重置搜索树。
        try:
            self.mcts.dirichlet_epsilon = float(self.dir_eps_var.get())
        except Exception:
            self.dir_eps_var.set(self.mcts.dirichlet_epsilon)
        try:
            self.mcts.dirichlet_alpha = float(self.dir_alpha_var.get())
        except Exception:
            self.dir_alpha_var.set(self.mcts.dirichlet_alpha)
        if self.ui.allow_komi:
            try:
                new_komi = float(self.komi_var.get())
            except Exception:
                new_komi = self.komi
                self.komi_var.set(self.komi)
            if new_komi != self.komi:
                self.komi = new_komi
                self.game.komi = new_komi  # 更新当前游戏的贴目
                self._update_score_diff()   # 更新显示
        try:
            self.mcts.root = None
            self.mcts.init_root(self.game)
            self._update_current_value()   # 参数改变，重新估值
            self._draw()
            self._start_search()
        except Exception as e:
            # 参数已经写进 mcts 了，但树没建起来 —— 至少把搜索链恢复，
            # 否则自动落子会静默失效，只能手动点「AI 落子」。
            self._log_exception(type(e), e, e.__traceback__)
            try:
                self._rebuild_tree_and_search()
            except Exception as e2:
                self._log_exception(type(e2), e2, e2.__traceback__)

    def _reset(self):
        self._search_gen += 1
        self._history.clear()
        self.game.reset()
        self.game.komi = self.komi
        self._update_score_diff()
        self._rebuild_tree_and_search()


# 旧名字保留：历史代码/测试里有引用。
GomokuGUI = GameGUI


if __name__ == "__main__":
    GameGUI()
