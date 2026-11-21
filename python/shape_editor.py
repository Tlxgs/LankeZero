"""棋盘形状 / 自定义开局 / 死活题编辑器（Tk）：点格子画洞或摆子，或者直接粘 ASCII。

两种输入共用一份文本：文本框是唯一真相，格子视图只是它的渲染结果。
字符约定（同时也是 shapes/*.txt 的落盘格式）：

    '.'  空点        '#'  洞
    'X'  黑子        'O'  白子

死活题模式额外允许两个关键点（见 shapes.from_text_ld）：

    'b'  黑的关键点   'w'  白的关键点   '*'  黑白同一点

空盘（只有 . 和 #）就是普通棋盘；带 X/O 就是自定义开局，加载后永远黑先。
点格子循环：空 → 黑 → 白 → 洞 → 空。

    mask, stones, name = edit_shape(parent, n, current_mask, current_stones)
    mask, stones, bpt, wpt = edit_ld_problem(parent, n, mask, stones, bpt, wpt)
    取消返回 None
"""
import os
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import numpy as np

import shapes


class _ShapeDialog(tk.Toplevel):
    CELL = 22

    def __init__(self, parent, board_size, hole_mask=None, stones=None,
                 ld=False, black_pt=None, white_pt=None):
        super().__init__(parent)
        self.ld = bool(ld)
        self.title(("死活题" if self.ld else "棋盘/开局") + " - %d 路" % board_size)
        self.resizable(False, False)
        self.transient(parent)
        self.n = int(board_size)
        self.result = None
        self.result_stones = None
        self.result_bpt = None
        self.result_wpt = None
        self.result_name = None
        self.mode = tk.StringVar(master=self, value="stone")

        n = self.n
        mask = shapes.solid(n) if hole_mask is None else np.asarray(hole_mask, bool)
        self.text = tk.Text(self, width=max(24, 2 * n + 6), height=n + 2,
                            font=("Consolas", 12))
        self.text.insert("1.0", (shapes.to_text_ld(mask, stones, black_pt, white_pt)
                                 if self.ld else shapes.to_text_board(mask, stones)))
        self.text.grid(row=0, column=0, rowspan=2, padx=6, pady=6, sticky="ns")
        self.text.bind("<KeyRelease>", lambda e: self._render())

        size = n * self.CELL + 2
        self.canvas = tk.Canvas(self, width=size, height=size, bg="white",
                                highlightthickness=1)
        self.canvas.grid(row=0, column=1, rowspan=2, padx=6, pady=6)
        self.canvas.bind("<Button-1>", self._on_click)

        bar = ttk.Frame(self)
        bar.grid(row=2, column=0, columnspan=2, sticky="ew", padx=6, pady=(0, 6))
        ttk.Button(bar, text="整盘", command=self._fill_solid).pack(side="left")
        ttk.Button(bar, text="方块洞", command=self._ask_square).pack(side="left", padx=2)
        ttk.Button(bar, text="反转", command=self._invert).pack(side="left", padx=2)
        ttk.Button(bar, text="载入", command=self._load).pack(side="left", padx=2)
        ttk.Button(bar, text="保存", command=self._save).pack(side="left", padx=2)
        ttk.Button(bar, text="确定", command=self._ok).pack(side="right")
        ttk.Button(bar, text="取消", command=self.destroy).pack(side="right", padx=4)
        if self.ld:
            ttk.Label(bar, text="   模式:").pack(side="left")
            for val, txt in (("stone", "画子"), ("b", "黑点"), ("w", "白点")):
                ttk.Radiobutton(bar, text=txt, value=val,
                                variable=self.mode).pack(side="left")

        self.grab_set()
        self._render()

    # ---------------- 文本 ↔ (掩码, 棋子, 黑点, 白点) ----------------

    def _current(self):
        """解析文本框；非法返回 None。始终返回 4 元组（形状模式下后两个为 None）。"""
        try:
            txt = self.text.get("1.0", "end")
            if self.ld:
                return shapes.from_text_ld(txt, board_size=self.n)
            mask, stones = shapes.from_text_board(txt, board_size=self.n)
            return mask, stones, None, None
        except ValueError:
            return None

    def _set(self, mask, stones=None, bpt=None, wpt=None):
        try:
            text = (shapes.to_text_ld(mask, stones, bpt, wpt) if self.ld
                    else shapes.to_text_board(mask, stones))
        except ValueError:
            text = shapes.to_text(mask)
        self.text.delete("1.0", "end")
        self.text.insert("1.0", text)
        self._render()

    def _render(self):
        cur = self._current()
        self.canvas.delete("all")
        c = self.CELL
        bpt = wpt = None
        if cur is not None:
            bpt, wpt = cur[2], cur[3]
        for r in range(self.n):
            for col in range(self.n):
                x, y = col * c, r * c
                if cur is None:
                    fill = "#FFFFFF"
                elif cur[0][r, col]:
                    fill = "#A9A9A9"
                else:
                    fill = "#F5DEB3"
                self.canvas.create_rectangle(x, y, x + c, y + c,
                                             fill=fill, outline="#999999")
                if cur is None:
                    continue
                v = 0 if cur[1] is None else int(cur[1][r, col])
                if v != 0:
                    pad = 3
                    self.canvas.create_oval(
                        x + pad, y + pad, x + c - pad, y + c - pad,
                        fill=("#1a1a1a" if v == 1 else "#f2f2f2"),
                        outline="#333333")
                    continue
                both = (bpt == (r, col) and wpt == (r, col))
                if both:
                    self.canvas.create_rectangle(x + 3, y + 3, x + c - 3, y + c - 3,
                                                 outline="#6a1b9a", width=2)
                    self.canvas.create_text(x + c / 2, y + c / 2, text="*",
                                            fill="#6a1b9a",
                                            font=("Consolas", 11, "bold"))
                elif bpt == (r, col):
                    self.canvas.create_rectangle(x + 3, y + 3, x + c - 3, y + c - 3,
                                                 outline="#1565c0", width=2)
                    self.canvas.create_text(x + c / 2, y + c / 2, text="B",
                                            fill="#1565c0",
                                            font=("Consolas", 9, "bold"))
                elif wpt == (r, col):
                    self.canvas.create_oval(x + 3, y + 3, x + c - 3, y + c - 3,
                                            outline="#c62828", width=2)
                    self.canvas.create_text(x + c / 2, y + c / 2, text="W",
                                            fill="#c62828",
                                            font=("Consolas", 9, "bold"))

    # ---------------- 交互 ----------------

    def _on_click(self, ev):
        r, c = ev.y // self.CELL, ev.x // self.CELL
        if not (0 <= r < self.n and 0 <= c < self.n):
            return
        cur = self._current()
        if cur is None:
            return
        mask, stones, bpt, wpt = cur
        if self.ld and self.mode.get() in ("b", "w"):
            # 关键点只能标在空点上（洞和棋子上不行）
            if mask[r, c] or (stones is not None and stones[r, c] != 0):
                return
            if self.mode.get() == "b":
                bpt = None if bpt == (r, c) else (r, c)
            else:
                wpt = None if wpt == (r, c) else (r, c)
            self._set(mask, stones, bpt, wpt)
            return
        if bpt == (r, c):
            bpt = None
        if wpt == (r, c):
            wpt = None
        if stones is None:
            stones = np.zeros((self.n, self.n), dtype=np.int8)
        else:
            stones = stones.copy()
        if stones[r, c] == 0 and not mask[r, c]:
            stones[r, c] = 1                    # 空 → 黑
        elif stones[r, c] == 1:
            stones[r, c] = -1                   # 黑 → 白
        elif stones[r, c] == -1:
            stones[r, c] = 0                    # 白 → 空
            mask[r, c] = True                   # 空 → 洞
        else:                                   # 洞 → 空
            mask[r, c] = False
        self._set(mask, stones if stones.any() else None, bpt, wpt)

    def _fill_solid(self):
        self._set(shapes.solid(self.n), None)

    def _invert(self):
        cur = self._current()
        if cur is None:
            return
        mask = ~cur[0]
        stones = None if cur[1] is None else cur[1].copy()
        if stones is not None:
            stones[mask] = 0                    # 洞上不能留子
            stones = stones if stones.any() else None
        self._set(mask, stones)

    def _ask_square(self):
        h = simpledialog.askinteger(
            "方块洞", "洞的边长（与 %d 路同奇偶，至少留两圈）：" % self.n,
            parent=self, minvalue=3, maxvalue=max(3, self.n - 4))
        if not h:
            return
        try:
            self._set(shapes.centered_square(self.n, h), None)
        except ValueError as exc:
            messagebox.showerror("无法居中", str(exc), parent=self)

    def _start_dir(self):
        return shapes.LD_DIR if self.ld else shapes.SHAPE_DIR

    def _load(self):
        path = filedialog.askopenfilename(
            parent=self, title=("载入死活题" if self.ld else "载入棋盘/开局"),
            initialdir=self._start_dir(),
            filetypes=[("文本", "*.txt"), ("全部", "*.*")])
        if not path:
            return
        try:
            if self.ld:
                mask, stones, bpt, wpt = shapes.load_ld_file(path, board_size=self.n)
                self._set(mask, stones, bpt, wpt)
            else:
                mask, stones = shapes.load_board_file(path, board_size=self.n)
                self._set(mask, stones)
        except (ValueError, OSError) as exc:
            messagebox.showerror("载入失败", str(exc), parent=self)

    def _save(self):
        cur = self._current()
        if cur is None:
            messagebox.showerror("文本非法", "文本里有无法解析的内容", parent=self)
            return
        path = filedialog.asksaveasfilename(
            parent=self, title=("保存死活题" if self.ld else "保存棋盘/开局"),
            initialdir=self._start_dir(), defaultextension=".txt",
            filetypes=[("文本", "*.txt")])
        if not path:
            return
        mask, stones, bpt, wpt = cur
        try:
            mask = shapes.validate_mask(mask, self.n)
            stones = shapes.validate_stones(stones, mask, self.n)
        except ValueError as exc:
            messagebox.showerror("棋盘非法", str(exc), parent=self)
            return
        if not self.ld:
            shapes.save_shape_file(path, mask, stones)
            messagebox.showinfo("已保存", path, parent=self)
            return

        # 死活题：先落 .txt，再顺手生成两个 npz（黑先 / 白先）
        shapes.save_ld_file(path, mask, stones, bpt, wpt)
        name = os.path.basename(path)
        name = name[:-4] if name.endswith(".txt") else name
        msg = path
        if bpt is None or wpt is None:
            msg += "\n\n[警告] 黑/白关键点没标全，npz 未生成"
        else:
            try:
                import ld_data
                paths = ld_data.build_npz(name, mask, stones, bpt, wpt,
                                          out_dir=os.path.dirname(path) or ".",
                                          board_size=self.n)
                msg += "\n\n已生成两个 npz:\n" + "\n".join(
                    os.path.basename(p) for p in paths)
            except Exception as exc:                    # noqa: BLE001
                msg += "\n\n[警告] npz 没生成：%s" % exc
        messagebox.showinfo("已保存", msg, parent=self)

    def _ok(self):
        cur = self._current()
        if cur is None:
            messagebox.showerror("文本非法", "文本里有无法解析的内容", parent=self)
            return
        mask, stones, bpt, wpt = cur
        try:
            mask = shapes.validate_mask(mask, self.n)
            stones = shapes.validate_stones(stones, mask, self.n)
        except ValueError as exc:
            messagebox.showerror("棋盘非法", str(exc), parent=self)
            return
        self.result = mask
        self.result_stones = stones
        self.result_bpt = bpt
        self.result_wpt = wpt
        self.result_name = "整盘" if (not mask.any() and stones is None) else "自定义"
        self.destroy()


def edit_shape(parent, board_size, hole_mask=None, stones=None):
    """弹棋盘/开局编辑器；返回 (mask, stones, name)，取消返回 (None, None, None)。"""
    dlg = _ShapeDialog(parent, board_size, hole_mask, stones)
    parent.wait_window(dlg)
    return dlg.result, dlg.result_stones, dlg.result_name


def edit_ld_problem(parent, board_size, hole_mask=None, stones=None,
                    black_pt=None, white_pt=None):
    """弹死活题编辑器；返回 (mask, stones, 黑点, 白点)，取消返回 (None,)*4。"""
    dlg = _ShapeDialog(parent, board_size, hole_mask, stones, ld=True,
                       black_pt=black_pt, white_pt=white_pt)
    parent.wait_window(dlg)
    return dlg.result, dlg.result_stones, dlg.result_bpt, dlg.result_wpt
