"""棋盘形状 = 一张 (n, n) 布尔掩码。

全项目只有这一个形状原语：

    mask[r, c] == True  表示该点不存在（不能落子）

引擎只认 `GoGame.board` 里的 HOLE 哨兵，气/提子/数目所有判定都把 HOLE 当墙，
所以**心形、圆环、多个洞、贴着盘边的缺口、把棋盘切成两块** 都不需要任何
特殊分支 —— 只要把掩码刷进 board。

洞与棋盘同奇偶时，居中方块才是真正居中的（14 路配 6 路洞：上下各留 4 行）。
偶数路棋盘没有正中心交叉点，这是几何事实，不影响任何规则。

D4 旋转/镜像对**任何**形状都成立：棋盘和棋子一起转，得到的仍是同一局棋的
同构视角，策略/归属标签跟着转即可，不需要限定对称群。
"""
import os

import numpy as np

# 文本表示：'#' 是洞，'.' 是可下点，'X' 是黑子，'O' 是白子。
# 同一个文件既是「棋盘形状」也是「开局摆子」——空盘（只有 . 和 #）就是普通开局。
HOLE_CHARS = frozenset("#")
EMPTY_CHARS = frozenset(". ")
BLACK_CHARS = frozenset("Xx")
WHITE_CHARS = frozenset("Oo")


# ---------------- 基本校验 ----------------

def validate_mask(mask, board_size=None):
    """校验并返回 (n, n) 的 bool 掩码。

    只要求：形状对、至少留一个可落子点。不要求洞连通，也不要求洞是凸的 ——
    圆环（棋盘本身就是一圈）和两个分离的洞都是合法棋形。
    """
    m = np.asarray(mask, dtype=bool)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError("形状掩码必须是 n×n 方阵，收到 %r" % (m.shape,))
    n = m.shape[0]
    if board_size is not None and int(board_size) != n:
        raise ValueError("掩码边长 %d != 棋盘 %d" % (n, int(board_size)))
    if not m.any():
        return m                      # 实心盘
    if m.all():
        raise ValueError("棋盘上没有任何可落子点")
    return m


def playable_area(mask):
    """可落子点数。所有按面积缩放的规则（手数上限/下限/pass 上限）都用它。"""
    m = np.asarray(mask, dtype=bool)
    return int(m.size - m.sum())


# ---------------- 内置形状 ----------------

def solid(n):
    """实心盘。"""
    n = int(n)
    return np.zeros((n, n), dtype=bool)


def centered_square(n, h):
    """居中方块洞。要求洞与棋盘同奇偶、且至少留两圈。

    14 路配 6 = 洞占 4..9，上下各 4 行，正中；8 路配 4 同理。
    奇偶不同时洞会偏半格（一边比另一边多一行），这里直接拒绝，
    免得同一句「13 路 6 镂空」在不同代码路径里画出两种盘。
    """
    n, h = int(n), int(h)
    if h == 0:
        return solid(n)
    if h % 2 != n % 2:
        raise ValueError("棋盘 %d 路与镂空 %d 奇偶不同，无法居中" % (n, h))
    if h < 3:
        raise ValueError("镂空边长至少 3: %d" % h)
    if h > n - 4:
        raise ValueError("棋盘 %d 路至少留两圈，镂空最大 %d，收到 %d" % (n, n - 4, h))
    lo = (n - h) // 2
    m = np.zeros((n, n), dtype=bool)
    m[lo:lo + h, lo:lo + h] = True
    return m


def spec_from_name(n, name):
    """按名字取 (洞掩码, 开局棋子)。

    棋子是 (n,n) int8：1=黑、-1=白、0=空；没有棋子时返回 None。
    GUI / 命令行都走这里，保证「同一个文件 = 同一张棋盘 + 同一个开局」。
    """
    n = int(n)
    name = str(name or "solid").strip().lower()
    if name in ("", "solid", "none", "0"):
        return solid(n), None
    if name.startswith("square"):
        h = int(name.split(":")[1]) if ":" in name else 0
        return centered_square(n, h), None
    if name == "ring":
        return ring(n), None
    path = shape_file(name)
    if path and os.path.exists(path):
        return load_board_file(path, board_size=n)
    raise ValueError("未知形状 %r：内置只支持 solid / square:H / ring，"
                     "其它形状请放进 %s" % (name, SHAPE_DIR))


def ring(n, thickness=1):
    """圆环盘：只保留最外几圈可下，中间全部是洞。

    这是「洞不一定拓扑等价于一个圆」的最直接例子 —— 这里洞就是一个大圆盘，
    棋盘本身才是个环。
    """
    n, t = int(n), max(1, int(thickness))
    m = np.ones((n, n), dtype=bool)
    m[:t, :] = False
    m[-t:, :] = False
    m[:, :t] = False
    m[:, -t:] = False
    return m


# 心形、十字这类「有具体造型」的盘一律不做内置生成 —— 用几何公式凑出来的
# 形状又难调又难看。需要什么形状就画成文本文件放进 shapes/（见下面的 IO），
# 文件名当形状名用。

# ---------------- 形状文件 ----------------

SHAPE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "shapes")


def shape_file(name):
    """把形状名解析成 shapes/ 下的文件路径（找不到返回 None）。"""
    name = str(name).strip()
    if name.endswith(".txt") or os.path.sep in name:
        return os.path.abspath(name)
    return os.path.join(SHAPE_DIR, name + ".txt")


def load_board_file(path, board_size=None):
    """从文本文件读 (洞掩码, 开局棋子)。'#' 洞、'.' 空、X 黑、O 白。"""
    with open(path, "r", encoding="utf-8") as f:
        return from_text_board(f.read(), board_size=board_size)


def save_shape_file(path, mask, stones=None):
    """写文本文件（可手改、可进版本库）。stones 为空时与旧格式完全一致。"""
    path = os.path.abspath(path)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_text_board(mask, stones) + "\n")
    return path


# ---------------- 文本互转 ----------------

def from_text(text, board_size=None):
    """旧接口：只要形状掩码（棋子会被丢弃，GUI 请改用 from_text_board）。"""
    return from_text_board(text, board_size=board_size)[0]


def from_text_board(text, board_size=None):
    """把 ASCII 画出来的棋盘/开局转成 (掩码, 棋子)。

    允许的字符：'#' = 洞，'.' / 空格 = 空点，'X'/'x' = 黑子，'O'/'o' = 白子；
    空行忽略。所有行必须等长。
    """
    rows = [ln.rstrip("\n") for ln in str(text).splitlines() if ln.strip()]
    if not rows:
        raise ValueError("形状文本为空")
    n = len(rows)
    for i, ln in enumerate(rows):
        if len(ln) != n:
            raise ValueError("第 %d 行长度 %d != %d（棋盘必须是正方形）"
                             % (i + 1, len(ln), n))
    if board_size is not None and int(board_size) != n:
        raise ValueError("形状边长 %d != 棋盘 %d" % (n, int(board_size)))
    m = np.zeros((n, n), dtype=bool)
    stones = np.zeros((n, n), dtype=np.int8)
    for r, ln in enumerate(rows):
        for c, ch in enumerate(ln):
            if ch in HOLE_CHARS:
                m[r, c] = True
            elif ch in BLACK_CHARS:
                stones[r, c] = 1
            elif ch in WHITE_CHARS:
                stones[r, c] = -1
            elif ch not in EMPTY_CHARS:
                raise ValueError("第 %d 行第 %d 列出现无法识别的字符 %r"
                                 % (r + 1, c + 1, ch))
    m = validate_mask(m)
    if stones.any() and (stones[m] != 0).any():
        raise ValueError("洞上不能放棋子")
    return m, (stones if stones.any() else None)


def to_text(mask):
    """掩码转回 ASCII（'#' = 洞），用于日志、SGF 注释和界面回显。"""
    return to_text_board(mask, None)


def to_text_board(mask, stones=None):
    """掩码 + 棋子转回 ASCII：'#' 洞、'.' 空、'X' 黑、'O' 白。"""
    m = np.asarray(mask, dtype=bool)
    n = m.shape[0]
    if stones is None:
        return "\n".join("".join("#" if v else "." for v in row) for row in m)
    st = np.asarray(stones, dtype=np.int8)
    out = []
    for r in range(n):
        row = []
        for c in range(n):
            if m[r, c]:
                row.append("#")
            elif st[r, c] == 1:
                row.append("X")
            elif st[r, c] == -1:
                row.append("O")
            else:
                row.append(".")
        out.append("".join(row))
    return "\n".join(out)


def validate_stones(stones, mask, board_size=None):
    """校验开局棋子：形状对、洞上无子；返回 (n,n) int8（全空时返回 None）。"""
    if stones is None:
        return None
    st = np.asarray(stones, dtype=np.int8)
    m = np.asarray(mask, dtype=bool)
    if st.shape != m.shape:
        raise ValueError("棋子形状 %r 与棋盘 %r 不一致" % (st.shape, m.shape))
    if board_size is not None and int(board_size) != st.shape[0]:
        raise ValueError("棋子边长 %d != 棋盘 %d" % (st.shape[0], int(board_size)))
    if not np.isin(st, (-1, 0, 1)).all():
        raise ValueError("棋子只能是 -1/0/1")
    if (st[m] != 0).any():
        raise ValueError("洞上不能放棋子")
    return st if st.any() else None


# ---------------- 序列化 ----------------

def pack_bits(mask):
    """把掩码压成 uint8 位图（23×23 = 67 字节），存进 npz。"""
    m = np.asarray(mask, dtype=bool)
    return np.packbits(m.reshape(-1))


def unpack_bits(bits, n):
    """从位图还原掩码；n 必须与压缩时一致。

    np.packbits 会凑满整字节，所以解出来要按 n*n 截断（23 路 = 529 位 → 536 位）。
    """
    n = int(n)
    arr = np.unpackbits(np.asarray(bits, dtype=np.uint8))
    if arr.size < n * n:
        raise ValueError("形状位图长度 %d < %d×%d" % (arr.size, n, n))
    return arr[:n * n].astype(bool).reshape(n, n)
