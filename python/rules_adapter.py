"""棋种适配层：MCTS / 自对弈 / 数据管线只通过这里的对象拿「与棋种相关」的行为。

go 是默认实现（GoRules），其他棋种在自己的目录里实现同名接口并显式传进来，
go 的所有默认路径保持不变（不传 rules 就是 GoRules）。

接口（显式、够用即可，不追求完备）：

    make_move(board, st, go, fp, cap, hist, row, col) -> bool
        Numba 可调用；签名与 game._make_move 完全一致，MCTS 把它当参数传进
        已编译的 _select_np，所以两种规则共用同一份搜索代码。
    terminal_value(game, utility_fn) -> float
        终局叶子值（当前行棋方视角）。utility_fn(win_logit, score) 是
        MCTS 的混合效用函数，规则自己决定终局只用胜负还是也看目差。
    max_moves(board_size) -> int
    game_class
    state_to_sample(...)      自对弈落盘用
    encoder                   训练时把 Sample 批量编成网络输入；None = go 内联编码
    exploration_opening(n)    自对弈开局（不产生训练样本）
    terminal_analysis(game)   终局归属图（没有归属头的棋种返回 None）
    score_of(game)            终局目差（没有目差的棋种返回 0.0）
"""
import game as _game
from game import PASS_MOVE
from hyperparams import (BOARD_SIZES, KOMI, MODEL_PATH, PASS_TERMINAL_LOSS,
                         board_rules)


def _default_state_to_sample(*args, **kwargs):
    from replay import state_to_sample
    return state_to_sample(*args, **kwargs)


def _default_exploration_opening(board_size):
    from selfplay import _exploration_opening
    return _exploration_opening(board_size)


class UISpec:
    """界面能力表：go 的 GUI / 训练 GUI 只按这些开关决定显示什么。

    默认值是围棋（全部打开）；五子棋关掉贴目、形状、归属/目差，并换掉棋盘尺寸表。
    """

    def __init__(self, title, board_sizes, default_size, model_path,
                 data_dir=None, allow_komi=True, allow_shape=True,
                 show_ownership=True, show_score=True, show_pass=True,
                 train_gui=True, default_alpha=None, trt_engine_dir=None):
        self.title = title
        self.board_sizes = tuple(board_sizes)
        self.default_size = int(default_size)
        self.model_path = model_path
        self.data_dir = data_dir or (lambda n: "data/")
        self.allow_komi = bool(allow_komi)
        self.allow_shape = bool(allow_shape)
        self.show_ownership = bool(show_ownership)
        self.show_score = bool(show_score)
        self.show_pass = bool(show_pass)
        self.train_gui = bool(train_gui)
        # None = 用 trt_engine 的默认目录（<repo>/trt_engines）；棋种可以各用各的，
        # 否则同尺寸的两个模型会抢同一个引擎文件名。
        self.trt_engine_dir = trt_engine_dir
        if default_alpha is None:
            from hyperparams import ALPHA
            default_alpha = ALPHA
        self.default_alpha = float(default_alpha)


class GoRules:
    """围棋规则（默认）。所有行为与改动前完全一致。"""

    name = "go"
    komi = KOMI
    own_mask = 1                         # 训练时是否使用归属标签（五子棋没有归属）
    move_kind = 0                        # 落子内核：0=围棋提子/劫，见 game._make_move_dispatch
    win_len = 0                          # 连子判胜长度（围棋不用）
    handicap_opening = True              # 探索开局的 1~4 子由 setup_handicap 一次性摆上
    ui = UISpec(title="围棋", board_sizes=BOARD_SIZES,
                default_size=19, model_path=MODEL_PATH,
                data_dir=lambda n: "data/" if int(n) == 19 else "data%d/" % int(n))
    game_class = _game.GoGame
    encoder = None                       # None = 用 replay.TrainingData 里的内联编码
    state_to_sample = staticmethod(_default_state_to_sample)
    exploration_opening = staticmethod(_default_exploration_opening)

    def __init__(self):
        pass

    @staticmethod
    def new_game(board_size, hole_mask=None, komi=None, start_board=None):
        return _game.GoGame(board_size,
                            komi=KOMI if komi is None else float(komi),
                            hole_mask=hole_mask, start_board=start_board)

    @staticmethod
    def captures_of(game, player):
        """累计提子数（行棋方视角，己方/对方）。"""
        black, white = int(game._st[13]), int(game._st[14])
        return (black, white) if player == 1 else (white, black)

    @staticmethod
    def max_moves(board_size):
        return int(board_rules(board_size)["max_moves"])

    @staticmethod
    def terminal_value(game, utility_fn):
        if game.winner is None:
            game._compute_winner()
        if game.winner == 0:
            return 0.0
        if game.last_move == PASS_MOVE:
            # 价值按「最后一手 PASS 的人」结算，而不是按当前行棋方。
            passer = -game.current_player
            return -1.0 if game.winner == passer else -PASS_TERMINAL_LOSS
        term_score = game.final_points * game.current_player
        term_wl = 6.0 if game.winner == game.current_player else -6.0
        return utility_fn(term_wl, term_score)

    @staticmethod
    def terminal_analysis(game):
        return game.terminal_analysis()

    @staticmethod
    def score_of(game):
        return float(game.final_points)


GO_RULES = GoRules()


def get(name):
    """按名字取规则；未注册的名字由调用方自己传对象（gomoku 就是这样挂上来的）。"""
    if name == "go":
        return GO_RULES
    raise ValueError("未知棋种 %r（只有 go 内置；其他棋种请直接传 rules 对象）" % (name,))
