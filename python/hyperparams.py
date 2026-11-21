"""统一超参数配置 —— 全项目唯一修改入口。


"""

import hashlib
import math
import os
import shutil

# 代码在 python/ 下运行，权重/数据/日志仍以仓库根目录为基准。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

_PY_DIR = os.path.dirname(os.path.abspath(__file__))


def _guard_numba_cache():
    """源码变过就清掉 numba 缓存（python/__pycache__）。

    """
    try:
        h = hashlib.sha1()
        for name in sorted(os.listdir(_PY_DIR)):
            if not name.endswith(".py"):
                continue
            with open(os.path.join(_PY_DIR, name), "rb") as fh:
                data = fh.read()
            h.update(name.encode("utf-8"))
            h.update(data)
        digest = h.hexdigest()
        stamp = os.path.join(LOG_DIR, ".numba_cache_stamp")
        old = ""
        if os.path.exists(stamp):
            with open(stamp, "r", encoding="utf-8") as fh:
                old = fh.read().strip()
        if old != digest:
            shutil.rmtree(os.path.join(_PY_DIR, "__pycache__"), ignore_errors=True)
            with open(stamp, "w", encoding="utf-8") as fh:
                fh.write(digest)
    except Exception:
        pass


_guard_numba_cache()


def _env_int(name, default):
    """读取整型环境变量覆盖值；未设置或非法时返回 default（不改动本机行为）。"""
    try:
        return int(os.environ[name])
    except (KeyError, ValueError):
        return default


def _env_float(name, default):
    """读取浮点环境变量覆盖值；未设置或非法时返回 default（不改动本机行为）。"""
    try:
        return float(os.environ[name])
    except (KeyError, ValueError):
        return default


# ---------------- 棋盘 / 规则 ----------------
BOARD_SIZE = 13                    # 默认棋盘路数

BOARD_SIZES = tuple(range(7, 26))   # 7–25 路
MAX_BOARD_SIZE = max(BOARD_SIZES)
MAX_BOARD_CELLS = MAX_BOARD_SIZE * MAX_BOARD_SIZE
KOMI = 7.5                         # 贴目（白方补偿；所有棋盘共用，保证同一模型输入口径一致）
PASS_MOVE = (-1, -1)               # 停一手标记
HOLE = 3                           # 棋盘数组里的洞哨兵；0=空、1=黑、-1=白


MAX_MOVES_PER_POINT = 1.8
MIN_MOVES_PER_POINT = 0.0


def _area_round(area, ratio):
    return max(1, int(area * ratio + 0.5))


def board_rules_for_area(area):
    """按可落子面积返回 (max_moves, min_moves)。

    实心盘、方块洞、心形盘、圆环盘全都走这一条。
    """
    area = int(area)
    if area <= 0:
        raise ValueError("可落子面积为 0，无法定义对局规则")
    return {
        "max_moves": _area_round(area, MAX_MOVES_PER_POINT),
        "min_moves": _area_round(area, MIN_MOVES_PER_POINT),
    }


def board_rules(board_size=BOARD_SIZE):
    """整盘（实心）的规则。只给「手上没有形状、只需要一个上界」的场合用。"""
    return board_rules_for_area(int(board_size) ** 2)




# ---------------- 自对弈（训练） ----------------
# 每步 MCTS 模拟数，训练与评估共用（分开调会让两者强度不可比）。
# 单局耗时随模拟数近似线性增长（模拟数翻倍，耗时也大致翻倍）。
# ⚠️ 环境变量名 GO_SIMS 保留，便于命令行临时覆盖。
NUM_SIMULATIONS = _env_int('GO_SIMS', 360)   # [可覆盖 GO_SIMS] 每步 MCTS 模拟数
# ---------------- MCTS 选择打分 ----------------
# 统一 KataGo 风格：Q 是“胜率 + 目差”的 utility，先在混合时压进 [-1,1]（见 mcts._utility），
# 因此 Q 的刻度是绝对的，c_puct 可以跨局面通用。探索项再随父访问数对数增长、
# 随子节点效用的标准差缩放（波动局面多探、平稳局面少探）。
C_PUCT = _env_float('GO_CPUCT', 1.0)   # MCTS 探索常数基准
# 默认值取自 KataGo 的 gtp_example.cfg（cpuct 三行等价于
# cpuctExploration / cpuctExplorationLog / cpuctUtilityStdevPrior+Scale）。
CPUCT_LOG = _env_float('GO_CPUCT_LOG', 0.6)        # c_puct 随 log(父访问数) 增长的比例
CPUCT_BASE = _env_float('GO_CPUCT_BASE', 1000.0)     # 上述 log 增长的基准访问数
UTIL_SCORE_PER_LINE = _env_float('GO_UTIL_SCORE_PER_LINE', 2)


def util_score_scale(area):
    """目差的 tanh 尺度（目）：随 √可落子面积 缩放（实心盘 = 1.6·边长）。"""
    return UTIL_SCORE_PER_LINE * math.sqrt(max(float(area), 1.0))


def is_half_step_komi(komi, eps=1e-9):
    """
    贴目必须是 0.5 的整数倍
    """
    v = float(komi)
    twice = v * 2.0
    return abs(twice - round(twice)) < eps


def validate_komi(komi):
    v = float(komi)
    if not is_half_step_komi(v):
        raise ValueError(
            '贴目必须是 0.5 的整数倍（例如 7.5 或 7），当前为 %r' % (komi,))
    if not -100.0 <= v <= 100.0:
        raise ValueError('贴目超出合理范围: %r' % (komi,))
    return v

UTIL_STDEV_PRIOR = _env_float('GO_UTIL_STDEV_PRIOR', 0.1)  # 效用标准差基准（等于它时不缩放；已访问子不足 2 个时 σ 亦取它）
UTIL_STDEV_SCALE = _env_float('GO_UTIL_STDEV_SCALE', 0.8)  # 波动加成的强度（0=关掉该加成）

FPU_REDUCTION = _env_float('GO_FPU', 0.02)   # [可覆盖 GO_FPU] 未访问点的首访惩罚

RESIGN_THRESHOLD = _env_float('GO_RESIGN', 0.10)

TEMPERATURE = 0.6                  # 训练开局温度（train_gui 滑块默认值）
TEMPERATURE_ZERO_PER_POINT = 0.2  # 第 N 手（含）起转贪心，N 同样按可落子面积缩放

def board_temperature_zero_for_area(area):
    """返回某棋盘的贪心切换手数；按可落子面积线性缩放，任意形状同一套。"""
    return _area_round(area, TEMPERATURE_ZERO_PER_POINT)

TOP_P = 0.95                       # 累计概率截断，落子采样与训练目标共用一套规则（支撑集若不一致，
                                   # 网络就会去拟合一批对局时从未被考虑过的着法）

PCR_ENABLED = _env_int('GO_PCR', 0) > 0           # [可覆盖 GO_PCR] 1=默认开启
PCR_PROB = _env_float('GO_PCR_PROB', 0.25)        # [可覆盖 GO_PCR_PROB] 全搜索局面比例
PCR_FAST_DIV = _env_int('GO_PCR_FAST_DIV', 6)     # [可覆盖 GO_PCR_FAST_DIV] 快搜索访问数除数

# ---------------- MCTS 搜索 ----------------
DIRICHLET_ALPHA = _env_float('GO_DIR_ALPHA', 0.03)   # [可覆盖 GO_DIR_ALPHA] 根节点 Dirichlet 噪声浓度
DIRICHLET_EPSILON = _env_float('GO_DIR_EPS', 0.20)   # [可覆盖 GO_DIR_EPS] 根节点噪声混合比例（<=0 短路关闭，见 mcts.py）
# 必须整型：mcts.py 的 vloss 累加数组是 int32，传浮点会被截断且让 numba 重编译（cache=True）
VIRTUAL_LOSS = _env_int('GO_VL', 5)    # [可覆盖 GO_VL] 虚拟损失（并行搜索去重）
BATCH_SIZE_MCTS = _env_int('GO_MCTS_BATCH', 16)  # [可覆盖 GO_MCTS_BATCH] 推理 batch
                                    # 显存小就调小、显存大可调大
MCTS_CAP = _env_int('GO_MCTS_CAP', 600000)       # [可覆盖 GO_MCTS_CAP] 搜索树节点容量上限
                                    # （训练默认；GUI 按 GUI_MAX_SIMS 另算更大）
# 候选裁剪（candidate_mass / min_candidates）已删除：实测不裁剪最好。
# 每个节点保留全部合法着法 —— 提前砍掉低先验点会漏掉「先验低但唯一正着」。

# ---------------- 神经网络 ----------------
# 空间通道只放“逐点不同”的平面；全局标量/向量单独走 global_features，
# 不再用整张常数平面去挤占空间通道。新增空间通道仍只能追加在末尾。
INPUT_CHANNELS = 18                # 空间输入通道数
CH_OWN_STONES = 0                  # 己方棋子
CH_OPP_STONES = 1                  # 对方棋子
CH_LAST_MOVE = 2                   # 上一手位置（无上一手则全 0）
CH_OWN_URGENCY = 3                 # 己方棋块危急度（1 气=1.0，>=4 气=0）
CH_OPP_URGENCY = 4                 # 对方棋块危急度
CH_LEGAL_MASK = 5                  # 合法点掩码：可落子 1、被禁 0。
                                   # 它净增的信息只有「劫禁手 + 自杀禁手」两类点 —— 空点集合网络
                                   # 本来就能从 ch0/ch1 读出，而劫点坐标原先躺在 st[3]/st[4] 里没喂进网络。
CH_BOUNDARY = 6                    # 棋盘边界：洞 -1、可落子 0、内外边界 +1（由形状掩码生成，不存数据）
                                   # 实心盘退化为「最外圈 1、内部 0」。
                                   # ⚠️ 改这个值 = 换了输入编码，旧 data/ 里的样本全部作废（张量已存盘，事后补不出来）
CH_OWN_LADDER = 7                  # 己方「只剩 1 气」棋块：按己方先走，+1 可逃到 >=3 气，-1 被提，0 不确定/不在打吃
CH_OPP_LADDER = 8                  # 对方「只剩 1 气」棋块：按对方先走，+1 可逃，-1 被提，0 不确定/不在打吃
# 空点按「包围它的棋色」分成三块（只标空点，棋子/洞全 0）：
CH_OWN_REGION = 9                  # 只挨己方棋子的空区域（己方地盘）
CH_OPP_REGION = 10                 # 只挨对方棋子的空区域
CH_NEUTRAL_REGION = 11             # 其余空区域：双方都挨（单官）或都不挨（开局空盘）
# 链级特征（同色连通块 = 一条链）。注意方向不同：
#   链子数是正向（子越多值越大）；眼个数/眼面积是**危险度**（和危急度同向，值越大越危险）。
CH_OWN_CHAIN_SIZE = 12             # 己方链子数 / 20（1 子=0.05，>=20 子=1.0，正向）
CH_OPP_CHAIN_SIZE = 13             # 对方链子数 / 20
CH_OWN_EYE_CNT = 14                # 己方链挨着的「单色封闭区域」个数：0 个=1.0，1 个=0.5，>=2 个=0（危险度）
CH_OPP_EYE_CNT = 15                # 同上，对方链
CH_OWN_EYE_AREA = 16               # 己方链挨着的单色区域总面积：0=1.0，1=0.9，线性，>=10=0（危险度）
CH_OPP_EYE_AREA = 17               # 同上，对方链

# 全局特征：当前 6 维；模型只依赖 GLOBAL_DIM，编码侧与模型输入端都读这里的顺序。
# 数量类特征统一用 2*count/可落子面积 - 1 归一到 [-1, 1]：
#   0 子 → -1，半盘 → 0，满盘 → +1。
GLOBAL_DIM = 6
G_KOMI = 0                          # player * komi / KOMI_SCALE
G_OWN_STONES = 1                    # 己方棋子数（归一化）
G_OPP_STONES = 2                    # 对方棋子数（归一化）
G_OWN_CAPS = 3                      # 己方累计提子数（归一化，裁到 [-1,1]）
G_OPP_CAPS = 4                      # 对方累计提子数（归一化，裁到 [-1,1]）
G_AREA = 5                          # 可落子面积 / 361 - 1（标准 19 路 = 0）
KOMI_SCALE = 14.0

CHANNELS = 128                     # 主干通道数
HEAD_CHANNELS = 64                 # 头部通道数
NUM_RES_BLOCKS = 10                # pre-activation 残差块数
INPUT_KERNEL = 3                   # 输入 stem 卷积核
GPOOL_BLOCKS = (3, 6, 10)          # 1-based 的 gpool 位置；只在少数块做全局池化
DROPOUT_RATE = 0                # 残差块 dropout（推理无成本）
# 主干是纯卷积（10 个 PreActResBlock）。

# ---------------- 训练 ----------------
BATCH_SIZE = _env_int('GO_BATCH', 128)  # [可覆盖 GO_BATCH] 训练批大小（按显存调；4GB 卡实测 64 峰值约 1.7GB）
GRAD_ACCUM_STEPS = 2                # 梯度累积：每 N 个 micro-batch 才 step 一次（等效 batch = N×BATCH_SIZE），1=关闭
LEARNING_RATE = _env_float('GO_LR', 0.0002)  # [可覆盖 GO_LR] 学习率（GUI 当前默认，训练入口以此为准）
# 全局特征（棋子数 / 提子数）单独学习率。这四列是从 0 附近起步的新参数，
# 和已经训了很久的贴目列抢同一份梯度是吃亏的，所以单独放大。
GLOBAL_LR = _env_float('GO_GLOBAL_LR', 1e-3)  # [可覆盖 GO_GLOBAL_LR] 全局特征列的学习率，0=关
GLOBAL_INIT_STD = _env_float('GO_GLOBAL_INIT', 0.05)  # 新增全局特征列的初始化标准差
WEIGHT_DECAY = 0.0001              # 权重衰减
MOMENTUM = 0.90                    # SGD 动量（GUI 当前默认）
OPTIMIZER_NAME = 'SGD'             # Adam / SGD（GUI 当前默认）
WARMUP_STEPS = 20                  # 学习率 warmup 步数（前 N 步线性升至目标 lr）
# 训练时随机清空「上一手」通道（CH_LAST_MOVE）的样本比例。
# 0.5 = 一半样本看不到对手刚下在哪，逼模型自己从棋子分布判断要点，
# 而不是顺着对手节奏应招（自对弈数据里这个提示太强，模型会依赖它）。
# 与 D4 增强独立叠加；推理/MCTS 侧照常喂真实上一手，两种输入都要能应对。
AUG_LAST_MOVE_DROP = _env_float('GO_LAST_MOVE_DROP', 0.5)  # [可覆盖 GO_LAST_MOVE_DROP] 0 = 关闭
# ---------------- 权重滑动平均（EMA） ----------------
# 自对弈和保存都用 EMA 权重，原始权重只用于训练。
# 为什么：自对弈拿"刚更新完的原始权重"产数据，等于把这一步的噪声原样喂回搜索；
# 保存时也一样，存下来的是一帧抖动值。EMA 是标准做法（AlphaZero / KataGo 都是）。
# 足够压掉单步噪声，又不会让 EMA 落后于正在改进的模型。
EMA_ENABLED = _env_int('GO_EMA', 1)             # [可覆盖 GO_EMA] 0 = 关闭（EMA 恒等于原始权重）
EMA_DECAY = _env_float('GO_EMA_DECAY', 0.99)     # [可覆盖 GO_EMA_DECAY] 常数衰减系数
# 策略输出熵约束：H(q) 超过 ENTROPY_TARGET 才罚，罚项 = w·(H(q)−target)²。
# 用 hinge 而非线性项（−w·H(q)），是因为线性项对「已经够尖」的输出照样给梯度，
# 会一路压下去；hinge 只在过散时推一把，把目标值定在阈值上而不是"越尖越好"。
ENTROPY_TARGET = _env_float('GO_ENT_TARGET', 2.0)      # [可覆盖 GO_ENT_TARGET] 熵约束阈值
# ⚠️ 「关掉熵约束」只能把**权重**置 0。把阈值调到 0 会让罚项变成 w·H(q)²，成了全程惩罚 ——
#    与关闭正好相反。权重为 0 时罚项恒为 0、不产生梯度，阈值留着不用即可。
ENTROPY_HINGE_WEIGHT = _env_float('GO_ENT_W', 0.0)     # [可覆盖 GO_ENT_W] 熵约束权重（0 = 关闭该项）
OWN_WEIGHT = 1.5                   # 领地头损失权重
# 宏观领地项权重：((Σtanh(own) − 实际终局目差)/有效面积)^2 直接并入平均领地损失。
OWN_MACRO_WEIGHT = _env_float('GO_OWN_MACRO', 0.0)
WIN_WEIGHT = 0.5                   # 胜率头损失权重（tanh(win_logit) → 当前玩家胜负 ±1）

ALPHA = _env_float('GO_ALPHA', 0.0)   # [可覆盖 GO_ALPHA] MCTS 价值线性混合因子：
                                      # value = (1-α)·胜率logit + α·(归属头目差+贴目)；0=纯胜率，1=纯目差
GRAD_CLIP_NORM = 5.0               # 梯度裁剪范数
SAVE_INTERVAL = 20                 # 保存模型间隔（局数，自对弈模式）
UI_UPDATE_BATCHES = 10             # 仅训练模式：每 N 个 Batch 更新一次界面（显示 N 个 Batch 损失均值）
SAVE_INTERVAL_BATCHES = 100        # 仅训练模式：每 N 个 Batch 保存一次模型
TRAIN_GAMES = 10000                 # GUI 默认训练局数（自对弈模式）
TRAIN_STEPS_PER_GAME = 16          # 自对弈模式每局训练步数（= 每局反向传播次数）；除 GRAD_ACCUM_STEPS
                                   # 后才是每局 optimizer.step 次数 —— 调训练量看的是这个商
TRAINING_DATA_MAX = _env_int('GO_DATA_MAX', 150000)  # [可覆盖 GO_DATA_MAX] 训练数据缓冲上限
                                    # 紧凑样本约 2KB/条；200k 约 400MB 内存


# ---------------- 文件路径 / 目录 ----------------
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model.pt')  # 全项目唯一权重；训练后原地原子更新
DATA_DIR = 'data/'                 # 每局一个紧凑 .npz

# ---------------- GUI（人对弈 gui.py） ----------------
# 模拟数与探索常数直接复用上面的 NUM_SIMULATIONS / C_PUCT。
GUI_MAX_SIMS = 10000                # GUI 最多支持模拟数（用于计算搜索树容量）
GUI_TEMPERATURE = 0.0              # GUI 落子温度（人对弈，固定贪心）
GUI_SIMS_PER_TICK = 16             # GUI 每次 UI 步进最多跑多少模拟（界面刷新粒度）
GUI_STEP_MS = 10                   # GUI 搜索步进间隔（ms）
GUI_DEBUG = True                   # GUI 调试输出开关
GUI_CRASH_LOG = os.path.join(LOG_DIR, 'gui_crash.log')        # GUI 异常日志（诊断闪退用）
GUI_CRASH_NATIVE_LOG = os.path.join(LOG_DIR, 'gui_crash_native.log')

# ---------------- 推理 / 运行时 ----------------
OMP_NUM_THREADS = _env_int('OMP_NUM_THREADS', 8)  # [可覆盖 OMP_NUM_THREADS] OpenMP 线程数
PYTORCH_CUDA_ALLOC_CONF = os.environ.get('GO_CUDA_ALLOC_CONF', 'max_split_size_mb:64')
                                    # [可覆盖 GO_CUDA_ALLOC_CONF] torch 显存碎片控制


def temperature_schedule(temperature, move_count, zero_after):
    """训练自对弈温度调度：第 zero_after 手（含）起恒为 0（纯贪心 argmax），
    之前恒为 temperature（不衰减）。
    传 zero_after=None 表示全程都用 temperature（评估对局用，不切贪心）。
    """
    if zero_after is not None and move_count >= zero_after:
        return 0.0
    return temperature

