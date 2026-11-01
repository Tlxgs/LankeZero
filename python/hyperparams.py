"""统一超参数配置 —— 全项目唯一修改入口。

game / mcts / model / train / train_gui / gui 都从这里取值，
各文件不得自行定义同名常量（否则"改一个漏一个"，训练/评估/推理参数会不一致）。

带 GO_ 前缀的环境变量可覆盖标了 [可覆盖] 的项，用于同一份代码在不同显存的
机器上跑不同规模。默认值按本机 RTX 3050 Laptop 4GB 设定；更小的卡可把
GO_BATCH / GO_MCTS_BATCH / GO_DATA_MAX 调低。

手数/Pass 规则按可落子面积比例计算，写入 GoGame._st 运行时字段，
不再有 9/13/19 的尺寸表。

注释只解释"为什么"，不复述参数取值——写死的数值会在参数改动后变成谎言。
"""

import os

# 代码在 python/ 下运行，权重/数据/日志仍以仓库根目录为基准。
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)


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
BOARD_SIZE = 19                    # 默认棋盘路数
BOARD_SIZES = (7, 9, 11, 13, 15, 17, 19, 21, 23)
MAX_BOARD_SIZE = max(BOARD_SIZES)
MAX_BOARD_CELLS = MAX_BOARD_SIZE * MAX_BOARD_SIZE
KOMI = 7.5                         # 贴目（白方补偿；所有棋盘共用，保证同一模型输入口径一致）
PASS_MOVE = (-1, -1)               # 停一手标记
HOLE = 3                           # 棋盘数组里的镂空哨兵；0=空、1=黑、-1=白
MIN_HOLLOW = 3                     # 最小镂空边长（3×3）；0 表示实心

# 手数/Pass 规则统一按可落子面积线性缩放；不再维护 9/13/19 的表格。
# 数值 = round(可落子点数 × 系数)，因此实心盘和镂空盘共用同一套比例。
MAX_MOVES_PER_POINT = 1.3
MIN_MOVES_PER_POINT = 0.6
PASS_LIMIT_PER_POINT = 0.05


def validate_hollow(board_size, hollow=0):
    """校验并返回中心镂空边长。0=实心；否则必须是奇数，且至少留两圈。

    至少两圈的条件是 hollow <= board_size - 4：例如 13 路最大 hollow=9，
    13 路 hollow=11（只剩一圈）不允许。只允许奇数，所以 3/5/7/9 是 13 路的全部选择。
    """
    n = int(board_size)
    h = int(hollow)
    if h == 0:
        return 0
    if h % 2 == 0:
        raise ValueError("镂空边长必须是奇数或 0: %d" % h)
    if h < MIN_HOLLOW:
        raise ValueError("镂空边长至少 %d: %d" % (MIN_HOLLOW, h))
    if h > n - 4:
        raise ValueError("棋盘 %d 路至少留两圈，镂空最大 %d，收到 %d"
                         % (n, n - 4, h))
    return h


def hollow_choices(board_size):
    """返回 GUI/CLI 可选的镂空边长列表：0, 3, 5, 7, ... <= n-4。"""
    n = int(board_size)
    return [0] + list(range(MIN_HOLLOW, n - 3, 2))


def playable_area(board_size, hollow=0):
    """可落子点数（洞不算棋盘）。"""
    n = int(board_size)
    h = validate_hollow(n, hollow)
    return n * n - h * h


def _area_round(area, ratio):
    return max(1, int(area * ratio + 0.5))


def board_rules(board_size, hollow=0):
    """返回棋盘的 (max_moves, min_moves, pass_limit)。

    只按可落子点数线性缩放，任意奇数棋盘/任意合法镂空都自动适用。
    """
    area = playable_area(board_size, hollow)
    return {
        "max_moves": _area_round(area, MAX_MOVES_PER_POINT),
        "min_moves": _area_round(area, MIN_MOVES_PER_POINT),
        "pass_limit": _area_round(area, PASS_LIMIT_PER_POINT),
    }


# 终局领地判定：盘上所有子都算活 + 空区域 flood fill（Tromp-Taylor 式，
# AlphaZero / Leela Zero 的做法）。
# 数目之前先移除「子数 <= 2」或「独立气数 == 1」的棋块：终局的死子/残子绝大多数
# 就是这两类，判据硬、不会误伤活棋。
PRUNE_DEAD_LUMPS = _env_int('GO_PRUNE_LUMP', 1)

# ---------------- 自对弈（训练） ----------------
# 每步 MCTS 模拟数，训练与评估共用（分开调会让两者强度不可比）。
# 单局耗时随模拟数近似线性增长（模拟数翻倍，耗时也大致翻倍）。
# ⚠️ 环境变量名 GO_SIMS 保留，便于命令行临时覆盖。
NUM_SIMULATIONS = _env_int('GO_SIMS', 300)   # [可覆盖 GO_SIMS] 每步 MCTS 模拟数
# ---------------- MCTS 选择打分 ----------------
# 统一 KataGo 风格：Q 是“胜率 + 目差”的 utility，先在混合时压进 [-1,1]（见 mcts._utility），
# 因此 Q 的刻度是绝对的，c_puct 可以跨局面通用。探索项再随父访问数对数增长、
# 随子节点效用的标准差缩放（波动局面多探、平稳局面少探）。
C_PUCT = _env_float('GO_CPUCT', 1.0)   # MCTS 探索常数基准
# 默认值取自 KataGo 的 gtp_example.cfg（cpuct 三行等价于
# cpuctExploration / cpuctExplorationLog / cpuctUtilityStdevPrior+Scale）。
CPUCT_LOG = _env_float('GO_CPUCT_LOG', 0.45)        # c_puct 随 log(父访问数) 增长的比例
CPUCT_BASE = _env_float('GO_CPUCT_BASE', 500.0)     # 上述 log 增长的基准访问数
UTIL_SCORE_SCALE = _env_float('GO_UTIL_SCORE_SCALE', 30.0)  # 目差压进 [-1,1] 的 tanh 尺度（目）
# 基准原样搬 KataGo cfg 是 0.40，但实测我们的根节点 σ 中位只有 0.02~0.07 量级 ——
# 用 0.40 时乘数恒 < 1，等于把探索常数静默砍到约两成，「波动加成」退化成常数压制。
# 现按实测中位重标；σ 只在**已访问**的子招法上统计（未访问子是同值填补，混进去会把它稀释到 0）。
# 量法与全部读数见 tools/srv/probe_cpuct_sigma.py 与当日日志。
UTIL_STDEV_PRIOR = _env_float('GO_UTIL_STDEV_PRIOR', 0.05)  # 效用标准差基准（等于它时不缩放；已访问子不足 2 个时 σ 亦取它）
UTIL_STDEV_SCALE = _env_float('GO_UTIL_STDEV_SCALE', 0.85)  # 波动加成的强度（0=关掉该加成）
# 未访问子节点的**首访惩罚**（KataGo 的 fpuReductionMax）。本实现给未访问点填的是父节点
# 均值 pv，等于假设"没试过的点与平均水平一样好"；而探索项 c·P·√N/(1+n) 恰好在未访问点
# 最强（分母 1+n 最小）。两者叠加后，先验只要过千分之几的点都能拿到一次"礼貌性首访"，
# 访问顺序完全由先验排列、不含任何搜索证据。KataGo 填的是 pv − fpu（根节点另有更小的
# rootFpuReductionMax），给首访设了一道门槛。
# 设 0 = 恢复原行为。⚠️ 改它等于换数据生成策略，与旧缓冲不同源。
FPU_REDUCTION = _env_float('GO_FPU', 0.05)   # [可覆盖 GO_FPU] 未访问点的首访惩罚
# PASS 终局的失败方价值（按最后一手 PASS 的人结算）。比 -1 略高，让劣势方在
# “停一手输”与“继续填子也输”之间选择停一手；获胜方仍为 +1.0。
# 只影响 MCTS 搜索值，不改训练标签。
PASS_TERMINAL_LOSS = _env_float('GO_PASS_LOSS', -0.95)
# 前 zero_after 手按访问计数比例采样（τ=1，即 AlphaZero 标准），之后转贪心。
# 窗口内不做衰减：τ<1 是在"搜索标签本身已经变尖"之上再锐化一遍，等于对同一件事罚两次。
# 手数长的棋盘可以晚点转贪心；9 路对局本来就短，若沿用 80 手会几乎整局都在 τ=1 采样。
TEMPERATURE = 1.0                  # 训练开局温度（train_gui 滑块默认值）
TEMPERATURE_ZERO_PER_POINT = 0.22  # 第 N 手（含）起转贪心，N 同样按可落子面积缩放

def board_temperature_zero(board_size, hollow=0):
    """返回某棋盘的贪心切换手数；实心/镂空都按可落子面积线性缩放。"""
    return _area_round(playable_area(board_size, hollow), TEMPERATURE_ZERO_PER_POINT)

TOP_P = 0.95                       # 累计概率截断，落子采样与训练目标共用一套规则（支撑集若不一致，
                                   # 网络就会去拟合一批对局时从未被考虑过的着法）
# 探索模式开局：每局在四角各摆一子（黑2白2，角点从星位/小目/三三随机取，
# 黑棋随机同侧或对角）。用途是自对弈的**开局多样性**——根噪声扰动的是"每一手怎么选"，
# 管不住整局布局，开局一固定几百局就会反复走同一个布局，新增数据的边际信息量趋近于零。
# 代价：这 4 手是强制落子、不过 MCTS，因此不进训练样本。
# 本机默认开启；训练界面可单独勾选。只影响新生成的自对弈对局，不影响已有数据。
EXPLORATION_MODE = True

# ---------------- MCTS 搜索 ----------------
DIRICHLET_ALPHA = _env_float('GO_DIR_ALPHA', 0.03)   # [可覆盖 GO_DIR_ALPHA] 根节点 Dirichlet 噪声浓度
DIRICHLET_EPSILON = _env_float('GO_DIR_EPS', 0.20)   # [可覆盖 GO_DIR_EPS] 根节点噪声混合比例（<=0 短路关闭，见 mcts.py）
# 必须整型：mcts.py 的 vloss 累加数组是 int32，传浮点会被截断且让 numba 重编译（cache=True）
VIRTUAL_LOSS = _env_int('GO_VL', 5)    # [可覆盖 GO_VL] 虚拟损失（并行搜索去重）
BATCH_SIZE_MCTS = _env_int('GO_MCTS_BATCH', 16)  # [可覆盖 GO_MCTS_BATCH] 推理 batch
                                    # 显存小就调小、显存大可调大
MCTS_CAP = _env_int('GO_MCTS_CAP', 600000)       # [可覆盖 GO_MCTS_CAP] 搜索树节点容量上限
                                    # （训练默认；GUI 按 GUI_MAX_SIMS 另算更大）
CANDIDATE_MASS = _env_float('GO_CAND_MASS', 0.95)   # 展开时保留的累计策略质量
MIN_CANDIDATES = _env_int('GO_MIN_CAND', 8)         # 每个节点至少保留的候选数

# ---------------- 神经网络 ----------------
INPUT_CHANNELS = 10                # 输入通道数（下标见下；新增通道一律**追加在末尾**）
# 通道下标 = 平面张量第一维的切片顺序。这是单一定义点：编码侧（game._canonical_state）与
# 权重输入维都读它，改布局只动这里。
# 为什么新增通道只能追加：加载器把旧权重对齐到**前** N 个通道，新通道用**小初始化**
# （见 model.load_model 的 9→10 迁移），旧权重语义不变。
# 若插在中间，旧权重的通道含义会整体错位（掩码那片的权重被套到眼通道上），而且**不报错**。
CH_OWN_STONES = 0                  # 己方棋子
CH_OPP_STONES = 1                  # 对方棋子
CH_LAST_MOVE = 2                   # 上一手位置（无上一手则全 0）
CH_KOMI = 3                        # 贴目标量：全盘常数，随行棋方变号
CH_OWN_URGENCY = 4                 # 己方棋块危急度（1 气=1.0，>=4 气=0）
CH_OPP_URGENCY = 5                 # 对方棋块危急度
CH_LEGAL_MASK = 6                  # 合法点掩码：可落子 1、被禁 0。
                                   # 它净增的信息只有「劫禁手 + 自杀禁手」两类点 —— 空点集合网络
                                   # 本来就能从 ch0/ch1 读出，而劫点坐标原先躺在 st[3]/st[4] 里没喂进网络。
CH_OWN_EYES = 7                    # 己方棋块真眼数：0 眼=0、1 眼=0.5、>=2 眼=1（整块共享）
CH_OPP_EYES = 8                    # 对方棋块真眼数（判据见 game._group_channels）
CH_BOUNDARY = 9                    # 棋盘边界：洞 -1、可落子 0、内外边界 +1（按尺寸/hollow 生成，不存数据）
                                   # hollow=0 时退化为「最外圈 1、内部 0」。
                                   # ⚠️ 改这个值 = 换了输入编码，旧 data/ 里的样本全部作废（张量已存盘，事后补不出来）
CHANNELS = 128                      # 主干通道数
HEAD_CHANNELS = 64                 # 头部通道数（KataGo 风格轻量塔：归属/胜率头共用第一层）
NUM_RES_BLOCKS = 10                # 残差块数
DROPOUT_RATE = 0.05                 # 残差块 dropout
# 主干保持纯卷积。曾试过在末端挂自注意力块，想补"远端两点直接连线"这类路径判断
# （卷积要靠层层叠加传远，传过去时位置精度已被反复加权揉碎）。四轮对照下来，
# 块的权重位移与损失都和基线没有区别，判定它不解决当前的瓶颈，故移除。
# 完整数据见 AGENT.md 的"注意力块"一节。

# ---------------- 训练 ----------------
BATCH_SIZE = _env_int('GO_BATCH', 64)  # [可覆盖 GO_BATCH] 训练批大小（按显存调；4GB 卡实测 64 峰值约 1.7GB）
GRAD_ACCUM_STEPS = 2                # 梯度累积：每 N 个 micro-batch 才 step 一次（等效 batch = N×BATCH_SIZE），1=关闭
LEARNING_RATE = _env_float('GO_LR', 0.0002)  # [可覆盖 GO_LR] 学习率（GUI 当前默认，训练入口以此为准）
# 眼通道（ch7/ch8）的**单独学习率**，0 = 关（与主干同 lr）。
# 为什么需要：这两个通道零初始化 + 眼平面大部分是 0 ⇒ 梯度 ∝ 输入值 ⇒ 位移比主干小几个数量级，
# "唤醒期"可能长到让实验失去意义。给新参数单独放大 lr 是「加通道」技能里推荐的第一条修法。
# 为什么不用 param_groups：眼通道与旧通道在**同一个** `conv_input.weight` 张量里，参数组按整个
# Parameter 切、切不出子张量。改成放大这两片的梯度（`train.py:_boost_eye_grads`），对 SGD(+动量)
# 与放大 lr 严格等价，而且免疫「param_groups 的 lr 倍率被 warmup / 运行时改 lr 抹平」那类静默失效。
EYE_LR = _env_float('GO_EYE_LR', 0.0)        # [可覆盖 GO_EYE_LR] 眼通道单独学习率，0=关
# gpool（每个残差块里的全局池化偏置）单独学习率。和眼通道同一套做法：**放大梯度**而不是
# param_groups —— 两者共用同一条更新路径，才不会出现「两套机制并存、不知哪条在起作用」。
GPOOL_LR = _env_float('GO_GPOOL_LR', 0.0)     # [可覆盖 GO_GPOOL_LR] gpool 单独学习率，0=关
WEIGHT_DECAY = 0.0001              # 权重衰减
MOMENTUM = 0.90                    # SGD 动量（GUI 当前默认）
OPTIMIZER_NAME = 'SGD'             # Adam / SGD（GUI 当前默认）
WARMUP_STEPS = 20                  # 学习率 warmup 步数（前 N 步线性升至目标 lr）
# 策略输出熵约束：H(q) 超过 ENTROPY_TARGET 才罚，罚项 = w·(H(q)−target)²。
# 用 hinge 而非线性项（−w·H(q)），是因为线性项对「已经够尖」的输出照样给梯度，
# 会一路压下去；hinge 只在过散时推一把，把目标值定在阈值上而不是"越尖越好"。
ENTROPY_TARGET = _env_float('GO_ENT_TARGET', 2.0)      # [可覆盖 GO_ENT_TARGET] 熵约束阈值
# ⚠️ 「关掉熵约束」只能把**权重**置 0。把阈值调到 0 会让罚项变成 w·H(q)²，成了全程惩罚 ——
#    与关闭正好相反。权重为 0 时罚项恒为 0、不产生梯度，阈值留着不用即可。
ENTROPY_HINGE_WEIGHT = _env_float('GO_ENT_W', 0.0)     # [可覆盖 GO_ENT_W] 熵约束权重（0 = 关闭该项）
OWN_WEIGHT = 1.5                   # 领地头损失权重
WIN_WEIGHT = 0.5                   # 胜率头损失权重（tanh(win_logit) → 当前玩家胜负 ±1）
                                   # 实测 1.5 时共享主干梯度约 3/4 来自 win，0.5 更平衡
ALPHA = 0.5                        # MCTS价值线性混合因子：value = (1-α)·胜率logit + α·(归属头目差+贴目)；0=纯胜率，1=纯目差
GRAD_CLIP_NORM = 1.0               # 梯度裁剪范数
SAVE_INTERVAL = 10                 # 保存模型间隔（局数，自对弈模式）
UI_UPDATE_BATCHES = 10             # 仅训练模式：每 N 个 Batch 更新一次界面（显示 N 个 Batch 损失均值）
SAVE_INTERVAL_BATCHES = 100        # 仅训练模式：每 N 个 Batch 保存一次模型
TRAIN_GAMES = 1000                 # GUI 默认训练局数（自对弈模式）
TRAIN_STEPS_PER_GAME = 16          # 自对弈模式每局训练步数（= 每局反向传播次数）；除 GRAD_ACCUM_STEPS
                                   # 后才是每局 optimizer.step 次数 —— 调训练量看的是这个商
TRAINING_DATA_MAX = _env_int('GO_DATA_MAX', 200000)  # [可覆盖 GO_DATA_MAX] 训练数据缓冲上限
                                    # 紧凑样本约 2KB/条；200k 约 400MB 内存


# ---------------- 文件路径 / 目录 ----------------
MODEL_PATH = os.path.join(PROJECT_ROOT, 'model.pt')  # 全项目唯一权重；训练后原地原子更新
DATA_DIR = 'data/'                 # 每局一个紧凑 .npz

# ---------------- GUI（人对弈 gui.py） ----------------
# 模拟数与探索常数直接复用上面的 NUM_SIMULATIONS / C_PUCT。
GUI_MAX_SIMS = 3000                # GUI 最多支持模拟数（用于计算搜索树容量）
GUI_TEMPERATURE = 0.0              # GUI 落子温度（人对弈，固定贪心）
GUI_SIMS_PER_TICK = 8             # GUI 每次 UI 步进最多跑多少模拟（界面刷新粒度）
GUI_STEP_MS = 15                   # GUI 搜索步进间隔（ms）
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
