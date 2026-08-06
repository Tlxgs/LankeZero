"""
============================================================
统一超参数配置（唯一修改入口）
------------------------------------------------------------
game / mcts / model / train / train_gui / eval / gui
全部从本文件取值，各文件不要再自行定义同名常量，
避免"改一个文件漏另一个"导致训练/评估/推理参数不一致。

注意（numba 缓存）：
game.py 的 @njit(cache=True) 函数（_make_move / _legal_moves_and_mask
/ _safe_capture_dead 等）在首次编译时把 MAX_MOVES / PASS_LIMIT /
SAFE_CAPTURE_PASSES 等作为编译期常量捕获。修改这些常量后，若运行
结果未变，请删除 __pycache__/ 下对应的 .nbc 缓存（或删除整个
__pycache__/）再运行。
============================================================
"""

# ---------------- 棋盘 / 规则 ----------------
BOARD_SIZE = 19                    # 棋盘路数（19路）
KOMI = 7.5                         # 贴目（白方补偿）
MAX_MOVES = 400                    # 单局最大手数（防死循环）
PASS_MOVE = (-1, -1)               # 停一手标记
MIN_MOVES_BEFORE_PASS = 300        # 前 N 手不允许 Pass
SCALE = 10                         # 贴目/目差归一化缩放
PASS_LIMIT = 100                     # 连续 Pass 上限（终局判定）
SAFE_CAPTURE_PASSES = 3            # 安全点迭代轮数上限（判死/判活）

# ---------------- 自对弈（训练） ----------------
NUM_SIMULATIONS = 300              # 每步 MCTS 模拟数
C_PUCT = 4.0                       # MCTS 探索常数
TEMPERATURE = 1.0                  # 训练开局温度（GUI 滑块默认值）
TEMPERATURE_DECAY = 0.995           # 温度指数衰减系数
TEMPERATURE_ZERO_AFTER = 80        # 训练温度调度
TOP_P = 0.9                        # 依概率采样时保留的累计概率
EXPLORATION_MODE = False           # 是否随机开局（四角探索）

# ---------------- MCTS 搜索 ----------------
DIRICHLET_ALPHA = 0.2              # 根节点 Dirichlet 噪声浓度
DIRICHLET_EPSILON = 0.10           # 根节点噪声混合比例
VIRTUAL_LOSS = 3                   # 虚拟损失（并行搜索去重）
BATCH_SIZE_MCTS = 8                # 推理 batch（减半：4GB 显存与 torch 训练共享）
MCTS_CAP = 600000                  # 搜索树节点容量上限（训练默认；GUI 按 GUI_MAX_SIMS 另算更大）

# ---------------- 神经网络 ----------------
INPUT_CHANNELS = 6                 # 输入通道数（己/彼棋子、贴目、危急度×2）
CHANNELS = 128                      # 主干通道数
HEAD_CHANNELS = 64                 # 头部通道数（KataGo 风格轻量塔：归属/胜率头共用第一层；三个头各约 0.1M 参数）
NUM_RES_BLOCKS = 10                # 残差块数
DROPOUT_RATE = 0.05                 # 残差块 dropout

# ---------------- 训练 ----------------
BATCH_SIZE = 256                   # 训练批大小
LEARNING_RATE = 0.0005             # 学习率（GUI 当前默认，训练入口以此为准）
WEIGHT_DECAY = 0.0001              # 权重衰减
MOMENTUM = 0.90                    # SGD 动量（GUI 当前默认）
OPTIMIZER_NAME = 'SGD'             # Adam / SGD（GUI 当前默认）
WARMUP_STEPS = 20                  # 学习率 warmup 步数（前 N 步线性升至目标 lr）
ENTROPY_WEIGHT = 0                 # 策略熵正则权重
OWN_WEIGHT = 1.0                   # 领地头损失权重
WIN_WEIGHT = 1.0                   # 胜率头损失权重（tanh(win_logit) → 当前玩家胜负 ±1）
ALPHA = 0.01                       # MCTS价值线性混合因子：value = (1-α)·胜率logit + α·(归属头目差+贴目)；0=纯胜率，1=纯目差
GRAD_CLIP_NORM = 1.0               # 梯度裁剪范数
SAVE_INTERVAL = 10                 # 保存模型间隔（局数，自对弈模式）
UI_UPDATE_BATCHES = 10             # 仅训练模式：每 N 个 Batch 更新一次界面（显示 N 个 Batch 损失均值）
SAVE_INTERVAL_BATCHES = 100        # 仅训练模式：每 N 个 Batch 保存一次模型
TRAIN_GAMES = 1000                 # GUI 默认训练局数（自对弈模式）
TRAIN_STEPS_PER_GAME = 4           # 自对弈模式每局训练步数
TRAINING_DATA_MAX = 50000          # 训练数据缓冲上限



# ---------------- 评估（eval.py / GUI"评估N次"） ----------------
EVAL_GAMES = 5                     # 评估局数（GUI Spinbox 默认）
EVAL_TEMPERATURE = 0.3             # 评估温度（近贪心，减小胜负噪声）
EVAL_WIN_RATE = 0.6                # 更新最佳模型的胜率门槛

# ---------------- 文件路径 / 目录 ----------------
MODEL_PATH = 'model.pt'            # 当前模型 .pt
ONNX_PATH = 'model.onnx'           # 当前模型 .onnx
BEST_MODEL_PATH = 'model_best.pt'  # 最佳模型 .pt
DATA_DIR = 'data/'                 # 训练/评估对局数据目录（GUI 与 eval 一致）
EVAL_WORK_DIR = 'eval_work'        # 评估引擎/快照工作目录（独立 trt_cache，与训练隔离）
SNAPSHOT_ONNX = 'eval_current.onnx'  # 评估期间冻结的"当前模型"快照名

# ---------------- GUI（人对弈 gui.py） ----------------
GUI_PROVIDER = 'cuda'              # GUI 推理执行器：cuda=稳健默认 / trt=最快 / cpu
GUI_MAX_SIMS = 10000               # GUI 最多支持模拟数（用于计算搜索树容量）
GUI_NUM_SIMULATIONS = 400          # GUI 默认模拟数（人对弈交互，保持原值）
GUI_C_PUCT = 4.0                   # GUI 探索常数
GUI_TEMPERATURE = 0.0              # GUI 落子温度（人对弈，固定贪心）
GUI_BATCH = 16                     # GUI 推理 batch
GUI_STEP_MS = 10                   # GUI 搜索步进间隔（ms）
GUI_DEBUG = True                   # GUI 调试输出开关
GUI_CRASH_LOG = 'gui_crash.log'    # GUI 异常日志（诊断闪退用）

# ---------------- 推理 / 运行时 ----------------
OMP_NUM_THREADS = 8                # OpenMP 线程数（mcts 启动时写入环境变量）
TRT_WORKSPACE = 1 << 28            # 256MB：TRT 构建期峰值显存减半（4GB 显存与训练共享）
TRT_OPT_LEVEL = 5                  # TRT builder 优化级别
TRT_FP16 = True                    # TRT FP16 加速
PYTORCH_CUDA_ALLOC_CONF = 'max_split_size_mb:64'  # torch 显存碎片控制（4GB 显存共享）


def temperature_schedule(temperature, move_count, decay=TEMPERATURE_DECAY,
                         zero_after=TEMPERATURE_ZERO_AFTER):
    """训练自对弈温度调度：
    第 zero_after 手之前：temperature * decay^move_count（下限 0.1）；
    第 zero_after 手（含）起：恒为 0（纯贪心 argmax）。
    传 zero_after=None 时保持旧的纯衰减行为（评估对局用）。
    """
    if zero_after is not None and move_count >= zero_after:
        return 0.0
    return max(0.1, temperature * (decay ** move_count))
