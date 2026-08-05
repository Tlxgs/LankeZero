# LankeGo

基于深度强化学习的19路围棋AI（AlphaZero架构实现）

---

## 📖 项目简介

**LankeGo**（烂柯围棋）是一个从零实现的19路围棋AI，核心算法基于DeepMind的AlphaZero架构。

> "烂柯"典出南朝《述异记》——王质入山观棋，斧柄烂尽，归而百年已过。以此命名，既致敬围棋千年文化，亦寓意AI在棋道中"观棋烂柯"般的超然算力。

项目实现了完整的训练与对弈闭环：

- **神经网络**：10层残差网络，多任务输出（策略/领地/胜率）
- **MCTS搜索**：带虚拟损失的蒙特卡洛树搜索，支持批处理推理
- **自对弈训练**：多进程数据生成 + 异步训练流水线
- **GUI对弈**：基于Tkinter的可视化界面，支持人机对战与训练监控

---

## 🎯 核心特性

| 模块 | 技术实现 | 亮点 |
| :--- | :--- | :--- |
| **神经网络** | 10层ResBlock + 策略头/归属头/胜率头 | 多任务学习，端到端训练 |
| **MCTS搜索** | PUCT算法 + 虚拟损失 + 批处理 | 400次模拟/步，实时调参 |
| **自对弈训练** | multiprocessing + TRT推理加速 | 训练与数据生成解耦，内存优化 |
| **终局判定** | 安全点捕获法（KataGo风格） | 精确判定死子，目差计算准确 |
| **GUI界面** | Tkinter + 实时搜索可视化 | 胜率/访问量/领地归属多模式展示 |

---

## 🏗️ 项目结构

```
LankeGo/
├── game.py              # 围棋逻辑（落子/提子/劫争/终局判定）
├── model.py             # 神经网络定义（ResBlock + 多头输出）
├── mcts.py              # MCTS搜索（含TensorRT推理封装）
├── selfplay.py          # 自对弈数据生成（纯推理，不加载torch）
├── train.py             # 训练核心（SelfPlayTrainer）
├── train_gui.py         # 训练GUI界面（可视化训练过程）
├── gui.py               # 对弈GUI界面（人机对弈）
├── eval.py              # 模型评估（当前 vs 最佳）
├── hyperparams.py       # 统一超参数配置
├── ui_utils.py          # GUI通用组件（棋盘绘制/统计面板）
├── warmup_engine.py     # TRT引擎预热脚本
├── requirements.txt     # 依赖列表
└── data/                # 对局数据存储目录
```

---

## 🚀 快速开始

### 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/LankeGo.git
cd LankeGo

# 安装依赖
pip install -r requirements.txt
```

### 主要依赖

- Python 3.8+
- PyTorch 2.0+
- onnxruntime-gpu（TensorRT EP可选）
- numpy / numba
- tkinter（系统自带）

### 1. 模型初始化

```bash
python -c "from model import PolicyValueNet; PolicyValueNet().save_model('model.pt')"
```

### 2. 启动对弈（人机对战）

```bash
python gui.py
```

**GUI操作说明：**
- **鼠标点击棋盘**：手动落子
- **空格键**：AI落子
- **侧边栏**：调节模拟次数 / c_puct / 温度 / 贴目
- **显示模式**：切换"对弈棋盘 / MCTS统计 / 策略网络概率 / 领地归属"

### 3. 启动训练

```bash
python train_gui.py
```

**训练流程：**
1. 设置训练参数（模拟次数 / 批大小 / 保存间隔）
2. 点击"开始训练" → 自动启动worker生成对局
3. 训练数据存入 `data/` 目录
4. 每N局自动保存模型并导出ONNX

### 4. TRT引擎预热（可选，加速GUI启动）

```bash
python warmup_engine.py
```

### 5. 模型评估

```bash
python eval.py --games 10 --sims 400
```

当前模型 vs 最佳模型，胜率达标则自动更新最佳模型。

---

## 🧠 核心算法

### 神经网络架构

```
输入: 6通道 (19×19)
  ├── 己方棋子
  ├── 对方棋子
  ├── 上一步落子
  ├── 贴目（常量通道）
  ├── 己方棋块危急度（1气=1.0，4气=0）
  └── 对方棋块危急度

主干: Conv2d → BN → ReLU → 10×ResBlock

策略头: ResBlock×3 → Conv1×1 → (361+1) 策略概率
归属头: ResBlock×3 → Conv1×1 → tanh → 361点领地归属 (-1~1)
胜率头: MaxPool×3 → FC → tanh logit（当前玩家胜率）
```

### MCTS搜索

1. **选择**：PUCT算法 + Q值Min-Max归一化（KataGo风格）
2. **扩展**：叶子节点调用神经网络获取先验概率
3. **回传**：虚拟损失机制支持批处理并行
4. **落子**：按温度采样或贪心选择

---

## 📊 训练数据格式

对局数据以 `.pkl` 格式存储在 `data/` 目录：

```python
{
    'states':    [array(6, 19, 19)],   # 每步局面
    'policies':  [array(362)],         # MCTS访问概率
    'players':   [1 / -1],             # 当前执棋方
    'winner':    1 / -1 / 0,           # 终局胜负
    'score_diff': float,               # 目差
    'ownership': array(19, 19)         # 领地归属标签
}
```

---

## 📈 性能优化

- **TRT推理加速**：通过 `warmup_engine.py` 预构建TensorRT引擎，GUI秒开（19路首次构建约30-60s）
- **数据生成与训练解耦**：worker进程不加载torch，节省内存约1GB/进程
- **numba加速**：棋盘操作 / 终局判定使用 `@njit(cache=True)` 编译加速
- **显存碎片控制**：`PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64`
