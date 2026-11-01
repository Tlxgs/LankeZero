# 烂柯 Zero

AlphaZero 风格的围棋引擎：全卷积策略/价值网络配合 MCTS 自对弈训练，提供本机对弈界面和无界面训练/对弈入口。

## 特性

- 一套权重支持多种棋盘尺寸和中心镂空棋盘。
- 完整围棋规则：自杀禁手、位置超级劫、终局计分与归属。
- AlphaZero 式训练：MCTS 自对弈、访问计数策略目标、归属/胜率辅助头、数据增强。
- 本机单进程运行，练习、对弈和训练都在本机完成。
- Tk 对弈界面和训练界面，训练参数可调，支持回放与手动修正策略标签。
- 可选 TensorRT 推理加速；不可用或引擎不匹配时自动回退 PyTorch。

## 快速开始

需要 Python 3.8+ 和 PyTorch；训练建议使用 CUDA 显卡。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r python/requirements.txt
python python/gui.py        # 对弈界面
python python/train_gui.py  # 训练界面
```

仓库包含唯一训练权重 `model.pt`，训练时会原地原子更新。

也可以不打开界面，直接跑无界面训练或对弈：

```bash
python python/train_headless.py --help
python python/play_headless.py --help
```

## 目录

```text
python/
  game.py           规则、位置超级劫、终局与输入编码
  model.py          CNN 策略/归属/胜率网络
  mcts.py           MCTS
  selfplay.py       自对弈
  replay.py         训练数据与回放
  train.py          训练循环
  trt_engine.py     可选 TensorRT 后端（自动回退）
  build_trt.py      TensorRT 引擎缓存构建
  train_headless.py 无界面训练入口
  play_headless.py  无界面对弈入口
  gui.py            对弈界面
  train_gui.py      训练界面
  tests/            单元测试
model.pt            唯一训练权重
README.md
```

## 测试

```bash
python python/tests/test_core.py
```
