# 烂柯 Zero

一个在本机运行的 Katago 式极简围棋引擎，
**全卷积神经网络 + MCTS + 自对弈训练**。
**支持任意尺寸、任意形状（如心形、中间镂空等）、任意开局**。

## 特性

- **一份权重支持任意尺寸棋盘**，也支持镂空棋盘（心形、圆环、自定义形状），
  不需要按尺寸分别训练。
- **三个头**：策略、归属、胜率；
  归属头逐点求和就是目差估计。
- **自对弈训练**：自对弈产生对局 → MCTS 访问计数当策略标签 → 训练 → 再自对弈；
  也能导入人类棋谱。
- **完整规则**：禁自杀、禁全同、终局计分与归属、让子局、任意贴目。
- **两种界面**：对弈界面（人人 / 人机 / 双 AI）和训练界面（自对弈 + 训练 + 回放 + 手动标注）。
- **可选 TensorRT**：有匹配的引擎就用 REFIT 推理，没有或结构不匹配自动回退 PyTorch。

## 安装

需要 Python 3.8+、PyTorch(需要Nvidia显卡，CPU训练较慢不推荐)。

```bat
python -m venv venv
venv\Scripts\activate
pip install -r python\requirements.txt
:: torch 请按你的 CUDA 版本安装，例如 cu124：
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## 运行

在本目录下直接执行（源码都在 `python\` 下）：

```bat
python python\gui.py            :: 对弈界面
python python\train_gui.py      :: 训练界面
python python\train_headless.py --help     :: 无界面训练
python python\play_headless.py  --help     :: 无界面对弈
python python\bench.py --sims 300 --batch 16
```

需要一个权重文件；路径由 `--model` 指定（默认本目录的 `model.pt`）。
本仓库附带一个经过一定训练的权重，该模型结构是10个残差块、128通道。主要在13路完整棋盘上训练，因此在其他大小棋盘上可能水平相对较低。棋力仅供参考，不代表该参数量下最强水平。

没有权重就从零开始：
```bat
python python\train_headless.py --init-model --channels 128 --blocks 10 --board-size 13 --games 1000
```
