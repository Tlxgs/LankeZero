LankeZero
--------

项目概述
--------
基于 KataGo 风格的轻量级围棋AI，使用蒙特卡洛树搜索（MCTS）与残差神经网络
（策略、价值、领地归属头）联合训练。支持自对弈生成数据、模型训练/评估、
GUI 人机对弈。

主要特点：
- 19路棋盘，贴目可调
- MCTS 使用 PUCT + 根节点Dirichlet噪声
- 神经网络输出：策略（362维）、胜率logit、领地归属图
- 训练：自对弈 + 数据缓冲 + 梯度累积
- GUI：人机对弈（TensorRT/ONNX 推理）、训练监控、对局回放
- 评估：当前模型 vs 最佳模型，自动更新最佳

目录结构
--------
| 文件            | 说明                                           |
|-----------------|------------------------------------------------|
| `game.py`       | 围棋逻辑（落子、提子、劫、终局判定）           |
| `mcts.py`       | MCTS搜索（numba加速，支持TensorRT/ONNX推理）   |
| `model.py`      | PyTorch神经网络定义（残差网络+三头）           |
| `train.py`      | 训练器（SelfPlayTrainer）、数据管理            |
| `train_gui.py`  | 训练GUI（启动自对弈、监控、回放）              |
| `selfplay.py`   | 自对弈生成数据（独立模块，不依赖torch）        |
| `gui.py`        | 人机对弈GUI                                    |
| `eval.py`       | 独立评估脚本（当前 vs 最佳）                   |
| `hyperparams.py`| 统一超参数配置（唯一修改入口）                 |
| `ui_utils.py`   | GUI通用组件                                    |
| `test.py`       | 模型参数健康检查                               |

快速开始
--------
1. 安装依赖：
   PyTorch, onnxruntime-gpu, numpy, numba, tkinter

2. 训练（自对弈 + 训练）：
   python train_gui.py
   - 在GUI中调整超参数，点击“开始训练”
   - 训练数据保存至 data/ 目录
   - 模型自动保存为 model.pt / model.onnx

3. 人机对弈：
   python gui.py

超参数配置
----------
所有参数统一在 hyperparams.py 中，包括：
- 棋盘尺寸（BOARD_SIZE）
- MCTS模拟数（NUM_SIMULATIONS）
- 探索常数（C_PUCT）
- 温度调度（TEMPERATURE, TEMPERATURE_DECAY）
- 网络结构（通道数、残差块数）
- 训练批大小、学习率、优化器等

运行环境
--------
- 推荐GPU（CUDA）运行推理/训练
- 若使用TensorRT推理，需安装 onnxruntime-gpu 及 TensorRT
