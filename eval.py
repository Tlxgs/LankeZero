"""独立评估：当前模型 vs 最佳模型，一次性评估 N 局后退出（无循环、无日志文件）。

设计要点：
- 评估与训练互斥：GUI 点"评估N次"→ 本脚本跑 N 局 → 结束（无定时、无循环监督）
- 帧通道 = 原始 stdout：每局结束发一帧（4字节长度 + pickle 局数据），train_gui 读帧用于对局回放
- 文本通道 = stderr（utf-8）：进度/错误显示在 GUI 日志区，不写任何 json/txt 日志文件
- 对局数据原子保存到 --data-dir（同训练 pkl 格式，可纳入训练；--no-save-data 关闭）

用法：
  python eval.py --games 5 [--sims 300] [--data-dir data/] [--device cuda] [--provider auto]
"""
import argparse
import os
import pickle
import shutil
import struct
import sys
from datetime import datetime

from selfplay import play_one_game    # 数据生成模块（纯生成，不加载 torch；原在 train.py）
from mcts import MCTS
from hyperparams import (BOARD_SIZE, NUM_SIMULATIONS, C_PUCT, INPUT_CHANNELS,
                         EVAL_GAMES, EVAL_TEMPERATURE, EVAL_WIN_RATE,
                         DATA_DIR, MODEL_PATH, BEST_MODEL_PATH,
                         EVAL_WORK_DIR, SNAPSHOT_ONNX)

# ---- 输出通道规划（保证帧通道纯净、文本通道不乱码） ----
# 帧通道 = 原始 stdout（GUI 用 stdout 管道读对局帧）；所有文本输出（含
# mcts.TensorRTModel / numba / onnxruntime 的 print）统一改道 stderr（GUI 日志区）。
# 否则这些 print 会混进帧流，把帧协议读坏（表现为"评估结束（0 局）"）。
try:
    sys.stderr.reconfigure(encoding='utf-8')   # 中文不乱码（GUI 按 utf-8 解码）
except Exception:
    pass
_frame_out = sys.stdout.buffer     # 帧通道（原始 stdout 的二进制缓冲）
sys.stdout = sys.stderr            # 所有 print（含第三方库）→ stderr

DEFAULT_DATA_DIR = DATA_DIR          # 与 train_gui.py 的训练数据目录一致（EVAL_WORK_DIR/SNAPSHOT_ONNX 见 hyperparams.py）


def _atomic_copy(src, dst):
    tmp = dst + '.tmp'
    shutil.copy(src, tmp)
    os.replace(tmp, dst)


def _save_game(game_data, data_dir):
    """对局数据存为训练同格式 pkl（原子写，可被训练 load_all 使用）。"""
    states, policies, players, moves, winner, score_diff, ownership = game_data
    ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path = os.path.join(data_dir, 'data_%s.pkl' % ts)
    while os.path.exists(path):      # 超快对局同微秒时间戳会撞名，加唯一性保护
        ts = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        path = os.path.join(data_dir, 'data_%s.pkl' % ts)
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump({'states': states, 'policies': policies, 'players': players,
                     'winner': winner, 'score_diff': score_diff, 'ownership': ownership,
                     'source': 'eval'}, f)
    os.replace(tmp, path)


def _send_frame(game_data):
    """向帧通道发送一帧（4字节长度 + pickle 局数据），供 train_gui 对局回放。"""
    if _frame_out is None or _frame_out.isatty():   # 直接命令行运行时不输出二进制帧
        return
    payload = pickle.dumps(game_data, protocol=4)
    _frame_out.write(struct.pack('<I', len(payload)) + payload)
    _frame_out.flush()


def _inspect_onnx(path):
    """读取 ONNX 输入/输出 shape，并尽力推断 channels / res_blocks（评估前诊断用）。
    解析失败返回 {'error': ...}，不阻塞评估。"""
    try:
        import onnx
        m = onnx.load(path, load_external_data=False)

        def _dims(dim):
            out = []
            for x in dim:
                if x.HasField('dim_value') and x.dim_value > 0:
                    out.append(x.dim_value)
                else:
                    out.append(x.dim_param or '?')
            return out

        inputs = [(i.name, _dims(i.type.tensor_type.shape.dim)) for i in m.graph.input]
        outputs = [(o.name, _dims(o.type.tensor_type.shape.dim)) for o in m.graph.output]
        dims = {init.name: list(init.dims) for init in m.graph.initializer}
        channels = blocks = None
        conv = [d for d in dims.values() if len(d) == 4 and d[1] == 6]   # 输入卷积 [C,6,3,3]
        if conv:
            channels = int(conv[0][0])
        import re
        bset = set()
        for n in dims:
            mm = re.match(r'res_blocks\.(\d+)\.', n)
            if mm:
                bset.add(int(mm.group(1)))
        if bset:
            blocks = max(bset) + 1
        return {'inputs': inputs, 'outputs': outputs,
                'channels': channels, 'blocks': blocks}
    except Exception as e:
        return {'error': str(e)}


def evaluate_and_update_best(model_path=MODEL_PATH, best_path=BEST_MODEL_PATH,
                             board_size=BOARD_SIZE, num_simulations=NUM_SIMULATIONS,
                             c_puct=C_PUCT, temperature=EVAL_TEMPERATURE,
                             eval_games=EVAL_GAMES, win_rate=EVAL_WIN_RATE,
                             device='cuda', data_dir=DEFAULT_DATA_DIR,
                             provider='auto', save_data=True):
    """当前 vs 最佳，评估 eval_games 局（黑白交替，温度近贪心）。
    当前胜率 >= win_rate → 原子覆盖最佳模型(.pt+.onnx)。
    每局：保存 pkl（可选）+ 帧（回放）。返回 (wins, eval_games)。"""
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    onnx_path = model_path.replace('.pt', '.onnx')
    best_onnx = best_path.replace('.pt', '.onnx')

    if not os.path.exists(best_path):
        _atomic_copy(model_path, best_path)
        if os.path.exists(onnx_path):
            _atomic_copy(onnx_path, best_onnx)
        log('首次评估：以当前模型初始化最佳模型 %s' % best_path)

    # 引擎与训练隔离：快照放 eval_work/{cur,best}/，各自独立 trt_cache（否则互相拆台、每轮全量重建）
    cur_dir = os.path.join(EVAL_WORK_DIR, 'cur')
    best_dir = os.path.join(EVAL_WORK_DIR, 'best')
    os.makedirs(cur_dir, exist_ok=True)
    os.makedirs(best_dir, exist_ok=True)
    snapshot = os.path.join(cur_dir, SNAPSHOT_ONNX)                 # 本轮冻结的当前模型
    best_snapshot = os.path.join(best_dir, os.path.basename(best_onnx))
    if os.path.exists(onnx_path):
        _atomic_copy(onnx_path, snapshot)
    if os.path.exists(best_onnx):
        _atomic_copy(best_onnx, best_snapshot)

    os.makedirs(data_dir, exist_ok=True)

    # ---- 评估前诊断 + 兼容性校验 ----
    # 两个模型 channels/blocks 允许不同：评估是纯强度比较（冠军追踪），只要 I/O
    # 接口一致即可对战——输入 [B,6,n,n] → 输出 policy[B,n²+1] / value[B,1] /
    # ownership[B,1,n,n]；TRT 引擎按各自 ONNX 独立构建（eval_work/{cur,best} 各自
    # 独立 trt_cache）。但 棋盘尺寸 / 输入通道 不一致会导致 ORT/TRT 晦涩的形状错误
    # 崩溃，这里提前明确报错（同时打印双方架构，便于判断比较是否合理）。
    n = board_size
    expect_in = [INPUT_CHANNELS, n, n]
    expect_out = {'policy': [n * n + 1], 'value': [1], 'ownership': [1, n, n]}
    for tag, p in (('当前', snapshot), ('最佳', best_snapshot)):
        if not os.path.exists(p):
            log('[错误] %s模型 ONNX 缺失: %s' % (tag, p))
            raise SystemExit('评估中止：%s模型 ONNX 文件不存在' % tag)
        info = _inspect_onnx(p)
        if info is None or 'error' in info:
            log('%s模型 ONNX 解析失败: %s' % (tag, (info or {}).get('error', '未知')))
            continue
        arch = 'channels=%s blocks=%s' % (info['channels'], info['blocks']) \
            if info['channels'] is not None else '架构未知'
        log('%s模型: %s (%s)' % (tag, arch, os.path.basename(p)))
        ins = dict(info['inputs'])
        outs = dict(info['outputs'])
        bad = []
        ishape = ins.get('input')
        if ishape and ishape[1:] != expect_in:
            bad.append('输入 %s ≠ 期望 [B,%s]' % (ishape, ','.join(map(str, expect_in))))
        for oname, eshape in expect_out.items():
            oshape = outs.get(oname)
            if oshape and oshape[1:] != eshape:
                bad.append('输出 %s %s ≠ 期望 [B,%s]' % (oname, oshape, ','.join(map(str, eshape))))
        if bad:
            log('[错误] %s模型与本轮评估不兼容：%s' % (tag, '；'.join(bad)))
            raise SystemExit('评估中止：%s模型形状与 board_size=%d 不兼容' % (tag, n))

    # 整轮复用同一对 MCTS（避免每局重建 TRT/ORT 会话）
    mcts = MCTS(c_puct=c_puct, num_simulations=num_simulations,
                temperature=temperature, onnx_path=snapshot, device=device,
                board_size=board_size, provider=provider)
    mcts_b = MCTS(c_puct=c_puct, num_simulations=num_simulations,
                  temperature=temperature, onnx_path=best_snapshot, device=device,
                  board_size=board_size, provider=provider)

    wins = 0
    for i in range(eval_games):
        try:
            if i % 2 == 0:      # 当前模型执黑
                data = play_one_game(board_size=board_size,
                                     num_simulations=num_simulations,
                                     device=device, c_puct=c_puct,
                                     temperature=temperature,
                                     temperature_zero_after=None,
                                     exploration=False,
                                     onnx_path=snapshot,
                                     model_b=best_snapshot,
                                     mcts=mcts, mcts_b=mcts_b)
                win = (data[4] == 1)     # 黑胜 = 当前模型胜
                side = '黑'
            else:               # 当前模型执白
                data = play_one_game(board_size=board_size,
                                     num_simulations=num_simulations,
                                     device=device, c_puct=c_puct,
                                     temperature=temperature,
                                     temperature_zero_after=None,
                                     exploration=False,
                                     onnx_path=best_snapshot,
                                     model_b=snapshot,
                                     mcts=mcts_b, mcts_b=mcts)
                win = (data[4] == -1)    # 白胜 = 当前模型胜
                side = '白'
            if save_data:
                _save_game(data, data_dir)
            _send_frame(data)            # 回传 GUI 对局回放
            if data[4] == 0:
                log('第%d/%d局 当前模型执%s: 和棋' % (i + 1, eval_games, side))
            else:
                pts = '%.1f' % abs(data[5])
                if pts.endswith('.0'):
                    pts = pts[:-2]
                log('第%d/%d局 当前模型执%s: %s（%s目）' % (i + 1, eval_games, side, '胜' if win else '负', pts))
                if win:
                    wins += 1
        except Exception as e:
            log('第%d/%d局异常（跳过继续）: %s: %s' % (i + 1, eval_games, type(e).__name__, e))

    threshold = int(eval_games * win_rate + 0.5)
    if wins >= threshold:
        _atomic_copy(model_path, best_path)
        if os.path.exists(snapshot):
            _atomic_copy(snapshot, best_onnx)
        log('当前模型 %d/%d 胜，更新最佳模型' % (wins, eval_games))
    else:
        log('当前模型 %d/%d 胜，未达标(%d胜)，保留原最佳模型' % (wins, eval_games, threshold))
    return wins, eval_games


def main():
    ap = argparse.ArgumentParser(
        description='一次性评估：当前模型 vs 最佳模型（N 局后退出，无循环、无日志文件）',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--games', type=int, default=EVAL_GAMES, help='评估对局数')
    ap.add_argument('--sims', type=int, default=NUM_SIMULATIONS, help='每步模拟数')
    ap.add_argument('--temp', type=float, default=EVAL_TEMPERATURE, help='评估温度（近贪心，减小胜负噪声）')
    ap.add_argument('--threshold', type=float, default=EVAL_WIN_RATE, help='更新最佳模型的胜率门槛')
    ap.add_argument('--board-size', type=int, default=BOARD_SIZE, help='棋盘尺寸')
    ap.add_argument('--device', default='cuda', help='推理设备')
    ap.add_argument('--provider', default='auto',
                    help='ONNX provider: auto(默认，TRT优先) / cuda(CUDA EP) / cpu')
    ap.add_argument('--model', default=MODEL_PATH, help='当前模型 .pt 路径')
    ap.add_argument('--best', default=BEST_MODEL_PATH, help='最佳模型 .pt 路径')
    ap.add_argument('--data-dir', default=DEFAULT_DATA_DIR,
                    help='对局数据保存目录（训练 load_all 会读取；与训练数据目录一致，默认 data/）')
    ap.add_argument('--no-save-data', action='store_true', help='不保存对局数据')
    args = ap.parse_args()
    evaluate_and_update_best(model_path=args.model, best_path=args.best,
                             board_size=args.board_size, num_simulations=args.sims,
                             c_puct=C_PUCT, temperature=args.temp,
                             eval_games=args.games, win_rate=args.threshold,
                             device=args.device, data_dir=args.data_dir,
                             provider=args.provider, save_data=not args.no_save_data)
    _frame_out.flush()


if __name__ == '__main__':
    main()
