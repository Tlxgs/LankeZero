"""Build/cache a TensorRT REFIT engine for one board size.

Build once per board size; afterwards training only refits weights into it.
Run this when training is stopped.

Example:
  ..\\venv\\Scripts\\python.exe python\\build_trt.py --board-size 19
"""
import argparse
import os
import shutil
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

from hyperparams import BOARD_SIZES, BATCH_SIZE_MCTS, MODEL_PATH  # noqa: E402
from model import PolicyValueNet  # noqa: E402
import trt_engine  # noqa: E402


def _load_model_copy(model_path):
    """Load from a copy so the long build never holds model.pt open."""
    fd, tmp = tempfile.mkstemp(prefix="trt_model_", suffix=".pt")
    os.close(fd)
    try:
        shutil.copy2(model_path, tmp)
        model = PolicyValueNet.load_model(tmp, device="cpu")
        return model.eval()
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser(description="构建 TensorRT REFIT 引擎（按棋盘尺寸缓存）")
    ap.add_argument("--board-size", type=int, choices=BOARD_SIZES, default=19)
    ap.add_argument("--batch", type=int, default=BATCH_SIZE_MCTS)
    ap.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    ap.add_argument("--engine-dir", default=None,
                    help="默认 trt_engines/（或 GO_TRT_ENGINE_DIR）")
    ap.add_argument("--model", default=MODEL_PATH)
    ap.add_argument("--force", action="store_true", help="覆盖已有引擎")
    ap.add_argument("--check", action="store_true",
                    help="只检查缓存引擎是否与当前模型/环境匹配")
    args = ap.parse_args()

    if args.batch != 16:
        print("[警告] 训练 MCTS batch 固定为 16；这里建 %d 的引擎不会在训练中使用"
              % args.batch)
    if not os.path.exists(args.model):
        raise SystemExit("权重不存在: %s" % args.model)
    if trt_engine.import_tensorrt() is None:
        raise SystemExit(
            "TensorRT 不可用: %s\n"
            "请设置 GO_TRT_LIB_DIR（含 nvinfer/cuDNN 8 DLL 的目录）和 "
            "GO_TRT_PYTHONPATH（含 tensorrt 包的目录），"
            "或安装匹配的 tensorrt wheel。" % trt_engine.last_error())

    model = _load_model_copy(args.model)
    directory = trt_engine.engine_dir(args.engine_dir)
    engine_path, manifest_path = trt_engine.engine_paths(
        directory, args.board_size, args.batch, args.precision)

    if args.check:
        if not (os.path.exists(engine_path) and os.path.exists(manifest_path)):
            print("未找到缓存引擎: %s" % engine_path)
            return 1
        manifest = trt_engine._read_manifest(manifest_path)
        problem = trt_engine.manifest_problem(
            manifest, model, args.board_size, args.batch, args.precision)
        if problem:
            print("缓存引擎不匹配: %s" % problem)
            return 1
        print("缓存引擎可用: %s" % engine_path)
        print("  TensorRT %s | %s | %s | %d 个权重"
              % (manifest.get("tensorrt_version"), manifest.get("gpu"),
                 manifest.get("precision"), len(manifest.get("weight_names", []))))
        return 0

    if os.path.exists(engine_path) and not args.force:
        print("引擎已存在: %s" % engine_path)
        print("如需重建请加 --force；训练时会直接加载并 refit。")
        return 0

    print("构建 %d 路 batch=%d %s REFIT 引擎，请确认训练已停止..."
          % (args.board_size, args.batch, args.precision))
    trt_engine.build_engine(
        model, args.board_size, engine_dir_path=directory,
        batch=args.batch, precision=args.precision, force=args.force,
        log=lambda msg: print(msg, flush=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
