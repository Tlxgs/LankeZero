"""导出浏览器演示用的 ONNX 模型。

这是从唯一权重 model.pt 派生的演示产物，不参与训练；默认写到 web/model/model.onnx，
该文件被 .gitignore 的 *.onnx 排除。网页上传时请把它放到 GitHub Release（或你自己
的静态文件地址），不要把权重提交进仓库。

用法（仓库根目录）：
  ..\venv\Scripts\python.exe web\tools\export_web_model.py
"""
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))

from model import PolicyValueNet  # noqa: E402


class WebModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, state):
        policy, value, _ownership, win = self.model(state)
        return policy, value, win


def main():
    model_path = ROOT / "model.pt"
    output = ROOT / "web" / "model" / "model.onnx"
    output.parent.mkdir(parents=True, exist_ok=True)
    model = PolicyValueNet.load_model(str(model_path), device="cpu").eval()
    wrapper = WebModel(model).eval()
    dummy = torch.zeros(1, 10, 19, 19, dtype=torch.float32)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(output),
        opset_version=17,
        input_names=["state"],
        output_names=["policy", "value", "win"],
        dynamic_axes={
            "state": {0: "batch", 2: "height", 3: "width"},
            "policy": {0: "batch"},
            "value": {0: "batch"},
            "win": {0: "batch"},
        },
        do_constant_folding=True,
    )
    print(f"written: {output} ({output.stat().st_size / 1024 / 1024:.1f} MiB)")


if __name__ == "__main__":
    main()
