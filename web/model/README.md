# 网页模型

`model.onnx` 是从根目录唯一训练权重 `model.pt` 导出的网页演示模型，不参与训练。
本仓库为 GitHub Pages 演示保留了这个派生文件；训练权重 `model.pt` 本身不入库。

生成/更新：

```bat
pip install -r web/tools/requirements.txt
python web/tools/export_web_model.py
```

模型输入：`state`，形状 `[batch, 10, n, n]`，float32。
模型输出：`policy` `[batch, n*n+1]`、`value` `[batch, 1]`、`win` `[batch, 1]`。
