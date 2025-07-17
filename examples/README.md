# 示例

这里的每个脚本都只依赖 numpy，可离线直接运行：

```bash
python examples/quantize_linear.py     # 权重量化误差与显存节省
python examples/paged_kvcache.py       # 分页 KV-cache + 分页注意力
python examples/speculative_decode.py  # 投机解码接受率
```

也可以直接用命令行：

```bash
liteinfer quant
liteinfer kv
liteinfer spec
liteinfer bench
```
