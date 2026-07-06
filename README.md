# liteinfer

[![CI](https://github.com/luke-ward88/liteinfer/actions/workflows/ci.yml/badge.svg)](https://github.com/luke-ward88/liteinfer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**大模型高效推理工具包 —— 纯 Python 参考实现。**

`liteinfer` 把三种在生产推理系统里最常见的加速手段，拆成**读得懂、跑得起来、可验证**的参考内核：

- **训练后量化（PTQ）**：对称 / 非对称 INT8、逐通道量化、INT4 分组量化（含 nibble 打包）。
- **KV-cache 分页管理**：块分配器、块表（block table）、分页注意力（paged attention），思路对齐 vLLM 的 PagedAttention。
- **投机解码（speculative decoding）**：draft/target 双模型 + 精确的接受/拒绝采样，保证输出分布与目标模型**严格一致**。

设计目标是**教学与实验**，不是替代 vLLM / TensorRT-LLM。所有内核默认只依赖 `numpy`，离线即可运行；PyTorch 是可选后端。

## 安装

```bash
pip install -e .
# 可选：装上开发依赖（pytest / ruff / mypy）
pip install -e ".[dev]"
```

要求 Python ≥ 3.9，唯一的运行时依赖是 `numpy`。

## 快速上手

三个子模块都能独立使用，命令行也提供了对应的 demo：

```bash
liteinfer quant     # 权重量化的误差与显存节省
liteinfer kv        # 分页 KV-cache + 分页注意力
liteinfer spec      # 投机解码接受率
liteinfer bench     # 扫描 gamma，观察接受率随提议长度的变化
```

### 量化一个线性层

```python
import numpy as np
from liteinfer import QuantLinear

weight = np.random.randn(256, 512).astype("float32")
layer = QuantLinear(weight)          # INT8 逐通道量化
y = layer(np.random.randn(8, 512))   # 前向照常调用
print(layer.memory_bytes())          # 约为 fp32 的 1/4
```

### 投机解码

```python
import numpy as np
from liteinfer.specdec.ngram import make_draft_target
from liteinfer.specdec.speculative import speculative_decode

draft, target = make_draft_target(vocab_size=128, seed=0)
res = speculative_decode([0, 1, 2], draft, target, gamma=4, max_new_tokens=64,
                         rng=np.random.default_rng(0))
print(res.acceptance_rate)   # 被接受的提议比例
```

## 文档

- [架构总览](docs/architecture.md)
- [使用指南](docs/usage.md)
- [设计笔记：投机解码为什么是无损的](docs/design-notes.md)
- [API 参考](docs/api-reference.md)

## 开发

```bash
ruff check . && ruff format --check .   # 代码风格
mypy                                    # 类型检查
pytest --cov=liteinfer                  # 测试 + 覆盖率
```

## 许可证

[MIT](LICENSE)
