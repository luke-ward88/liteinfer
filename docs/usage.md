# 使用指南

本页给出每个子模块的常见用法。所有示例只依赖 `numpy`，可直接运行。

## 量化

### 逐通道 INT8

```python
import numpy as np
from liteinfer.quant.int8 import quantize_symmetric_per_channel, dequantize

w = np.random.randn(64, 256).astype("float32")
q, params = quantize_symmetric_per_channel(w, axis=0)  # 每个输出通道一个 scale
w_hat = dequantize(q, params)
print(np.abs(w - w_hat).max())
```

### 分组 INT4

```python
from liteinfer.quant.int4 import quantize_group, dequantize_group

x = np.random.randn(1024).astype("float32")
packed = quantize_group(x, group_size=64)   # 每 64 个元素共享 scale/zero
print(packed.data.nbytes)                    # 打包后的字节数 ≈ 元素数 / 2
x_hat = dequantize_group(packed)
```

分组越小，量化误差越低，但存储的 `scale/zero` 也越多——这是精度与体积的权衡。

### 用校准数据确定量化范围

```python
from liteinfer.quant.calibration import calibrate
from liteinfer.quant.observer import PercentileObserver

batches = [np.random.randn(16, 128).astype("float32") for _ in range(10)]
params = calibrate(batches, bits=8, symmetric=False,
                   observer=PercentileObserver(percentile=0.1))
```

`PercentileObserver` 会裁掉极端离群值，让 scale 更贴合数据主体。

## KV-cache

```python
from liteinfer.kvcache.blocks import BlockAllocator
from liteinfer.kvcache.manager import BlockSpaceManager
from liteinfer.kvcache.paged_cache import PagedKVStore
from liteinfer.kvcache.attention import paged_attention

store = PagedKVStore(num_blocks=32, block_size=8, num_heads=4, head_dim=16)
mgr = BlockSpaceManager(BlockAllocator(num_blocks=32, block_size=8))
mgr.add_sequence(0)

for _ in range(20):                       # 逐 token 写入
    block_id, slot = mgr.append_token(0)
    store.write(block_id, slot, k, v)     # k, v 形状为 (num_heads, head_dim)

out = paged_attention(query, 0, store, mgr)  # 对已缓存的序列做一步注意力
mgr.free_sequence(0)                         # 序列结束，归还所有块
```

多个序列可以共享同一个物理块池，`BlockAllocator.num_free` 反映剩余容量。当块用尽时会抛出 `CacheFullError`。

## 投机解码

```python
import numpy as np
from liteinfer.specdec.ngram import make_draft_target
from liteinfer.specdec.speculative import speculative_decode
from liteinfer.specdec.metrics import summarize

draft, target = make_draft_target(vocab_size=128, order=3, seed=0,
                                  draft_temperature=1.3)
res = speculative_decode([0, 1], draft, target, gamma=4,
                         max_new_tokens=64, rng=np.random.default_rng(0))

print(res.acceptance_rate)
print(summarize(res, gamma=4))
```

`draft_temperature` 越接近 1，draft 与 target 越一致、接受率越高；调大它可以人为制造更“弱”的 draft，用来观察校正机制的行为。

## 命令行

```bash
liteinfer quant --rows 128 --cols 512
liteinfer kv --length 40 --block-size 8
liteinfer spec --gamma 4 --tokens 64
liteinfer bench --max-gamma 6
```
