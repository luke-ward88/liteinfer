# API 参考

只列出稳定的公开接口。带 `_` 前缀的名字视为内部实现，可能随时变动。

## 顶层导出（`liteinfer`）

| 名称 | 说明 |
|---|---|
| `__version__` | 版本号字符串 |
| `QuantConfig` | 量化参数容器（`bits` / `symmetric` / `per_channel` / `group_size`） |
| `SpecDecodeConfig` | 投机解码参数容器（`gamma` / `temperature` / `top_k` / `top_p` / …） |
| `QuantLinear` | 权重量化的线性层 |
| `quantize_symmetric` / `dequantize` | 对称 INT8 量化 / 反量化 |
| `quantize_group` / `dequantize_group` | 分组 INT4 量化 / 反量化 |
| `PagedKVStore` | 分页 K/V 存储 |
| `BlockAllocator` | 物理块分配器 |
| `BlockSpaceManager` | 序列到块表的管理器 |
| `paged_attention` | 分页注意力核 |
| `speculative_decode` / `SpecDecodeResult` | 投机解码主循环及其结果 |

## `liteinfer.quant`

### `quantize_symmetric(x) -> (codes, QuantParams)`
逐张量对称 INT8 量化。`scale = max(|x|) / 127`，`zero_point = 0`。

### `quantize_symmetric_per_channel(x, axis=0) -> (codes, QuantParams)`
逐通道对称量化，沿 `axis` 每个通道一个 `scale`。

### `quantize_asymmetric(x) -> (codes, QuantParams)`
逐张量非对称 UINT8 量化，把 `[min, max]` 映射到 `[0, 255]`。

### `dequantize(codes, params) -> ndarray`
上述任一量化的逆操作，自动区分逐张量 / 逐通道。

### `quantize_group(x, group_size=64) -> PackedInt4`
一维数组的分组 INT4 量化，返回打包后的 `PackedInt4`（含 `data` / `scale` / `zero` / `group_size` / `length`）。

### `dequantize_group(packed) -> ndarray`
`quantize_group` 的逆操作，按 `length` 裁掉尾部填充。

### `QuantLinear(weight, bias=None)`
- `__call__(x) -> ndarray`：`y = x @ W.T + b`，权重以逐通道 INT8 存储。
- `weight`：即时反量化得到的权重。
- `memory_bytes() -> int`：量化权重 + scale 占用的字节数。

### `calibrate(batches, *, bits=8, symmetric=True, observer=None) -> QuantParams`
遍历校准批次，用观察器统计范围并导出量化参数。观察器可选 `MinMaxObserver` 或 `PercentileObserver`。

## `liteinfer.kvcache`

### `BlockAllocator(num_blocks, block_size)`
- `allocate() -> int`：取一个空闲块，池空时抛 `CacheFullError`。
- `free(block_id)` / `free_many(ids)`：归还块。
- `num_free`：剩余空闲块数。

### `BlockSpaceManager(allocator)`
- `add_sequence(seq_id)`：登记一个新序列。
- `append_token(seq_id) -> (block_id, slot)`：为下一个 token 预留槽位，按需扩块。
- `slot_for(seq_id, pos) -> (block_id, slot)`：查已有位置的物理槽位。
- `free_sequence(seq_id)`：归还该序列的全部块。

### `PagedKVStore(num_blocks, block_size, num_heads, head_dim, dtype=float32)`
- `write(block_id, slot, k, v)`：写入单个 token 的 K/V。
- `gather(block_table, length) -> (keys, values)`：按块表收集为连续视图。

### `paged_attention(query, seq_id, store, manager) -> ndarray`
对已缓存序列做一步分页注意力，返回 `(num_heads, head_dim)` 的输出。

## `liteinfer.specdec`

### `speculative_decode(prefix, draft, target, *, gamma=4, max_new_tokens=32, rng=None) -> SpecDecodeResult`
draft/target 双模型投机解码。模型只需实现 `next_token_probs(prefix) -> ndarray`。

`SpecDecodeResult` 字段：`tokens` / `proposed` / `accepted`，属性 `acceptance_rate`。

### `accept_token(token, p, q, rng) -> bool`
接受-拒绝检验，接受概率为 `min(1, p[token] / q[token])`。

### `residual_distribution(p, q) -> ndarray` / `resample_on_reject(p, q, rng) -> int`
残差分布 `normalize(max(p-q, 0))` 及其采样。

### `softmax` / `top_k_filter` / `top_p_filter` / `apply_filters` / `sample`
采样工具函数，见 `sampler.py`。

### `make_draft_target(vocab_size, *, order=3, seed=0, draft_temperature=1.5) -> (draft, target)`
基于玩具 n-gram 模型构造一对 draft/target。
