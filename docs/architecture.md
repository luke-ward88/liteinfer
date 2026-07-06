# 架构总览

`liteinfer` 由三个相互独立的子包组成，各自对应一种推理加速技术，彼此之间没有强耦合，只共享一个极薄的数组后端（`liteinfer.backend`）。

```
liteinfer/
├── backend.py          # numpy / torch 数组的薄封装
├── config.py           # QuantConfig / SpecDecodeConfig 参数容器
├── errors.py           # 统一的异常类型
├── cli.py              # 命令行入口（quant / kv / spec / bench）
├── models/             # 离线可跑的玩具 n-gram 模型
├── quant/              # 训练后量化
│   ├── params.py       #   QuantParams：scale / zero_point 容器
│   ├── observer.py     #   MinMax / Percentile 观察器
│   ├── calibration.py  #   校准循环
│   ├── int8.py         #   对称/非对称/逐通道 INT8
│   ├── int4.py         #   nibble 打包 + 分组 INT4
│   └── linear.py       #   QuantLinear：权重量化的线性层
├── kvcache/            # 分页 KV-cache
│   ├── blocks.py       #   BlockAllocator：物理块空闲链表
│   ├── paged_cache.py  #   PagedKVStore：块化的 K/V 存储
│   ├── manager.py      #   BlockSpaceManager：序列 → 块表
│   └── attention.py    #   paged_attention：分页注意力核
└── specdec/            # 投机解码
    ├── sampler.py      #   softmax / top-k / top-p / rng
    ├── acceptance.py   #   接受-拒绝检验 + 残差分布
    ├── speculative.py  #   draft/target 主循环
    ├── ngram.py        #   构造 draft/target 模型对
    └── metrics.py      #   接受率、加速比估计
```

## 三个子系统

### 量化（quant）

量化的核心是把浮点张量映射到低位宽整数：`q = round(x / scale) + zero_point`。

- **对称**方案令 `zero_point = 0`，`scale = max(|x|) / qmax`，适合以 0 为中心的权重。
- **非对称**方案把 `[min, max]` 整体映射到 `[0, 2^bits-1]`，更适合激活。
- **逐通道**为每个输出通道单独计算 `scale`，显著降低权重量化误差。
- **INT4** 采用分组量化：每 `group_size` 个元素共享一组 `scale/zero`，再用 nibble 打包（两个 4-bit 塞进一个字节）。

`QuantLinear` 把这些组合起来：权重按逐通道 INT8 存储，前向时即时反量化再做矩阵乘。

### KV-cache 分页（kvcache）

自回归解码时，KV-cache 随序列增长。连续分配会造成碎片与浪费。分页方案把显存切成固定大小的**块**，每个序列维护一张**块表**记录它占用了哪些物理块：

```
逻辑位置 pos  ->  块表[pos // block_size] 的第 (pos % block_size) 个槽位
```

`BlockAllocator` 用一个空闲链表管理物理块；`BlockSpaceManager` 按需为序列追加块，序列结束时归还。`paged_attention` 通过块表把分散在各块中的 K/V 收集（gather）成连续视图，再做标准的缩放点积注意力——分页对注意力的数学是完全透明的。

### 投机解码（specdec）

见[设计笔记](design-notes.md)。要点：draft 模型便宜地提议 `gamma` 个 token，target 模型一次性打分，再用接受-拒绝校正保留一段前缀。校正保证最终输出分布与 target 严格一致。

## 数据流

一次投机解码迭代：

```
prefix ──► draft 自回归提议 γ 个 token（记录 q 分布）
        ──► target 对每个位置打分（得到 p 分布）
        ──► 逐个做 min(1, p/q) 接受检验
              ├─ 全部接受 ──► 再从 target 采样 1 个 bonus token
              └─ 第 i 个被拒 ──► 从残差 max(p-q,0) 重采样，丢弃其余
```

每次迭代产出 1 ~ γ+1 个 token。
