# 设计笔记：投机解码为什么是无损的

投机解码（speculative decoding）最反直觉的一点是：它在**不改变输出分布**的前提下加速采样。这里记录一下为什么。

## 问题

自回归采样每步都要跑一次大模型（target）的前向。如果能用一个便宜的小模型（draft）先猜几步，再让大模型**并行**验证，就有机会一次前向产出多个 token。难点在于：draft 的分布 `q` 和 target 的分布 `p` 并不相同，直接采纳 draft 的输出会引入偏差。

## 接受-拒绝校正

对 draft 提议的 token `x`（它由 `q` 采出），我们以概率

```
α(x) = min(1, p(x) / q(x))
```

接受它。如果被拒绝，就从**残差分布**

```
p_residual(x) = normalize( max(p(x) - q(x), 0) )
```

重新采样一个 token。

**关键结论**：经过这套流程输出的 token，其边缘分布恰好是 `p`。

## 简要证明

考虑任意 token `x`，它最终被输出的方式只有两种，互斥：

1. **它作为 draft 提议且被接受。** 概率为
   `q(x) · α(x) = q(x) · min(1, p(x)/q(x)) = min(q(x), p(x))`。

2. **前一个 draft 提议被拒绝，`x` 从残差里被采出。** 拒绝发生的总概率是
   `β = Σ_y q(y)·(1 - α(y)) = Σ_y (q(y) - min(q(y), p(y))) = Σ_y max(q(y)-p(y), 0)`。
   而残差在 `x` 上的质量是 `max(p(x)-q(x), 0) / Z`，其中归一化常数
   `Z = Σ_y max(p(y)-q(y), 0)`。注意 `Σ_y max(p-q,0) = Σ_y max(q-p,0)`（因为 `Σp = Σq = 1`），所以 `β = Z`，两者相消。第二种方式贡献的概率就是 `max(p(x)-q(x), 0)`。

把两部分相加：

```
min(q(x), p(x)) + max(p(x)-q(x), 0) = p(x)
```

分两种情况即可验证：
- 若 `p(x) ≥ q(x)`：`q(x) + (p(x)-q(x)) = p(x)` ✓
- 若 `p(x) < q(x)`：`p(x) + 0 = p(x)` ✓

于是无论 draft 好坏，输出分布都严格等于 target 的 `p`。draft 的质量只影响**速度**（接受率），不影响**正确性**。

## 一次迭代产出多少 token

我们一次提议 `γ` 个 token，逐个验证。设第一个被拒的位置为 `k`（从 0 计），则：

- 若 `k < γ`：接受前 `k` 个，第 `k` 个用残差替换，共产出 `k + 1` 个 token。
- 若全部接受（`k = γ`）：额外再从 target 采一个 bonus token，共产出 `γ + 1` 个。

因此单次迭代产出 `1 ~ γ+1` 个 token。期望产出随接受率单调上升，这也是为什么 draft 越贴近 target 越划算。

## 实现中的坑

- **接受比要封顶到 1**：`min(1, p/q)`。虽然 `rng.random() < ratio` 在 `ratio > 1` 时也总会接受，但显式封顶更清晰（见 `acceptance.py`）。
- **残差用 ReLU 而非对称裁剪**：必须是 `max(p-q, 0)`，不能是 `|p-q|`，否则校正不再精确。
- **draft 的 `q` 要存下来**：验证时需要 draft 当时的分布，而不是重新用 target 算一遍。
- **温度 / top-k / top-p 要对 draft 和 target 一致地施加**：否则 `p` 和 `q` 不在同一个支撑集上，比值失去意义。

## 参考

- Leviathan et al., *Fast Inference from Transformers via Speculative Decoding*, ICML 2023.
- Chen et al., *Accelerating Large Language Model Decoding with Speculative Sampling*, 2023.
