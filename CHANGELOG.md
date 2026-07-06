# Changelog

本项目遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 与 [语义化版本](https://semver.org/lang/zh-CN/)。

## [Unreleased]

## [0.1.0] - 2026-07-06

首个公开版本。

### Added

- **量化**：对称 / 非对称 / 逐通道 INT8，分组 INT4（含 nibble 打包），`QuantLinear` 层，MinMax / Percentile 观察器与校准循环。
- **KV-cache**：块分配器、块空间管理器、分页 K/V 存储与分页注意力核。
- **投机解码**：接受-拒绝校正、残差重采样、draft/target 主循环，配套 top-k / top-p 采样与接受率指标。
- **玩具模型**：确定性 n-gram 模型，示例与测试可离线运行。
- **命令行**：`quant` / `kv` / `spec` / `bench` 四个子命令。
- 文档：架构、使用、设计笔记、API 参考。

[Unreleased]: https://github.com/luke-ward88/liteinfer/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/luke-ward88/liteinfer/releases/tag/v0.1.0
