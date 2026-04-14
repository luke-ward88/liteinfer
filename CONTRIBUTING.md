# 贡献指南

欢迎 issue 和 PR。这个项目以「读得懂的参考实现」为目标，所以清晰 > 极致性能。

## 开发环境

```bash
git clone https://github.com/luke-ward88/liteinfer.git
cd liteinfer
pip install -e ".[dev]"
pre-commit install   # 可选，自动跑 ruff
```

## 提交前自查

请确保以下命令都通过（CI 也会检查同样的内容）：

```bash
ruff check .
ruff format --check .
mypy
pytest --cov=liteinfer
```

## 代码风格

- 行宽 100，缩进 4 空格（见 `.editorconfig`）。
- 公开函数写类型注解和 docstring；注释只解释「为什么」，不复述「做了什么」。
- 新增内核请配套测试：数值内核要有和参考实现对拍的用例，采样类要有分布统计用例。

## 提交信息

推荐 [Conventional Commits](https://www.conventionalcommits.org/) 风格（`feat:` / `fix:` / `docs:` / `refactor:` …），但不强制。一个提交只做一件事。

## 分支与 PR

- 从 `main` 切分支开发。
- PR 请填写模板里的自查清单，并关联相关 issue。
- 涉及行为变化时，记得更新文档和 `CHANGELOG.md`。
