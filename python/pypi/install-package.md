
## Install Package

```bash
pip install git+https://github.com/org/repo@branch[web]
```

**pip 的行为：**
- pip 将整个 `branch[web]` 作为 git 引用（分支名）传递给 git
- git 尝试 checkout 名为 `branch[web]` 的分支，但这个分支不存在
- 方括号 `[web]` 实际上是 pip 的 **extras 语法**，用于指定可选依赖

```bash
uv pip install git+https://github.com/org/repo@branch[web]
```

**uv pip 的行为：**
- `uv` 对 PEP 508 依赖规范的解析更加准确和严格，能正确处理复杂的 URL 语法
- 能正确将 `[web]` 识别为 extras
- 只将 `branch` 作为 git 引用传递给 git

### 解决`pip`安装`branch[web]`问题

#### 方案 1：使用引号（推荐）
```bash
pip install "git+https://github.com/org/repo@branch#egg=package[web]"
```

#### 方案 2：先安装基础包，再安装 extras
```bash
pip install git+https://github.com/org/repo@branch
pip install package[web]
```

#### 方案 3：使用 URL 编码
```bash
pip install "git+https://github.com/org/repo@branch#egg=package%5Bweb%5D"
```
（`%5B` = `[`, `%5D` = `]`）
