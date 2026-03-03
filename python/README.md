# Python

## Python Package

### License Declaration

```text
# Option 1: Direct license identifier (if using a standard license)
license = {text = "MIT"}

# Option 2: SPDX identifier
license = "MIT"

# Option 3: File reference
license = {file = "LICENSE"}
```

### Multiple Package
```text
[tool.hatch.build.targets.wheel]
packages = ["src/package1", "src/package2"]
```

### Install Package

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

## pyyaml

Below will be `None` in python
- `null/Null/NULL/~` or `key: ` <- empty

Below will be **boolean**, If need string, should use `"True"` rather than `True`
- `true/True/TRUE/yes/Yes/YES/on/On/ON`
- `false/False/FALSE/no/No/NO/off/Off/OFF`

Below distinguish dict and string
- key: value in which `:` followed by **space** will be dict
- key:value like `https:` will be string

https://example.com/ 中的冒号 : 后面有斜杠 //，YAML 解析器会将整个值识别为字符串

## Regex
```python
matches = re.finditer(r"### \d+\. `(.*?)` - (.*?)\n\n```\w+\n(.*?)```", raw, re.MULTILINE | re.DOTALL)
for match in matches:
    file, statment, content = match.group(1), match.group(2), match.group(3)
    os.makedirs(os.path.dirname(file), exist_ok=True)
    with open(file, "w", encoding="utf8") as f:
        f.write(content)
```

## [Networkx](https://pypi.org/project/networkx/)
```python
import networkx as nx

G = nx.Graph()
G.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 1)])

subgraph = nx.subgraph(G, [1, 2, 3])

degrees = [v.degree() for v in subgraph.nodes()]
print(degrees)  # Output: [2, 2, 2]
```

## [Fabric](https://pypi.org/project/fabric/)
Fabric is a high level Python library designed to execute shell commands remotely over SSH, yielding useful Python objects in return.
It builds on top of Invoke (subprocess command execution and command-line features) and Paramiko (SSH protocol implementation), extending their APIs to complement one another and provide additional functionality.

## References
- https://www.python.org/
- https://docs.python.org/3/
- https://donate.python.org/
- https://pypi.org/
- https://github.com/python/