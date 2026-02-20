# Python

## Python Package

### License Declaration

```text
# Option 1: Direct license identifier (if using a standard license)
license = {text = "MIT"}

# Option 2: SPDX identifier
license = "MIT"

# Option 3: File reference (what you currently have)
license = {file = "LICENSE"}
```

### Multiple Package
```text
[tool.hatch.build.targets.wheel]
packages = ["src/package1", "src/package2"]
```

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

## Regular
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

## References
- https://www.python.org/
- https://docs.python.org/3/
- https://donate.python.org/
- https://pypi.org/
- https://github.com/python/