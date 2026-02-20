# Compiler

## Parser

| 工具/语言 | 适用语言 | 是否需要GCC | 是否需要Clang |
|----------|----------|-------------|----------------|
| `ast`    | Python   | 否          | 否             |
| `pycparser` | C/C       | 否          | 否             |
| `cindex` | C/C      | 否          | 是（Clang）    |
| `javalang` | Java     | 否          | 否             |
| `gcc`    | C/C      | 是（命令行）| 否（需中间表示）|

```python
import clang.cindex
print(clang.cindex.Config.get_library_path())
```