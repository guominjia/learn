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

## References
- https://www.python.org/
- https://docs.python.org/3/
- https://donate.python.org/
- https://pypi.org/
- https://github.com/python/