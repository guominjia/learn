
## Create Package

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
