# UV - Python Package and Project Manager

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Package Management](#package-management)
5. [Project Management](#project-management)
6. [Python Version Management](#python-version-management)
7. [Virtual Environments](#virtual-environments)
8. [Configuration](#configuration)
9. [Advanced Usage](#advanced-usage)
10. [Migration from Other Tools](#migration-from-other-tools)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)
13. [Resources](#resources)

## Introduction

UV is an extremely fast Python package and project manager written in Rust. It's designed as a drop-in replacement for pip, pip-tools, pipx, poetry, pyenv, virtualenv, and more, offering significant performance improvements and a unified interface for Python development workflows.

**Key Features:**
- **Speed**: 10-100x faster than pip for package operations
- **Unified Interface**: Replace multiple tools with a single command
- **Python Management**: Built-in Python version management
- **Zero Dependencies**: Single binary installation
- **Lock Files**: Deterministic dependency resolution
- **Cross-Platform**: Works on Windows, macOS, and Linux

## Installation

### Install UV

**Unix/macOS:**
```bash
# Using the installer script
curl -LsSf https://astral.sh/uv/install.sh | sh

# Using Homebrew
brew install uv

# Using pipx
pipx install uv
```

**Windows:**
```powershell
# Using PowerShell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Using Chocolatey
choco install uv

# Using Scoop
scoop install uv
```

**Verify Installation:**
```bash
uv --version
```

## Quick Start

### Create a New Project
```bash
# Create a new Python project
uv init my-project
cd my-project

# Create project with specific Python version
uv init --python 3.12 my-project
```

### Add Dependencies
```bash
# Add a package
uv add requests

# Add development dependency
uv add --dev pytest

# Add package with version constraint
uv add "django>=4.0,<5.0"
```

### Run Your Project
```bash
# Run a script
uv run main.py

# Run with arguments
uv run main.py --verbose

# Run a module
uv run -m pytest
```

## Package Management

### Installing Packages
```bash
# Install packages from requirements.txt
uv pip install -r requirements.txt

# Install a single package
uv pip install requests

# Install with constraints
uv pip install "requests>=2.0"

# Install from Git
uv pip install git+https://github.com/user/repo.git

# Install editable package
uv pip install -e .
```

### Managing Dependencies
```bash
# Generate lock file
uv lock

# Sync environment with lock file
uv sync

# Update dependencies
uv lock --upgrade

# Update specific package
uv lock --upgrade-package requests
```

### Creating Requirements
```bash
# Generate requirements.txt
uv pip freeze > requirements.txt

# Compile requirements with constraints
uv pip compile requirements.in

# Compile with specific Python version
uv pip compile --python-version 3.12 requirements.in
```

## Project Management

### Project Structure
UV projects follow this structure:
```
my-project/
├── pyproject.toml      # Project configuration
├── uv.lock            # Lock file
├── README.md
├── src/
│   └── my_project/
│       └── __init__.py
└── tests/
    └── test_main.py
```

### pyproject.toml Configuration
```toml
[project]
name = "my-project"
version = "0.1.0"
description = "My Python project"
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
dependencies = [
    "requests>=2.25.0",
]
requires-python = ">=3.8"

[project.optional-dependencies]
dev = [
    "pytest>=6.0",
    "black",
    "flake8",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.uv]
dev-dependencies = [
    "pytest>=6.0",
    "black",
    "flake8",
]
```

### Project Commands
```bash
# Initialize existing directory
uv init

# Add dependency to project
uv add numpy pandas

# Remove dependency
uv remove numpy

# Show project info
uv tree

# Build project
uv build

# Publish to PyPI
uv publish
```

## Python Version Management

### Installing Python Versions
```bash
# List available Python versions
uv python list

# Install specific Python version
uv python install 3.12

# Install multiple versions
uv python install 3.11 3.12 3.13

# Install latest patch version
uv python install 3.12+
```

### Using Python Versions
```bash
# Create project with specific Python
uv init --python 3.12 my-project

# Change Python version for existing project
uv python pin 3.11

# Run with specific Python version
uv run --python 3.12 script.py
```

## Virtual Environments

### Creating Virtual Environments
```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.12

# Create with custom name
uv venv .venv-dev

# Create in specific location
uv venv /path/to/venv
```

### Using Virtual Environments
```bash
# Activate virtual environment (Unix)
source .venv/bin/activate

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Run command in virtual environment
uv run python script.py

# Install packages in active environment
uv pip install requests
```

## Configuration

### Configuration Files
UV can be configured through:
- `pyproject.toml` (project-specific)
- `uv.toml` (project-specific)
- Global configuration file

### Example uv.toml
```toml
[pip]
index-url = "https://pypi.org/simple"
extra-index-url = ["https://private-registry.com/simple"]
trusted-host = ["private-registry.com"]

[tool.uv]
cache-dir = "~/.cache/uv"
```

### Environment Variables
```bash
# Set index URL
export UV_INDEX_URL=https://pypi.org/simple

# Set cache directory
export UV_CACHE_DIR=~/.cache/uv

# Disable progress bar
export UV_NO_PROGRESS=1
```

## Advanced Usage

### Working with Lock Files
```bash
# Generate lock file without installing
uv lock --no-install

# Update lock file
uv lock --upgrade

# Lock with specific platform
uv lock --python-platform linux

# Export to requirements format
uv export --format requirements-txt --output-file requirements.txt
```

### Custom Indexes and Registries
```bash
# Use custom index
uv pip install --index-url https://custom-index.com/simple package

# Add extra index
uv pip install --extra-index-url https://extra-index.com/simple package

# Install from private registry
uv add --index-url https://private-registry.com/simple private-package
```

### Dependency Resolution
```bash
# Resolve dependencies without installing
uv pip compile requirements.in --dry-run

# Show dependency tree
uv tree

# Check for dependency conflicts
uv pip check
```

## Migration from Other Tools

### From pip + virtualenv
```bash
# Old way
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# UV way
uv venv
uv pip install -r requirements.txt
```

### From Poetry
```bash
# Convert pyproject.toml
uv add $(grep -E '^[a-zA-Z0-9_-]+' pyproject.toml | cut -d '=' -f1)

# Or initialize from poetry.lock
uv sync
```

### From pipenv
```bash
# Convert Pipfile
uv pip install -r <(pipenv requirements)

# Or create new project
uv init --python $(python --version | cut -d' ' -f2)
```

## Best Practices

### Project Setup
- Always use `uv init` for new projects
- Pin Python version in `pyproject.toml`
- Use lock files for reproducible builds
- Separate development and production dependencies

### Dependency Management
- Use version constraints (`>=`, `~=`, `==`)
- Keep dependencies minimal
- Regular dependency updates with `uv lock --upgrade`
- Use `uv tree` to understand dependency relationships

### Performance Optimization
- Use `uv sync` instead of `pip install` for existing projects
- Leverage UV's caching with `UV_CACHE_DIR`
- Use `--no-deps` when installing known-compatible packages

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv sync --frozen

- name: Run tests
  run: uv run pytest
```

## Troubleshooting

### Common Issues

**Slow installation:**
```bash
# Clear cache
uv cache clean

# Use different index
uv pip install --index-url https://pypi.org/simple package
```

**Python version conflicts:**
```bash
# Check available Python versions
uv python list

# Install required Python version
uv python install 3.12
```

**Lock file issues:**
```bash
# Regenerate lock file
rm uv.lock
uv lock

# Update specific dependency
uv lock --upgrade-package requests
```

### Debugging Commands
```bash
# Verbose output
uv --verbose pip install package

# Show UV configuration
uv --help

# Check project status
uv tree
uv pip check
```

## Resources

- [Official Documentation](https://docs.astral.sh/uv/)
- [Migration Guide](https://docs.astral.sh/uv/guides/integration/)
- [UV vs Other Tools Comparison](https://docs.astral.sh/uv/pip/compatibility/)
- [GitHub Repository](https://github.com/astral-sh/uv)
- [Python Packaging Guide](https://packaging.python.org/)