# Pip Package Index Configuration Guide

This guide explains how pip searches for packages across different repositories and how to configure multiple package indexes effectively.

## Table of Contents

- [Package Search Order](#-package-search-order)
  - [Using `-i` (Index URL)](#option-1-using--i-index-url)
  - [Using `--extra-index-url` (Recommended)](#option-2-using---extra-index-url-recommended)
  - [Multiple Extra Indexes](#option-3-multiple-extra-indexes)
- [Example Requirements File](#-example-requirements-file)
- [Configuration Methods](#-configuration-methods)
- [Package Installation Flow](#-package-installation-flow)
- [Important Considerations](#-important-considerations)
- [Docker Integration](#-docker-integration)
- [Troubleshooting](#-troubleshooting)

## 🔍 Package Search Order

### Option 1: Using `-i` (Index URL)
**Use Case**: Replace default PyPI completely

```bash
pip install -i http://localhost:8080/simple/ some-package
```

**Search Order:**
1. ✅ Your custom index first (`http://localhost:8080/simple/`)
2. ✅ Official PyPI as fallback (`https://pypi.org/simple/`)

**⚠️ Limitation**: If your private server is down, installation may fail even for public packages.

### Option 2: Using `--extra-index-url` (Recommended)
**Use Case**: Add private repository while keeping PyPI as primary source

```bash
pip install --extra-index-url http://localhost:8080/simple/ some-package
```

**Search Order:**
1. ✅ Official PyPI first (`https://pypi.org/simple/`)
2. ✅ Your custom index second (`http://localhost:8080/simple/`)

**✅ Advantage**: Robust fallback behavior - public packages always available.

### Option 3: Multiple Extra Indexes
**Use Case**: Multiple private repositories

```bash
pip install \
  --extra-index-url http://localhost:8080/simple/ \
  --extra-index-url http://another-repo:8080/simple/ \
  some-package
```

**Search Order:**
1. Official PyPI
2. First extra index
3. Second extra index
4. And so on...

## 📝 Example Requirements File

```txt
# requirements.txt
--extra-index-url http://localhost:8080/simple/
--trusted-host localhost

# Standard packages (from PyPI)
requests>=2.28.0
numpy>=1.21.0
pandas>=1.3.0

# Private packages (from your private repo)
my-private-package>=1.0.0
internal-tools>=2.1.0
```

## ⚙️ Configuration Methods

### 1. Requirements File (Recommended)
**Best for**: Project-specific package sources

```txt
# requirements.txt
--extra-index-url http://localhost:8080/simple/
--trusted-host localhost

# Your dependencies here
private-package>=1.0.0
public-package>=2.0.0
```

Install with:
```bash
pip install -r requirements.txt
```

### 2. Command Line Arguments
**Best for**: One-time installations or testing

```bash
pip install -r requirements.txt --extra-index-url http://localhost:8080/simple/ --trusted-host localhost
```

### 3. Global pip Configuration
**Best for**: System-wide or user-wide defaults

#### Linux/macOS: `~/.pip/pip.conf`
```ini
[global]
extra-index-url = http://localhost:8080/simple/
trusted-host = localhost

[install]
trusted-host = localhost
```

#### Windows: `%APPDATA%\pip\pip.ini`
```ini
[global]
extra-index-url = http://localhost:8080/simple/
trusted-host = localhost

[install]
trusted-host = localhost
```

### 4. Environment Variables
**Best for**: CI/CD pipelines and containerized environments

```bash
# Linux/macOS
export PIP_EXTRA_INDEX_URL=http://localhost:8080/simple/
export PIP_TRUSTED_HOST=localhost
pip install -r requirements.txt

# Windows PowerShell
$env:PIP_EXTRA_INDEX_URL="http://localhost:8080/simple/"
$env:PIP_TRUSTED_HOST="localhost"
pip install -r requirements.txt
```

## 🔄 Package Installation Flow

### How pip Searches for Packages

When you run `pip install package-name`, here's what happens:

#### Example 1: Private Package Installation
```bash
pip install --extra-index-url http://localhost:8080/simple/ my-private-package
```

**Flow:**
```
1. pip install my-private-package
   ↓
2. Check official PyPI (https://pypi.org/simple/)
   → my-private-package not found
   ↓  
3. Check extra index (http://localhost:8080/simple/)
   → my-private-package found! ✅
   ↓
4. Install from your private repository
```

#### Example 2: Public Package Installation
```bash
pip install --extra-index-url http://localhost:8080/simple/ requests
```

**Flow:**
```
1. pip install requests
   ↓
2. Check official PyPI (https://pypi.org/simple/)
   → requests found! ✅
   ↓
3. Install from official PyPI
   (Extra indexes not checked when package found in primary index)
```

## 🚨 Important Considerations

### 1. Version Priority Resolution
When the same package exists in multiple repositories, pip chooses based on:

**Priority Order:**
1. **Highest version number** (regardless of source)
2. **Source preference** (only if versions are equal)

**Example:**
```
Official PyPI: my-package==1.0.0
Private Repo:  my-package==1.1.0

Result: pip installs 1.1.0 from private repo ✅
```

### 2. Network Reliability

#### ❌ Risky Approach (using -i):
```bash
# If your private server is down, this will fail even for public packages
pip install -i http://localhost:8080/simple/ requests
```

#### ✅ Safe Approach (using --extra-index-url):
```bash
# Falls back to PyPI if private server is down
pip install --extra-index-url http://localhost:8080/simple/ requests
```

### 3. Security Considerations

- **Always use HTTPS** in production environments
- **Verify package authenticity** from private repositories
- **Use authentication** for private repositories containing sensitive code

## 🐳 Docker Integration

### Dockerfile Example
```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install packages with private repo access
RUN pip install -r requirements.txt \
    --extra-index-url http://your-private-repo:8080/simple/ \
    --trusted-host your-private-repo

COPY . .

CMD ["python", "app.py"]
```

### Docker Compose with Private PyPI
```yaml
# docker-compose.yml
version: '3.8'
services:
  app:
    build: .
    environment:
      - PIP_EXTRA_INDEX_URL=http://pypi-server:8080/simple/
      - PIP_TRUSTED_HOST=pypi-server
    depends_on:
      - pypi-server
  
  pypi-server:
    image: pypiserver/pypiserver:latest
    ports:
      - "8080:8080"
    volumes:
      - ./packages:/data/packages
    command: -p 8080 -a . /data/packages
```

## 🛠️ Troubleshooting

### Common Issues

**Issue**: `Could not find a version that satisfies the requirement`
- **Solution**: Check if package exists in your private repository
- **Debug**: Use `pip install --verbose` to see search process

**Issue**: `Connection error` or `timeout`
- **Solution**: Verify private repository URL and network connectivity
- **Workaround**: Use `--extra-index-url` instead of `-i` for fallback

**Issue**: `SSL certificate verify failed`
- **Solution**: Add `--trusted-host your-domain.com` or configure proper SSL certificates

**Issue**: Wrong package version installed
- **Solution**: Check version numbers across repositories; pip chooses highest version

### Debug Commands

```bash
# Show pip configuration
pip config list

# Verbose installation to see search process
pip install --verbose package-name

# Show available versions from all sources
pip index versions package-name

# Check which repository a package came from
pip show package-name
```