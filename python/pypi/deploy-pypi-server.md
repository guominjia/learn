# PyPI Server Deployment Guide

This guide covers multiple approaches to deploy a private PyPI server for hosting your Python packages.

## 🚀 Quick Start (Recommended)

For immediate use with existing packages:

```bash
# 1. Install pypiserver
pip install pypiserver

# 2. Create package directory  
mkdir ~/my-packages

# 3. Build your packages and copy to ~/my-packages
python -m build  # Creates dist/ folder
cp dist/*.whl dist/*.tar.gz ~/my-packages/

# 4. Start server
pypi-server run -p 8080 ~/my-packages

# 5. Install packages
pip install -i http://localhost:8080/simple/ your-package-name
```

## Deployment Options

### Option 1: Simple HTTP File Server (Development Only)
**Use Case**: Quick testing and development environments

#### Step 1: Create Repository Structure

```bash
mkdir my-pypi-repo
cd my-pypi-repo
mkdir -p simple/your-package-name
```

#### Step 2: Build Your Package

```bash
# In your package directory
python setup.py sdist bdist_wheel

# Or if using pyproject.toml (modern approach):
pip install build
python -m build
```

#### Step 3: Copy Packages to Repository

```bash
cp dist/*.whl my-pypi-repo/simple/your-package-name/
cp dist/*.tar.gz my-pypi-repo/simple/your-package-name/
```

#### Step 4: Generate Index Files

Create a script to generate the required HTML index files:
Create a script to generate the required HTML index files:

```python
#!/usr/bin/env python3
# create_index.py
import os
from pathlib import Path

def create_index_html(package_dir):
    """Create index.html for a package directory"""
    package_name = package_dir.name
    files = list(package_dir.glob("*"))
    
    html = f"""<!DOCTYPE html>
<html>
<head><title>Links for {package_name}</title></head>
<body>
<h1>Links for {package_name}</h1>
"""
    
    for file in files:
        if file.is_file():
            html += f'<a href="{file.name}">{file.name}</a><br>\n'
    
    html += "</body></html>"
    
    with open(package_dir / "index.html", "w") as f:
        f.write(html)

def create_main_index(repo_dir):
    """Create main index.html"""
    packages = [d for d in (repo_dir / "simple").iterdir() if d.is_dir()]
    
    html = """<!DOCTYPE html>
<html>
<head><title>Simple Python Package Index</title></head>
<body>
<h1>Simple Python Package Index</h1>
"""
    
    for pkg in packages:
        html += f'<a href="simple/{pkg.name}/">{pkg.name}</a><br>\n'
    
    html += "</body></html>"
    
    with open(repo_dir / "index.html", "w") as f:
        f.write(html)

# Usage
repo_path = Path("my-pypi-repo")
for package_dir in (repo_path / "simple").iterdir():
    if package_dir.is_dir():
        create_index_html(package_dir)

create_main_index(repo_path)
```

#### Step 5: Serve Repository

```bash
cd my-pypi-repo
python -m http.server 8080
```

#### Step 6: Install from Your Repository

```bash
pip install -i http://localhost:8080/simple/ your-package-name
```

### Option 2: PyPI Server (Recommended for Production)
**Use Case**: Production environments, robust package management with authentication support

#### Step 1: Install PyPI Server

```bash
pip install pypiserver[passlib]
```

#### Step 2: Create Package Directory

```bash
mkdir ~/pypi-packages
```

#### Step 3: Copy Your Packages

```bash
cp dist/*.whl ~/pypi-packages/
cp dist/*.tar.gz ~/pypi-packages/
```

#### Step 4: Start Server

```bash
# Simple server (no authentication)
pypi-server run -p 8080 ~/pypi-packages

# With authentication (see Authentication section below)
pypi-server run -p 8080 -P .htaccess ~/pypi-packages
```

#### Step 5: Install from Server

```bash
pip install -i http://localhost:8080/simple/ your-package-name
```

### Option 3: Docker-based PyPI Server
**Use Case**: Containerized deployments, easy scaling and management

#### Create docker-compose.yml

```yaml
version: '3.8'
services:
  pypi:
    image: pypiserver/pypiserver:latest
    ports:
      - "8080:8080"
    volumes:
      - ./packages:/data/packages
    command: -p 8080 -a . /data/packages
    restart: unless-stopped
```

#### Start Server

```bash
mkdir packages
# Copy your .whl and .tar.gz files to packages/
docker-compose up -d
```

## � Authentication Setup

For production environments, enable authentication to secure your PyPI server.

### Using htpasswd Authentication

```bash
# Install htpasswd (on Ubuntu/Debian)
sudo apt-get install apache2-utils

# Create password file
htpasswd -c .htaccess admin

# Start server with authentication
pypi-server run -p 8080 -P .htaccess ~/pypi-packages
```

### Using Python to Create Password Hash

```python
#!/usr/bin/env python3
# create_auth.py
import bcrypt

password = b"your_password"
hashed = bcrypt.hashpw(password, bcrypt.gensalt())

with open(".htaccess", "w") as f:
    f.write(f"admin:{hashed.decode()}\n")

print("Authentication file created!")
print("Start server with: pypi-server run -p 8080 -P .htaccess ~/pypi-packages")
```

### Install with Authentication

```bash
pip install -i http://admin:your_password@localhost:8080/simple/ your-package-name
```

## 📦 Building and Managing Packages

### For GitHub-hosted Packages

If you have packages hosted on GitHub, you can build and serve them through your PyPI server:

```python
#!/usr/bin/env python3
# build_github_packages.py
import subprocess
import shutil
import os
from pathlib import Path
import tempfile
import sys

def build_package_from_git(git_url, repo_dir):
    """Clone, build, and copy a package from a Git repository"""
    with tempfile.TemporaryDirectory() as temp_dir:
        print(f"Processing {git_url}")
        
        # Clone repository
        subprocess.run(["git", "clone", git_url, temp_dir], check=True)
        original_dir = os.getcwd()
        
        try:
            os.chdir(temp_dir)
            
            # Build package
            try:
                # Try modern build first
                subprocess.run([sys.executable, "-m", "build"], check=True)
            except subprocess.CalledProcessError:
                # Fallback to setup.py
                subprocess.run([sys.executable, "setup.py", "sdist", "bdist_wheel"], check=True)
            
            # Get package name from built files
            dist_dir = Path("dist")
            if not dist_dir.exists():
                print(f"No dist directory found for {git_url}")
                return
            
            # Extract package name from wheel or source distribution
            built_files = list(dist_dir.glob("*"))
            if not built_files:
                print(f"No built files found for {git_url}")
                return
            
            # Assume package name from first file
            first_file = built_files[0].name
            pkg_name = first_file.split('-')[0]
            
            # Create package directory in repo
            pkg_dir = repo_dir / pkg_name
            pkg_dir.mkdir(exist_ok=True)
            
            # Copy all built packages
            for dist_file in built_files:
                shutil.copy2(dist_file, pkg_dir)
                print(f"Copied {dist_file.name} to {pkg_name}/")
            
        finally:
            os.chdir(original_dir)

# Example usage
if __name__ == "__main__":
    # Your GitHub packages
    github_packages = [
        "https://github.com/yourusername/package1.git",
        "https://github.com/yourusername/package2.git",
    ]
    
    repo_dir = Path("my-pypi-repo/simple")
    repo_dir.mkdir(parents=True, exist_ok=True)
    
    for package_url in github_packages:
        try:
            build_package_from_git(package_url, repo_dir)
        except Exception as e:
            print(f"Failed to process {package_url}: {e}")
```

### Building Local Packages

```bash
# For packages with setup.py
python setup.py sdist bdist_wheel

# For packages with pyproject.toml (modern approach)
pip install build
python -m build

# Copy built packages to your PyPI server
cp dist/* ~/pypi-packages/
```

## 🛠️ Advanced Configuration

### Custom PyPI Server Script

```python
#!/usr/bin/env python3
# custom_pypi_server.py
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os
import argparse

class PyPIHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, packages_dir="packages", **kwargs):
        self.packages_dir = packages_dir
        super().__init__(*args, directory=packages_dir, **kwargs)
    
    def log_message(self, format, *args):
        print(f"[PyPI Server] {format % args}")

def main():
    parser = argparse.ArgumentParser(description="Simple PyPI Server")
    parser.add_argument("-p", "--port", type=int, default=8080, help="Port to serve on")
    parser.add_argument("-d", "--directory", default="packages", help="Directory containing packages")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")
    
    handler = lambda *args, **kwargs: PyPIHandler(*args, packages_dir=args.directory, **kwargs)
    server = HTTPServer((args.host, args.port), handler)
    
    print(f"PyPI server running at http://{args.host}:{args.port}")
    print(f"Serving packages from: {args.directory}")
    print(f"Install packages with: pip install -i http://{args.host}:{args.port}/simple/ package-name")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down server...")
        server.shutdown()

if __name__ == "__main__":
    main()
```

### Usage

```bash
python custom_pypi_server.py -p 8080 -d ~/my-packages
```

## � Best Practices

1. **Version Management**: Use semantic versioning for your packages
2. **Security**: Always use authentication in production environments
3. **Backup**: Regularly backup your packages directory
4. **Monitoring**: Monitor server logs for usage and errors
5. **SSL/TLS**: Use HTTPS in production (consider reverse proxy like nginx)
6. **Storage**: For large deployments, consider using object storage backends

## 🔗 Integration with pip.conf

Create a pip configuration file to automatically use your PyPI server:

### Linux/macOS: `~/.pip/pip.conf`
### Windows: `%APPDATA%\pip\pip.ini`

```ini
[global]
index-url = http://localhost:8080/simple/
trusted-host = localhost
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Package not found
- **Solution**: Check if package files are in the correct directory structure

**Issue**: Permission denied
- **Solution**: Ensure proper file permissions on package directory

**Issue**: Authentication failures
- **Solution**: Verify `.htaccess` file format and credentials

**Issue**: SSL certificate errors
- **Solution**: Add `trusted-host` to pip configuration or use proper SSL certificates