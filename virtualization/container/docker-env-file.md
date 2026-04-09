# Docker Environment File Q&A Guide

## 📋 Table of Contents
- [Common Issues](#common-issues)
- [Quote Handling](#quote-handling)
- [Network Configuration](#network-configuration)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## ❓ Common Issues

### Issue 1: `httpx.UnsupportedProtocol: Request URL is missing an 'http://' or 'https://' protocol`

**🔍 Root Cause:**
- The `.env` file had quotes around URL values (`"http://gateway.example.com/v1"`)
- Docker reads these quotes as part of the actual value
- This results in the URL being `"http://gateway.example.com/v1"` (with quotes)
- HTTP clients cannot parse URLs that start with quotes

**✅ Solution:**
Remove quotes from environment variables in `.env` files when using Docker's `--env-file` option.

**❌ Incorrect:**
```properties
AI_AUTH_KEY="Bearer sk-xxxxx"
AI_BASE_URL="http://gateway.example.com/v1"
HF_ENDPOINT="https://hf-mirror.com"
```

**✅ Correct:**
```properties
AI_AUTH_KEY=Bearer sk-xxxxx
AI_BASE_URL=http://gateway.example.com/v1
HF_ENDPOINT=https://hf-mirror.com
```

## 🔤 Quote Handling

### When Docker Reads Environment Files

**Different Tools Handle Quotes Differently:**

| Tool | Quote Behavior | Example |
|------|----------------|---------|
| Shell (`source .env`) | Strips quotes | `API_KEY="value"` → `value` |
| Docker (`--env-file`) | Preserves quotes | `API_KEY="value"` → `"value"` |
| Node.js (`dotenv`) | Strips quotes | `API_KEY="value"` → `value` |
| Python (`python-dotenv`) | Configurable | Depends on settings |

### Best Practices for Docker

1. **Don't use quotes in `.env` files for Docker:**
```properties
# ✅ Good for Docker
DATABASE_URL=postgresql://user:pass@localhost/db
API_KEY=sk-1234567890

# ❌ Avoid for Docker  
DATABASE_URL="postgresql://user:pass@localhost/db"
API_KEY="sk-1234567890"
```

2. **Use quotes only when values contain spaces:**
```properties
# ✅ When spaces are needed
APP_NAME=My Application Name
# or escape spaces
APP_NAME=My\ Application\ Name
```

## 🎯 Best Practices

### Environment File Structure

```properties
# ===== API Configuration =====
API_BASE_URL=https://api.example.com/v1
API_KEY=your-api-key-here
API_TIMEOUT=30

# ===== Database Configuration =====
DB_HOST=localhost
DB_PORT=5432
DB_NAME=myapp
DB_USER=postgres

# ===== Feature Flags =====
ENABLE_DEBUG=false
ENABLE_LOGGING=true
```

### Security Considerations

1. **Never commit `.env` files with secrets**
2. **Use `.env.example` for templates:**
```properties
# .env.example
API_KEY=your-api-key-here
DATABASE_URL=postgresql://user:pass@host:port/db
```

3. **Validate environment variables in code:**
```python
import os

def validate_env():
    required_vars = ['API_KEY', 'BASE_URL']
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        raise ValueError(f"Missing environment variables: {missing}")
```

## 🔧 Troubleshooting

### Debug Environment Variables

**Check what Docker actually sees:**
```bash
docker run --env-file .env your-image env | grep YOUR_VAR
```

**Debug in Python:**
```python
import os
print(f"Raw value: '{os.getenv('YOUR_VAR')}'")
print(f"Length: {len(os.getenv('YOUR_VAR', ''))}")
print(f"Starts with quote: {os.getenv('YOUR_VAR', '').startswith('\"')}")
```

### Common Error Patterns

| Error | Likely Cause | Solution |
|-------|--------------|----------|
| `UnsupportedProtocol` | Quotes in URL | Remove quotes from `.env` |
| `ConnectionError` | Network isolation | Use `--network host` |
| `Invalid URL` | Missing protocol | Add `http://` or `https://` |
| `Empty variable` | Typo in variable name | Check spelling |

### Testing Environment Setup

```bash
# Test environment file loading
docker run --rm --env-file .env alpine:latest sh -c 'echo "API_KEY=$API_KEY"'

# Test network connectivity
docker run --rm --network host alpine:latest ping -c 1 google.com

# Test DNS resolution
docker run --rm alpine:latest nslookup your-internal-domain.com
```

## 📝 Code Examples

### Robust Environment Loading in Python

```python
import os
from typing import Optional

class Config:
    def __init__(self):
        self.api_key = self._get_required_env('API_KEY')
        self.base_url = self._get_required_env('BASE_URL')
        self.timeout = int(self._get_env('TIMEOUT', '30'))
    
    def _get_required_env(self, key: str) -> str:
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} is not set")
        # Strip quotes if present (defensive programming)
        return value.strip('"\'')
    
    def _get_env(self, key: str, default: str) -> str:
        return os.getenv(key, default).strip('"\'')

# Usage
config = Config()
```

### Docker Compose with Environment Files

```yaml
version: '3.8'
services:
  app:
    build: .
    env_file:
      - .env
    environment:
      - NODE_ENV=production
    networks:
      - app-network
    # Use host network for internal services
    # network_mode: host

networks:
  app-network:
    driver: bridge
```

## 🎉 Success Checklist

After applying fixes, verify:

- [ ] ✅ Environment variables load without quotes
- [ ] ✅ Network connectivity works (can reach APIs)
- [ ] ✅ No protocol errors in HTTP requests
- [ ] ✅ Application starts successfully
- [ ] ✅ All required environment variables are present
- [ ] ✅ No sensitive data in version control

---

**💡 Pro Tip:** Always test your Docker environment setup with a simple `echo` command before running complex applications to verify environment variables are loaded correctly.