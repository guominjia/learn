# Docker Compose Multiple Files Configuration

## Overview
Docker Compose supports using multiple configuration files to manage different environments (development, staging, production) through file layering and override patterns.

## 📋 Table of Contents
- [Base Configuration Pattern](#base-configuration-pattern)
- [Override File Structure](#override-file-structure)
- [Command Syntax](#command-syntax)
- [Practical Examples](#practical-examples)
- [Best Practices](#best-practices)
- [Common Gotchas](#common-gotchas)

## Base Configuration Pattern

### File Structure
```
project/
├── docker-compose.yml          # Base configuration
├── docker-compose.dev.yml      # Development overrides
├── docker-compose.prod.yml     # Production overrides
└── docker-compose.test.yml     # Testing overrides
```

### Base Configuration (`docker-compose.yml`)
```yaml
version: '3.8'
services:
    app:
        build: .
        ports:
            - "8000:8000"
        volumes:
            - app-data:/data
        environment:
            - NODE_ENV=production

volumes:
    app-data:
```

## Override File Structure

### Development Override (`docker-compose.dev.yml`)
```yaml
version: '3.8'
services:
    app:
        volumes:
            - ./local-data:/data
            - ./src:/app/src  # Live code reloading
        environment:
            - NODE_ENV=development
            - DEBUG=true
        ports:
            - "3000:8000"  # Different port for dev
```

### Production Override (`docker-compose.prod.yml`)
```yaml
version: '3.8'
services:
    app:
        restart: unless-stopped
        environment:
            - NODE_ENV=production
        deploy:
            resources:
                limits:
                    memory: 512M
```

## Command Syntax

### Multi-file Deployment
```bash
# Development environment
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Production environment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Testing environment
docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
```

### Using Environment Variables
```bash
# Set COMPOSE_FILE environment variable
export COMPOSE_FILE=docker-compose.yml:docker-compose.dev.yml
docker-compose up -d
```

## Practical Examples

### Example 1: Database Configuration Override
**Base configuration:**
```yaml
services:
    db:
        image: postgres:15
        environment:
            POSTGRES_DB: myapp
```

**Development override:**
```yaml
services:
    db:
        ports:
            - "5432:5432"  # Expose port for local access
        environment:
            POSTGRES_PASSWORD: devpassword
```

### Example 2: Volume Override Pattern
**Result of override:** The `./local-data:/data` volume from `docker-compose.dev.yml` completely replaces the `app-data:/data` volume from the base configuration.

## Best Practices

### File Organization
- Keep common configurations in the base `docker-compose.yml`
- Use descriptive suffixes (`.dev.yml`, `.prod.yml`, `.test.yml`)
- Document override purposes in comments

### Override Strategy
```yaml
# ✅ Good: Specific environment tweaks
services:
    app:
        environment:
            - DEBUG=true
        
# ❌ Avoid: Complete service redefinition
services:
    app:
        image: completely-different-image
        # This makes configurations hard to track
```

### Makefile Integration
```makefile
.PHONY: dev prod test

dev:
    docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

prod:
    docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

test:
    docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d
```

## Common Gotchas

### Override Behavior
- **Arrays are replaced**, not merged (ports, volumes, environment)
- **Objects are merged** (labels, deploy configuration)
- **Later files take precedence** when conflicts occur

### Environment Variable Priority
```bash
# Order matters - last file wins
docker-compose -f base.yml -f override1.yml -f override2.yml up
```

### Volume Mounting Issues
```yaml
# Base file
volumes:
    - app-data:/data

# Override file  
volumes:
    - ./local-data:/data
    # ⚠️ This completely replaces the base volume, doesn't add to it
```

## Summary
Multi-file Docker Compose configuration enables environment-specific deployments while maintaining a clean base configuration. Use the `-f` flag to layer configurations, with later files overriding earlier ones for flexible environment management.
