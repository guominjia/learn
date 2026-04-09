# Docker Compose Volumes Guide

## Table of Contents
- [Volume Types](#volume-types)
- [Named Volumes](#named-volumes)
- [Bind Mounts](#bind-mounts)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Volume Types

Docker Compose supports two primary volume types for data persistence:

### Named Volumes
Managed by Docker, stored in Docker's internal directory structure:

```yaml
volumes:
    ollama: {}
    open-webui: {}

services:
    ollama:
        volumes:
            - ollama:/root/.ollama
    
    open-webui:
        volumes:
            - open-webui:/app/backend/data
```

### Bind Mounts
Direct mapping between host directories and container paths:

```yaml
services:
    ollama:
        volumes:
            - ./data:/root/.ollama
            - ./config:/etc/ollama
```

## Named Volumes

### Declaration Syntax
```yaml
volumes:
    volume-name: {}  # Empty object = default settings
```

### Key Characteristics
- **Location**: Stored in `/var/lib/docker/volumes/` on Linux
- **Management**: Fully managed by Docker daemon
- **Persistence**: Survives container deletion
- **Sharing**: Can be shared across multiple containers

### Example Configuration
```yaml
version: '3.8'

volumes:
    postgres-data: {}
    redis-cache: {}

services:
    database:
        image: postgres:15
        volumes:
            - postgres-data:/var/lib/postgresql/data
        environment:
            POSTGRES_DB: myapp
            POSTGRES_PASSWORD: secret

    cache:
        image: redis:7-alpine
        volumes:
            - redis-cache:/data
```

## Bind Mounts

### When to Use Bind Mounts
- Development environments requiring live code changes
- Configuration files that need host-level editing
- Log files requiring external monitoring tools
- Backup strategies involving host filesystem

### Security Considerations
```yaml
services:
    app:
        volumes:
            # ✅ Good: Read-only configuration
            - ./config:/app/config:ro
            
            # ⚠️ Caution: Full write access
            - ./data:/app/data
            
            # ❌ Avoid: Root directory exposure
            - /:/host:ro
```

## Best Practices

### Production Deployments
```yaml
# Use named volumes for databases
volumes:
    postgres-data:
        driver: local
        driver_opts:
            type: none
            o: bind
            device: /opt/app/data

services:
    database:
        volumes:
            - postgres-data:/var/lib/postgresql/data
        restart: unless-stopped
```

### Development Workflow
```yaml
services:
    app:
        volumes:
            # Source code for live reload
            - ./src:/app/src
            # Dependencies cache (named volume)
            - node_modules:/app/node_modules
            # Build artifacts
            - ./dist:/app/dist

volumes:
    node_modules: {}
```

### Multi-Stage Configuration
```yaml
# docker-compose.yml (base)
services:
    app:
        volumes:
            - app-data:/data

# docker-compose.dev.yml (development override)
services:
    app:
        volumes:
            - ./local-data:/data  # Override with bind mount

volumes:
    app-data: {}
```

## Troubleshooting

### Common Issues

**Volume Not Persisting Data**
```bash
# Check volume exists
docker volume ls

# Inspect volume details
docker volume inspect project_volume-name

# Verify mount points
docker compose exec service-name df -h
```

**Permission Problems**
```yaml
services:
    app:
        user: "${UID}:${GID}"  # Use host user ID
        volumes:
            - ./data:/app/data
```

**Volume Cleanup**
```bash
# Remove unused volumes
docker volume prune

# Remove specific volume (data will be lost)
docker volume rm project_volume-name
```

### Performance Optimization

**For macOS/Windows Development**
```yaml
# Use cached or delegated consistency for better performance
services:
    app:
        volumes:
            - ./src:/app/src:cached
            - ./node_modules:/app/node_modules:delegated
```

### Volume Backup Strategy
```bash
# Backup named volume
docker run --rm \
    -v project_postgres-data:/source:ro \
    -v $(pwd):/backup \
    alpine tar czf /backup/postgres-backup.tar.gz -C /source .

# Restore named volume
docker run --rm \
    -v project_postgres-data:/target \
    -v $(pwd):/backup \
    alpine tar xzf /backup/postgres-backup.tar.gz -C /target
```

## Integration with Modern Tools

### UV Package Manager
```yaml
services:
    python-app:
        volumes:
            - ./src:/app/src
            - uv-cache:/root/.cache/uv  # Cache UV downloads
            - ./pyproject.toml:/app/pyproject.toml:ro

volumes:
    uv-cache: {}
```

### Multi-Service Data Sharing
```yaml
services:
    api:
        volumes:
            - shared-data:/app/data
    
    worker:
        volumes:
            - shared-data:/worker/input
    
    nginx:
        volumes:
            - shared-data:/var/www/static:ro

volumes:
    shared-data: {}
```