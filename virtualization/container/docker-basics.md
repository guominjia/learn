# Docker Reference Guide

## Table of Contents
1. [Installation and Setup](#installation-and-setup)
2. [User Management](#user-management)
3. [Basic Docker Commands](#basic-docker-commands)
4. [Docker Images](#docker-images)
5. [Docker Containers](#docker-containers)
6. [Docker Volumes](#docker-volumes)
7. [Docker Networks](#docker-networks)
8. [Dockerfile Best Practices](#dockerfile-best-practices)
9. [Docker Compose](#docker-compose)
10. [Troubleshooting](#troubleshooting)

## Installation and Setup

### Installing Docker on Linux

```bash
# Install below package on Ubuntu 20.04.6 LTS
sudo apt install docker docker-compose-v2
```

```bash
# Then configure below file

# /etc/systemd/system/docker.service.d/http-proxy.conf
[Service]
Environment="HTTP_PROXY=http://your-proxy:your-port"
Environment="HTTPS_PROXY=http://your-proxy:your-port"
```

```bash
# Use below to install docker if fail to install docker

# Update package index
sudo apt-get update

# Install required packages
sudo apt-get install ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

### Start and Enable Docker Service

```bash
# Start and enable Docker service
sudo systemctl start docker
sudo systemctl enable docker

# Check Docker status
sudo systemctl status docker
```

## User Management

### Adding User to Docker Group

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Apply the group change immediately
newgrp docker

# Or logout and login again for permanent effect
```

### Understanding `newgrp docker` and Login Requirements

**What's `newgrp docker`?**
- `newgrp docker` starts a new shell session with the docker group as the primary group
- It temporarily applies the group membership without requiring a full logout/login
- Only affects the current terminal session

**Why logout and login?**
- Group memberships are assigned when you log in
- Adding a user to a group doesn't immediately affect existing sessions
- Logout/login ensures the group change is permanent across all sessions
- Alternative to `newgrp` for system-wide effect

**Check your groups:**
```bash
# Check current groups
groups

# Check groups for specific user
groups $USER

# Check if docker group exists
getent group docker
```

### Test Docker Installation

```bash
# Test Docker without sudo
docker run hello-world

# Check Docker version
docker --version
docker version
docker info
```

## Basic Docker Commands

### Container Lifecycle

```bash
# List running containers
docker ps

# List all containers (including stopped)
docker ps -a

# Run a container
docker run [OPTIONS] IMAGE [COMMAND] [ARG...]

# Run container interactively
docker run -it ubuntu bash

# Run container in background (detached)
docker run -d nginx

# Stop a container
docker stop CONTAINER_ID

# Start a stopped container
docker start CONTAINER_ID

# Restart a container
docker restart CONTAINER_ID

# Remove a container
docker rm CONTAINER_ID

# Remove all stopped containers
docker container prune
```

### Container Interaction

```bash
# Execute command in running container
docker exec -it CONTAINER_ID bash

# Copy files to/from container
docker cp file.txt CONTAINER_ID:/path/to/destination
docker cp CONTAINER_ID:/path/to/file.txt ./local-file.txt

# View container logs
docker logs CONTAINER_ID
docker logs -f CONTAINER_ID  # Follow logs

# View container resource usage
docker stats
docker stats CONTAINER_ID
```

## Docker Images

### Image Management

```bash
# List images
docker images
docker image ls

# Pull an image
docker pull IMAGE_NAME:TAG

# Build an image from Dockerfile
docker build -t IMAGE_NAME:TAG .
docker build -t IMAGE_NAME:TAG -f Dockerfile.custom .

# Tag an image
docker tag SOURCE_IMAGE:TAG TARGET_IMAGE:TAG

# Push image to registry
docker push IMAGE_NAME:TAG

# Remove an image
docker rmi IMAGE_ID

# Remove unused images
docker image prune
docker image prune -a  # Remove all unused images
```

### Image Information

```bash
# Inspect an image
docker inspect IMAGE_NAME

# View image history
docker history IMAGE_NAME

# Search for images
docker search IMAGE_NAME
```

## Docker Containers

### Running Containers with Options

```bash
# Run with port mapping
docker run -p 8080:80 nginx

# Run with volume mounting
docker run -v /host/path:/container/path IMAGE

# Run with environment variables
docker run -e VAR_NAME=value IMAGE

# Run with custom name
docker run --name my-container IMAGE

# Run with memory and CPU limits
docker run --memory=512m --cpus=1.5 IMAGE

# Run with restart policy
docker run --restart=always IMAGE
```

### Container Networking

```bash
# List networks
docker network ls

# Create a network
docker network create my-network

# Run container on specific network
docker run --network=my-network IMAGE

# Connect container to network
docker network connect my-network CONTAINER_ID

# Disconnect container from network
docker network disconnect my-network CONTAINER_ID
```

## Docker Volumes

### Volume Management

```bash
# List volumes
docker volume ls

# Create a volume
docker volume create my-volume

# Use volume in container
docker run -v my-volume:/data IMAGE

# Mount host directory (bind mount)
docker run -v /host/path:/container/path IMAGE

# Mount with read-only
docker run -v /host/path:/container/path:ro IMAGE

# Remove volume
docker volume rm my-volume

# Remove unused volumes
docker volume prune
```

## Docker Networks

### Network Types and Commands

```bash
# Default networks
docker network ls

# Create bridge network
docker network create --driver bridge my-bridge

# Create overlay network (for swarm)
docker network create --driver overlay my-overlay

# Inspect network
docker network inspect my-network

# Remove network
docker network rm my-network
```

## Dockerfile Best Practices

### Sample Dockerfile

```dockerfile
# Use official base image
FROM node:18-alpine

# Set working directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy application code
COPY . .

# Create non-root user
RUN addgroup -g 1001 -S nodejs
RUN adduser -S nextjs -u 1001

# Change ownership
RUN chown -R nextjs:nodejs /app
USER nextjs

# Expose port
EXPOSE 3000

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:3000/health || exit 1

# Start application
CMD ["npm", "start"]
```

### Dockerfile Best Practices

- Use official base images
- Use specific tags, avoid `latest`
- Use multi-stage builds for smaller images
- Minimize layers by combining RUN commands
- Use `.dockerignore` to exclude unnecessary files
- Run as non-root user
- Use COPY instead of ADD when possible
- Set proper labels for metadata

### Multi-stage Build Example

```dockerfile
# Build stage
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

# Production stage
FROM node:18-alpine AS production
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY --from=builder /app/dist ./dist
USER node
CMD ["npm", "start"]
```

## Docker Compose

### Basic docker-compose.yml

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NODE_ENV=production
    volumes:
      - ./app:/app
    depends_on:
      - db
    restart: unless-stopped

  db:
    image: postgres:14
    environment:
      POSTGRES_DB: myapp
      POSTGRES_USER: user
      POSTGRES_PASSWORD: password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
```

### Docker Compose Commands

```bash
# Start services
docker-compose up
docker-compose up -d  # In background

# Stop services
docker-compose down
docker-compose down -v  # Remove volumes

# Build services
docker-compose build
docker-compose build --no-cache

# View logs
docker-compose logs
docker-compose logs web

# Scale services
docker-compose up --scale web=3

# Execute commands
docker-compose exec web bash
```

## Automation with Python

### Using Docker with Python

```python
import os
import subprocess
import docker

# Using docker-py library
client = docker.from_env()

# Run a container
container = client.containers.run(
    "ubuntu:latest",
    "echo hello world",
    remove=True,
    detach=False
)

# List containers
for container in client.containers.list():
    print(container.name)

# Build an image
image = client.images.build(path=".", tag="my-app:latest")
```

### Running Docker with Sudo (Not Recommended for Production)

```python
import os
import subprocess

# Set sudo password in environment (not recommended for production)
os.environ['SUDO_ASKPASS'] = '/path/to/password/script'

# Example with subprocess
result = subprocess.run([
    "sudo", "-A", "docker", "run", "-i", "--rm",
    "ubuntu:latest", "echo", "hello"
], capture_output=True, text=True)

print(result.stdout)
```

### Password Script for Sudo (Development Only)

```bash
#!/bin/bash
# /path/to/password/script
echo "your_password"
```

```bash
# Make script executable
chmod +x /path/to/password/script
```

## Troubleshooting

### Common Issues and Solutions

#### Permission Denied
```bash
# Error: permission denied while trying to connect to Docker daemon
# Solution: Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### Docker Daemon Not Running
```bash
# Start Docker daemon
sudo systemctl start docker

# Enable auto-start
sudo systemctl enable docker
```

#### Port Already in Use
```bash
# Find process using port
sudo lsof -i :8080
sudo netstat -tulpn | grep :8080

# Kill process
sudo kill -9 PID
```

#### Out of Disk Space
```bash
# Clean up Docker system
docker system prune
docker system prune -a  # Remove all unused objects

# Remove specific items
docker container prune
docker image prune
docker volume prune
docker network prune
```

### Useful Debugging Commands

```bash
# Check Docker installation
docker version
docker info

# Check system resources
docker system df
docker system events

# Inspect containers/images
docker inspect CONTAINER_OR_IMAGE
docker logs CONTAINER_ID

# Debug networking
docker network ls
docker port CONTAINER_ID
```

### Performance Monitoring

```bash
# Monitor container stats
docker stats

# Export container stats
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check container processes
docker top CONTAINER_ID
```

## Security Best Practices

1. **Never run containers as root in production**
2. **Use official images from trusted sources**
3. **Scan images for vulnerabilities**
4. **Limit container resources**
5. **Use secrets management for sensitive data**
6. **Keep Docker and images updated**
7. **Use read-only filesystems when possible**
8. **Implement proper network segmentation**

### Security Commands

```bash
# Scan image for vulnerabilities (if you have docker scan)
docker scan IMAGE_NAME

# Run container with security options
docker run --security-opt=no-new-privileges:true IMAGE

# Run with read-only root filesystem
docker run --read-only IMAGE

# Drop capabilities
docker run --cap-drop=ALL --cap-add=NET_BIND_SERVICE IMAGE
```

## Useful Resources

- [Docker Official Documentation](https://docs.docker.com/)
- [Docker Hub](https://hub.docker.com/)
- [Docker Best Practices](https://docs.docker.com/develop/dev-best-practices/)
- [Dockerfile Reference](https://docs.docker.com/engine/reference/builder/)
- [Docker Compose Reference](https://docs.docker.com/compose/compose-file/)
