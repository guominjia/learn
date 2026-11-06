# Docker Restart Policies in Docker Compose

## The Question

If you set `restart: unless-stopped` in your `docker-compose.yml`, will the containers automatically restart after a system reboot?

**Yes**, they will — as long as the Docker daemon itself starts on boot.

## Restart Policy Overview

Docker provides four restart policies:

| Policy | On Crash | On Reboot | After Manual `docker stop` + Reboot |
|---|---|---|---|
| `no` | ❌ | ❌ | ❌ |
| `on-failure` | ✅ | ❌ | ❌ |
| `unless-stopped` | ✅ | ✅ | ❌ |
| `always` | ✅ | ✅ | ✅ |

## `unless-stopped` vs `always`

The only difference between `unless-stopped` and `always` is:

- **`always`**: The container restarts no matter what — even if you manually stopped it before the reboot.
- **`unless-stopped`**: The container restarts on reboot **unless** it was explicitly stopped with `docker stop` before the system went down.

### Example

```yaml
services:
  web:
    image: nginx:latest
    restart: unless-stopped
```

With this configuration:

- If the container crashes → it will be restarted automatically.
- If the system reboots → it will be restarted automatically.
- If you run `docker stop web` and then reboot → it will **not** be restarted.

## Prerequisites

The restart policy only works if the Docker daemon is running after the system boots. Make sure Docker is enabled to start on boot:

**Linux:**

```bash
sudo systemctl enable docker
```

**Windows / macOS (Docker Desktop):**

Enable "Start Docker Desktop when you sign in" in Docker Desktop settings.

## Recommendation

For most production services, `unless-stopped` is the best default. It gives you automatic recovery from crashes and reboots while still respecting your manual `docker stop` commands. Use `always` only when you want a container to run unconditionally regardless of prior state.
