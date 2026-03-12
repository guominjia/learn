# MinIO Mirror and Backup: A Practical Guide to `mc mirror --watch`

## Introduction

When building data pipelines or microservices around MinIO, a common operational need is keeping two buckets in sync — whether for backup, disaster recovery, or environment promotion (staging → production). The MinIO Client (`mc`) provides `mc mirror --watch` for exactly this purpose. This post covers how it works, how sources and targets must be specified, and practical patterns for production use.

---

## How `mc mirror` Works

`mc mirror` copies objects from a source to a target, mirroring the structure and content. The `--watch` flag keeps the process running, continuously replicating any new or modified objects in real time.

```bash
mc mirror --watch source/my-bucket target/my-bucket
```

Under the hood, `mc` watches for events on the source (PUT, DELETE) and replays them on the target, making it behave like a live replication stream.

---

## Can You Use `http://minio:9000` Directly?

A natural first instinct is to write something like:

```bash
# This does NOT work
mc mirror --watch http://minio-source:9000/my-bucket http://minio-target:9000/my-bucket
```

**This will fail.** `mc` does not accept raw HTTP URLs as path arguments in mirror (or any other) commands.

### Why Not?

`mc` separates **connection configuration** from **command arguments**. Connection details — the endpoint URL, access key, secret key, and TLS settings — are stored in named **aliases**. Commands then reference data locations as `alias/bucket/prefix`, keeping the command lines clean and credential-free.

This design has practical benefits:

- Credentials are stored once in `~/.mc/config.json`, not repeated on every command.
- Aliases are reusable across scripts and shell sessions.
- Switching endpoints (e.g., from local MinIO to AWS S3) only requires changing the alias, not every script.

---

## The Correct Approach: Register Aliases First

### Step 1 — Register Aliases

```bash
mc alias set source http://minio-source:9000 ACCESS_KEY_SOURCE SECRET_KEY_SOURCE
mc alias set target http://minio-target:9000 ACCESS_KEY_TARGET SECRET_KEY_TARGET
```

Verify connectivity:

```bash
mc admin info source
mc admin info target
```

### Step 2 — Run the Mirror

```bash
mc mirror --watch source/my-bucket target/my-bucket
```

For Docker Compose environments where MinIO is reachable by its service hostname:

```bash
mc alias set myminio http://minio:9000 minioadmin minioadmin
mc mirror --watch myminio/uploads myminio-backup/uploads
```

---

## Common Scenarios

### Local Directory → MinIO Bucket

You can use a local filesystem path as the source:

```bash
mc mirror --watch /data/exports myminio/exports-backup
```

This is useful for ingesting files from a legacy system into object storage.

### MinIO Bucket → Local Directory

```bash
mc mirror --watch myminio/reports /backup/reports
```

Useful for pulling a live bucket into a local archive or NAS mount.

### MinIO Bucket → MinIO Bucket (Cross-Server)

```bash
mc alias set prod http://minio-prod:9000 KEY SECRET
mc alias set staging http://minio-staging:9000 KEY SECRET

mc mirror --watch prod/datasets staging/datasets
```

This pattern is common for promoting processed data from production to a staging environment for testing.

---

## Useful Flags

| Flag | Description |
|------|-------------|
| `--watch` | Keep running and replicate changes in real time |
| `--overwrite` | Overwrite existing objects at the target |
| `--remove` | Delete objects at the target that no longer exist at the source |
| `--exclude` | Exclude objects matching a pattern (e.g., `--exclude "*.tmp"`) |
| `--older-than` | Only mirror objects older than a duration (e.g., `24h`) |
| `--newer-than` | Only mirror objects newer than a duration |
| `--preserve` | Preserve object metadata and timestamps |
| `--json` | Output progress in JSON format (useful for log parsing) |

Example — mirror only non-temporary files and remove deleted objects at the target:

```bash
mc mirror --watch --remove --exclude "*.tmp" source/my-bucket target/my-bucket
```

---

## Running as a Background Service

For production deployments, wrap `mc mirror --watch` in a container or systemd service so it restarts on failure.

### Docker Compose Example

```yaml
services:
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    volumes:
      - minio-data:/data
    ports:
      - "9000:9000"
      - "9001:9001"

  minio-mirror:
    image: minio/mc
    depends_on:
      - minio
    entrypoint: >
      /bin/sh -c "
        mc alias set source http://minio:9000 minioadmin minioadmin &&
        mc alias set target http://minio-backup:9000 minioadmin minioadmin &&
        mc mirror --watch source/uploads target/uploads
      "

volumes:
  minio-data:
```

### systemd Unit File Example

```ini
[Unit]
Description=MinIO Mirror Watch
After=network.target

[Service]
ExecStart=/usr/local/bin/mc mirror --watch source/uploads target/uploads
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
```

---

## Alias Configuration File

Aliases are stored at `~/.mc/config.json`. You can inspect or manually edit this file:

```json
{
  "aliases": {
    "source": {
      "url": "http://minio-source:9000",
      "accessKey": "ACCESS_KEY",
      "secretKey": "SECRET_KEY",
      "api": "S3v4",
      "path": "auto"
    },
    "target": {
      "url": "http://minio-target:9000",
      "accessKey": "ACCESS_KEY",
      "secretKey": "SECRET_KEY",
      "api": "S3v4",
      "path": "auto"
    }
  }
}
```

In containerized environments, you can mount a pre-configured `config.json` into `/root/.mc/config.json` to avoid running `mc alias set` as part of the entrypoint.

---

## Key Takeaways

- `mc mirror --watch` does **not** accept raw `http://host:port` URLs as source or target arguments.
- Sources and targets must be specified as `alias/bucket` or a local filesystem path.
- Use `mc alias set` to register connection details before running `mc mirror`.
- The alias approach cleanly separates credentials from command logic, making scripts portable and secure.
- For production, run `mc mirror --watch` as a managed process (Docker service, systemd unit) with automatic restart.
