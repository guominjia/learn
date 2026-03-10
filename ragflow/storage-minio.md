# How RAGFlow Stores Files in MinIO

When deploying RAGFlow, one of the most common infrastructure questions is: **where are MinIO credentials stored?**

The short answer is simple:
- MinIO credentials are provided through environment variables.
- They are injected into runtime configuration.
- They are used directly by the MinIO client.
- They are **not** persisted in MySQL.

This post walks through the full configuration path and explains why this design is safer and easier to operate.

## 1) Credentials Start in Environment Variables

RAGFlow defines MinIO credentials in `docker/.env`:

```bash
MINIO_USER=rag_flow
MINIO_PASSWORD=infini_rag_flow
```

In real deployments, these values should come from your secret management process (for example, CI/CD secrets, Docker secrets, or a vault solution), not hard-coded defaults.

## 2) Runtime Config Is Templated via `service_conf.yaml.template`

Those environment variables are referenced by `docker/service_conf.yaml.template`:

```yaml
minio:
  user: '${MINIO_USER:-rag_flow}'
  password: '${MINIO_PASSWORD:-infini_rag_flow}'
  host: '${MINIO_HOST:-minio}:9000'
```

This gives a clean separation:
- **Configuration structure** lives in YAML.
- **Sensitive values** are injected from environment variables.

The same pattern appears for other object storage backends (S3, OSS, Azure), making storage backend switching consistent and predictable.

## 3) The Application Uses These Values to Build the MinIO Client

At runtime, RAGFlow initializes the MinIO connection in `rag/utils/minio_conn.py`:

```python
self.conn = Minio(
    settings.MINIO["host"],
    access_key=settings.MINIO["user"],
    secret_key=settings.MINIO["password"],
    secure=secure,
    http_client=http_client,
)
```

This is the critical boundary: credentials are consumed in memory for client authentication and are not written into relational business tables.

## 4) Production Recommendations

If you are running RAGFlow in production, apply these baseline practices:
- Rotate default credentials immediately.
- Use least-privilege keys for MinIO buckets.
- Enable TLS (`secure=true`) when traffic leaves a trusted internal network.
- Manage secrets outside source control.
- Set `MINIO_HOST` explicitly for external or managed MinIO endpoints.

## Final Takeaway

RAGFlow follows a standard and secure secret flow:

**Environment variables → configuration template → runtime client initialization**

That means MinIO credentials stay in your deployment/runtime secret boundary instead of being persisted in MySQL, which is exactly what you want for a safer architecture.

## References

- <https://deepwiki.com/search/asyncparsedocuments-can-only-h_3edd49f7-85b6-4fa5-bb7a-feffc3456fb1?mode=fast>