---
title: How Docker Compose Names RAGFlow's Containers Without `container_name`
categories: [ai, rag, ragflow]
tags: [ragflow, docker, docker-compose, container-name]
---

None of RAGFlow's main services set `container_name` in `docker/docker-compose.yml` or `docker/docker-compose-base.yml`. Yet `docker ps` and the docs consistently show names like `docker-ragflow-cpu-1`. This post covers where that name comes from.

## The Default Formula

When a service has no explicit `container_name`, Compose generates one as:

```text
{project name}-{service name}-{replica index}
```

- **Project name** — from `-p`, the `COMPOSE_PROJECT_NAME` env var, the top-level `name:` in the Compose file, or (lowest priority) the base name of the directory containing the Compose file. RAGFlow's compose files live under `docker/`, so the default project name is `docker`.
- **Service name** — the key under `services:`, e.g. `ragflow-cpu`, `mysql`, `es01`.
- **Replica index** — the instance number for that service; `1` unless the service is scaled.

Source: [Docker Compose: Specify a project name](https://docs.docker.com/compose/how-tos/project-name/).

## No `container_name` in the RAGFlow Services

`ragflow-cpu` in [docker/docker-compose.yml](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose.yml) has no `container_name`:

```yaml
ragflow-cpu:
  depends_on:
    mysql:
      condition: service_healthy
      required: false
  profiles:
    - cpu
  image: ${RAGFLOW_IMAGE}
```

Same for `mysql` in [docker/docker-compose-base.yml](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose-base.yml):

```yaml
mysql:
  profiles:
    - mysql
    - metadata-mysql
  image: mysql:8.0.40
```

`seekdb` is the one exception in the repo — it pins a fixed name:

```yaml
seekdb:
  profiles:
    - seekdb
  image: oceanbase/seekdb:latest
  container_name: seekdb
```

Per the [Compose file reference for `container_name`](https://docs.docker.com/reference/compose-file/services/#container_name), setting a fixed name also means Compose refuses to scale that service beyond one container, and a fixed name can't vary with the project name — running two RAGFlow stacks side by side (e.g. CI jobs) would collide on `seekdb`.

## Where the Name Shows Up

The README's health-check step assumes the default naming:

```bash
docker logs -f docker-ragflow-cpu-1
```

Source: [README.md](https://github.com/infiniflow/ragflow/blob/main/README.md).

The FAQ's `docker ps` example shows the same container name, confirming it in practice:

```text
5bc45806b680   infiniflow/ragflow:latest   "./entrypoint.sh"   ...   docker-ragflow-cpu-1
```

Source: [docs/faq.mdx](https://github.com/infiniflow/ragflow/blob/main/docs/faq.mdx).

## CI Reconstructs the Same Name to Avoid Collisions

RAGFlow's test workflows run many jobs in parallel on shared runners, so they set `COMPOSE_PROJECT_NAME` explicitly and then compute the resulting container name themselves, to `docker exec`/`docker logs` into it later:

```bash
# .github/workflows/tests.yml
COMPOSE_PROJECT_NAME="${GITHUB_RUN_ID}-${DOC_ENGINE}"
echo "COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME}" >> ${GITHUB_ENV}
echo "RAGFLOW_CONTAINER=${COMPOSE_PROJECT_NAME}-ragflow-cpu-1" >> ${GITHUB_ENV}
```

```bash
# .github/workflows/sep-tests.yml
COMPOSE_PROJECT_NAME="${GITHUB_RUN_ID}-${DOC_ENGINE}-${API_PROXY_SCHEME}"
echo "COMPOSE_PROJECT_NAME=${COMPOSE_PROJECT_NAME}" >> ${GITHUB_ENV}
echo "RAGFLOW_CONTAINER=${COMPOSE_PROJECT_NAME}-ragflow-cpu-1" >> ${GITHUB_ENV}
```

This is exactly the `{project}-ragflow-cpu-1` formula, applied manually because CI can't rely on the `docker` directory-name default when multiple jobs share a host.

`docker/docker-compose-CN-oc9.yml` follows the same unset-`container_name` pattern for `ragflow-cpu`/`ragflow-gpu`, so it gets default names too.

## Why the Project Name Also Matters for Volumes

The same project name that prefixes container names also prefixes named volumes. RAGFlow's backup/migration doc calls this out directly:

> The volume name prefix (e.g., `docker_`) comes from the Docker Compose project name. By default it is `docker` (derived from the directory name). If you started RAGFlow with `docker compose -p <project_name>`, your volumes will be prefixed with `<project_name>_` instead, for example `ragflow_mysql_data`.

Source: [docs/administrator/migration/backup_and_migration.md](https://github.com/infiniflow/ragflow/blob/main/docs/administrator/migration/backup_and_migration.md). So `-p`/`COMPOSE_PROJECT_NAME` changes container names and volume names together — a migration or backup script that assumes the `docker-*`/`docker_*` defaults needs the matching `-p` flag whenever a custom project name was used.

## References

- [Docker Compose: Specify a project name](https://docs.docker.com/compose/how-tos/project-name/) — project name precedence: `-p` > `COMPOSE_PROJECT_NAME` > top-level `name:` > Compose file's directory name.
- [Docker Compose file reference: `container_name`](https://docs.docker.com/reference/compose-file/services/#container_name) — semantics of an explicit `container_name`, including the no-scaling restriction.
- [RAGFlow `docker/docker-compose.yml`](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose.yml) — `ragflow-cpu`/`ragflow-gpu` have no `container_name`.
- [RAGFlow `docker/docker-compose-base.yml`](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose-base.yml) — `mysql` has no `container_name`; `seekdb` is the repo's one exception, with `container_name: seekdb`.
- [RAGFlow README](https://github.com/infiniflow/ragflow/blob/main/README.md) — health-check example using `docker logs -f docker-ragflow-cpu-1`.
- [RAGFlow `docs/faq.mdx`](https://github.com/infiniflow/ragflow/blob/main/docs/faq.mdx) — `docker ps` output confirming the `docker-ragflow-cpu-1` name in practice.
- [RAGFlow `.github/workflows/tests.yml`](https://github.com/infiniflow/ragflow/blob/main/.github/workflows/tests.yml) — CI sets `COMPOSE_PROJECT_NAME` and reconstructs `${COMPOSE_PROJECT_NAME}-ragflow-cpu-1`.
- [RAGFlow `.github/workflows/sep-tests.yml`](https://github.com/infiniflow/ragflow/blob/main/.github/workflows/sep-tests.yml) — same pattern, with `API_PROXY_SCHEME` added to the project name.
- [RAGFlow `docs/administrator/migration/backup_and_migration.md`](https://github.com/infiniflow/ragflow/blob/main/docs/administrator/migration/backup_and_migration.md) — the same project name prefixes Compose-managed volume names.
- [DeepWiki: how RAGFlow's default container names are assembled](https://deepwiki.com/search/composecontainer_d5daeec5-cb23-434e-bed6-c056bfe1d263?mode=fast) — original research thread this post is based on.