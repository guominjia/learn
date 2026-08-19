---
title: How RAGFlow's Docker Compose Decides Which Services Actually Start
categories: [ai, rag, ragflow]
tags: [ragflow, docker, docker-compose, profiles, deployment]
---

Running `docker compose up` against RAGFlow's `docker/docker-compose.yml` does not start every service defined in the file. Elasticsearch, Infinity, Kibana, Jaeger, and several other services only start when Docker Compose's **profiles** mechanism activates them. This post covers how profiles work in Compose generally, and exactly how RAGFlow wires them up.

## Docker Compose Profiles, Briefly

A service is gated behind a profile with the `profiles` attribute:

```yaml
services:
  frontend:
    image: frontend
    profiles: [frontend]

  backend:
    image: backend
```

Services with a `profiles` list only start when one of their listed profiles is active. Services with **no** `profiles` attribute are always enabled, regardless of which profiles are active.

There are three independent ways to control what actually runs:

| Mechanism | Effect |
|---|---|
| `COMPOSE_PROFILES=name1,name2` (env var, often set in `.env`) | Activates the listed profiles for every `docker compose` invocation |
| `docker compose --profile name up` (repeatable flag) | Activates one or more profiles for that invocation only |
| `docker compose up <service>` / `docker compose run <service>` | Starts that service (and its `depends_on` dependencies) directly, **even if its profile isn't activated** |
| `docker compose --profile "*" up` | Activates every profile |

The third row matters in practice: naming a service explicitly on the command line bypasses the need to activate its profile at all. It's the mechanism for one-off or debugging services that otherwise stay off by default.

## How RAGFlow Assigns Profiles

`docker/docker-compose.yml` pulls in `docker/docker-compose-base.yml` via `include`, so profiles declared in both files jointly decide what starts:

```yaml
include:
  - ./docker-compose-base.yml
```

Services declared with a `profiles` list, as of the current `main` branch:

| Service | File | Profile(s) |
|---|---|---|
| `es01` (Elasticsearch) | docker-compose-base.yml | `elasticsearch` |
| `opensearch01` | docker-compose-base.yml | `opensearch` |
| `infinity` | docker-compose-base.yml | `infinity` |
| `serenedb` | docker-compose-base.yml | `serenedb` |
| `oceanbase` | docker-compose-base.yml | `oceanbase` |
| `seekdb` | docker-compose-base.yml | `seekdb` |
| `sandbox-executor-manager` | docker-compose-base.yml | `sandbox` |
| `jaeger` | docker-compose-base.yml | `jaeger` |
| `nats` | docker-compose-base.yml | `ragflow-go` |
| `tei-cpu` / `tei-gpu` | docker-compose-base.yml | `tei-cpu` / `tei-gpu` |
| `kibana` | docker-compose-base.yml | `kibana` |
| `clickhouse` | docker-compose-base.yml | `clickhouse` |
| `mysql` | docker-compose-base.yml | `mysql`, `metadata-mysql`, `metadata-MySQL`, `metadata-MYSQL` |
| `deepdoc` | docker-compose.yml | `deepdoc` |
| `ragflow-cpu` | docker-compose.yml | `cpu` |
| `ragflow-gpu` | docker-compose.yml | `gpu` |

`minio` and `redis` declare no `profiles` at all, so they always start no matter which profiles are active.

Note that `mysql` is no longer profile-free. It used to start unconditionally like `minio`/`redis`, but the base compose file now gates it behind `mysql`/`metadata-mysql` profiles to support pluggable metadata backends (see the next section).

## Who Sets `COMPOSE_PROFILES`

`docker/.env` computes `COMPOSE_PROFILES` from three variables:

```dotenv
DOC_ENGINE=${DOC_ENGINE:-elasticsearch}
DEVICE=${DEVICE:-cpu}
METADATA_DB_PROFILE=${METADATA_DB_PROFILE:-mysql}
COMPOSE_PROFILES=${DOC_ENGINE},${DEVICE},metadata-${METADATA_DB_PROFILE}
```

- `DOC_ENGINE` selects the document/vector store: `elasticsearch` (default), `infinity`, `opensearch`, `oceanbase`, `seekdb`, `serenedb`, or `gaussdb`.
- `DEVICE` selects `cpu` (default) or `gpu`, which activates `ragflow-cpu` or `ragflow-gpu`.
- `METADATA_DB_PROFILE` selects the bundled metadata database profile (`mysql` by default); for an external GaussDB metadata store, `.env` comments instruct setting `DB_TYPE=gaussdb` alongside a matching `METADATA_DB_PROFILE` so the bundled MySQL container stays disabled.

With defaults, `COMPOSE_PROFILES` resolves to `elasticsearch,cpu,metadata-mysql` — which is exactly why a stock `docker compose up` starts `es01`, `ragflow-cpu`, and `mysql`, but not `infinity`, `opensearch01`, or `ragflow-gpu`.

Optional add-ons follow the same pattern; `.env` documents appending profiles rather than replacing them, for example:

```dotenv
# Enable Kibana:
COMPOSE_PROFILES=${COMPOSE_PROFILES},kibana
# Enable the embedding service:
COMPOSE_PROFILES=${COMPOSE_PROFILES},tei-cpu
# Enable Jaeger tracing:
COMPOSE_PROFILES=${COMPOSE_PROFILES},jaeger
# Enable the sandbox code executor:
COMPOSE_PROFILES=${COMPOSE_PROFILES},sandbox
```

## Switching the Doc Engine in Practice

The README describes changing `DOC_ENGINE` and restarting:

```bash
docker compose -f docker/docker-compose.yml down -v
# then set DOC_ENGINE=infinity in docker/.env
docker compose -f docker/docker-compose.yml up -d
```

`-v` deletes the container volumes, so existing indexed data is cleared as part of the switch — this isn't just a profile toggle, it's a full re-index.

## Three Ways to Control What Starts

Putting it together, there are three independent ways to decide which RAGFlow services run:

1. **Edit `docker/.env`** — change `DOC_ENGINE`, `DEVICE`, `METADATA_DB_PROFILE`, or append to `COMPOSE_PROFILES` directly, then `docker compose up -d`.
2. **Pass `--profile` on the CLI** — e.g. `docker compose --profile kibana --profile jaeger up -d`, without touching `.env`.
3. **Name the service directly** — e.g. `docker compose up kibana` starts `kibana` (and its `depends_on: es01`) even without activating the `kibana` profile.

`docker-compose-macos.yml` is the one exception to all of this: it builds RAGFlow locally from source via `build:` instead of pulling `${RAGFLOW_IMAGE}`, and its `ragflow` service carries no `profiles` entry, so the macOS path doesn't use profile gating for the app service itself.

## References

- [Docker Compose: Using profiles with Compose](https://docs.docker.com/compose/how-tos/profiles/) — official semantics for `profiles`, `--profile`, `COMPOSE_PROFILES`, and explicit service targeting.
- [RAGFlow `docker/docker-compose-base.yml`](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose-base.yml) — profile declarations for `es01`, `opensearch01`, `infinity`, `serenedb`, `oceanbase`, `seekdb`, `sandbox-executor-manager`, `mysql`, `minio`, `redis`, `jaeger`, `nats`, `tei-cpu`/`tei-gpu`, `kibana`, `clickhouse`.
- [RAGFlow `docker/docker-compose.yml`](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose.yml) — `deepdoc`, `ragflow-cpu`, `ragflow-gpu` profile declarations and the `include` of the base file.
- [RAGFlow `docker/.env`](https://github.com/infiniflow/ragflow/blob/main/docker/.env) — `DOC_ENGINE`, `DEVICE`, `METADATA_DB_PROFILE`, and the `COMPOSE_PROFILES` composition.
- [RAGFlow `docker/docker-compose-macos.yml`](https://github.com/infiniflow/ragflow/blob/main/docker/docker-compose-macos.yml) — local build path with no profile gating on the `ragflow` service.
- [RAGFlow README: Switch doc engine from Elasticsearch to Infinity](https://github.com/infiniflow/ragflow/blob/main/README.md) — the `down -v` / edit `.env` / `up -d` workflow for switching `DOC_ENGINE`.
- [DeepWiki: how RAGFlow decides which Compose services run](https://deepwiki.com/search/composeservices_d011e8be-b005-43bb-9639-13edeab2cce0?mode=fast) — original research thread this post is based on.