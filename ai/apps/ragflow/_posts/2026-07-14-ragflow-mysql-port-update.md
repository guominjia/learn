---
title: RAGFlow's Default MySQL Host Port Changed from 5455 to 3306
categories: [ai, rag, ragflow]
tags: [ragflow, mysql, docker, docker-compose, upgrade]
---

When upgrading RAGFlow from v0.24.0 to v0.26.4, one small Docker configuration change can affect database clients and even prevent the stack from starting: the default host port for the bundled MySQL service changed from `5455` to `3306`.

The precise variable is `EXPOSE_MYSQL_PORT`, not `MYSQL_PORT`. This distinction is important because the MySQL port inside the Docker network has remained `3306` throughout the change. Only the port published on the Docker host changed.

## The Configuration Change

RAGFlow v0.24.0 defines the following values in `docker/.env`:

```dotenv
MYSQL_PORT=3306
EXPOSE_MYSQL_PORT=5455
```

In v0.26.4, the defaults are:

```dotenv
MYSQL_PORT=3306
EXPOSE_MYSQL_PORT=3306
```

The change was already present in v0.25.0 and remains in the v0.26.x releases. The version comparison is therefore:

| Version | Internal connection port | Host-exposed port |
|---|---:|---:|
| v0.24.0 | `3306` | `5455` |
| v0.25.0 | `3306` | `3306` |
| v0.26.4 | `3306` | `3306` |

This is not a MySQL server port change. It is a Docker port-publishing change.

## Internal Port vs. Host Port

The bundled MySQL container listens on its standard port, `3306`. Docker Compose publishes that container port with a mapping equivalent to:

```yaml
ports:
	- ${EXPOSE_MYSQL_PORT}:3306
```

Docker port mappings use the form:

```text
HOST_PORT:CONTAINER_PORT
```

The effective network paths are therefore different depending on where the client runs.

### RAGFlow Running Inside Docker

The RAGFlow container reaches MySQL through the Compose network:

```text
ragflow container -> mysql:3306
```

This path uses `MYSQL_HOST=mysql` and `MYSQL_PORT=3306`. It does not use the host-published port. As a result, changing `EXPOSE_MYSQL_PORT` does not normally affect communication between the RAGFlow and MySQL containers.

### A Client Running on the Host

A host-side MySQL client, database IDE, backup script, or monitoring agent uses the published port:

```text
host client -> 127.0.0.1:EXPOSE_MYSQL_PORT -> mysql container:3306
```

With the unmodified configuration, that endpoint changed as follows:

```text
v0.24.0: 127.0.0.1:5455
v0.26.4: 127.0.0.1:3306
```

This is where the upgrade becomes visible.

## What Can Break During an Upgrade?

### Existing Client Configuration

Tools configured for `127.0.0.1:5455` will fail to connect after the new Compose configuration recreates MySQL with host port `3306`. Update those clients to use `3306`, or preserve the previous host port in the environment file.

A typical symptom is a connection-refused error on port `5455`, even though RAGFlow itself continues to connect to MySQL successfully inside Docker.

### A Port Conflict on the Host

Port `3306` is commonly occupied by a locally installed MySQL or MariaDB server. If another process already listens on that port, Docker cannot publish the RAGFlow MySQL container there. Startup may fail with an error similar to:

```text
Bind for 0.0.0.0:3306 failed: port is already allocated
```

The older `5455` default reduced the chance of this collision. Before upgrading, check whether host port `3306` is already in use.

### Automation and Firewall Rules

Backup jobs, health checks, firewall rules, CI scripts, and monitoring configuration may contain the old port explicitly. Search deployment configuration for `5455` rather than assuming that updating RAGFlow's image tag is sufficient.

The database data itself is not migrated or rewritten by this port change. The failure is at the network endpoint: clients may be looking at the wrong port, or Docker may be unable to claim the new one.

## How to Preserve Port 5455

There is no requirement to adopt host port `3306`. To keep the v0.24.0 behavior, set the following value in `docker/.env`:

```dotenv
MYSQL_PORT=3306
EXPOSE_MYSQL_PORT=5455
```

Then recreate the affected containers so Docker applies the updated port mapping:

```bash
docker compose up -d --force-recreate mysql
```

Host-side clients should continue connecting to `127.0.0.1:5455`, while RAGFlow containers should continue using `mysql:3306`.

Do not change `MYSQL_PORT` to `5455` merely to preserve the old host endpoint. For the bundled Compose MySQL service, doing so tells RAGFlow to contact the MySQL container on port `5455`, where it is not listening.

## How to Adopt Port 3306

If host port `3306` is available and the new default is preferred, keep both values at `3306`:

```dotenv
MYSQL_PORT=3306
EXPOSE_MYSQL_PORT=3306
```

Update all external clients from `5455` to `3306`, recreate the containers, and verify the resolved Compose configuration:

```bash
docker compose config
docker compose ps
```

The MySQL service should show a published mapping equivalent to `0.0.0.0:3306->3306/tcp` or a loopback-specific equivalent if the deployment restricts the bind address.

## External MySQL Deployments Are Different

When RAGFlow connects to a MySQL server managed outside this Compose stack, `MYSQL_HOST` and `MYSQL_PORT` describe that external endpoint. In that case, `MYSQL_PORT` may legitimately be something other than `3306`:

```dotenv
MYSQL_HOST=db.example.internal
MYSQL_PORT=4406
```

`EXPOSE_MYSQL_PORT` controls publication of the bundled MySQL container and is not the setting RAGFlow uses to connect to an external database. Avoid swapping the meanings of these variables:

| Variable | Purpose |
|---|---|
| `MYSQL_PORT` | Port RAGFlow uses to connect to MySQL |
| `EXPOSE_MYSQL_PORT` | Host port mapped to the bundled MySQL container's port `3306` |

## Security Consideration

Publishing MySQL on `3306` can make the service more obvious to network scanners and may expose it beyond the local machine, depending on Docker and firewall configuration. A nonstandard port is not a security control, but the database should not be publicly reachable unless external access is explicitly required.

Use a strong password, restrict the bind address or firewall rules, and avoid exposing MySQL at all when every database client runs inside the Compose network.

## Upgrade Checklist

Before moving from v0.24.0 to v0.26.4:

1. Inspect the effective value of `EXPOSE_MYSQL_PORT`.
2. Check whether host port `3306` is already occupied.
3. Decide whether to adopt `3306` or preserve `5455`.
4. Update host-side database clients, scripts, and firewall rules.
5. Keep `MYSQL_PORT=3306` for the bundled MySQL service.
6. Recreate the MySQL container and verify the published mapping.
7. Confirm that both RAGFlow and any external database clients can connect.

## Summary

Between RAGFlow v0.24.0 and v0.26.4, the default host-exposed MySQL port changed from `5455` to `3306`. More precisely, `EXPOSE_MYSQL_PORT` changed, while the internal `MYSQL_PORT` remained `3306`.

RAGFlow's container-to-container database connection is therefore unchanged. The impact is on host-side clients and on hosts where port `3306` is already occupied. Deployments can adopt the new default or retain the old behavior by explicitly setting `EXPOSE_MYSQL_PORT=5455`.

## References

- [RAGFlow v0.24.0 `docker/.env`](https://github.com/infiniflow/ragflow/blob/v0.24.0/docker/.env)
- [RAGFlow v0.25.0 `docker/.env`](https://github.com/infiniflow/ragflow/blob/v0.25.0/docker/.env)
- [RAGFlow v0.26.4 `docker/.env`](https://github.com/infiniflow/ragflow/blob/v0.26.4/docker/.env)
- [RAGFlow v0.26.4 Docker Compose base services](https://github.com/infiniflow/ragflow/blob/v0.26.4/docker/docker-compose-base.yml)
