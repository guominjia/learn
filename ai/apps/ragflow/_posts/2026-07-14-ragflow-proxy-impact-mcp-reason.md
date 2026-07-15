---
title: Reliable NO_PROXY Rules for RAGFlow MCP Clients in Docker
categories: [ai, rag, ragflow]
tags: [ragflow, mcp, docker, docker-compose, proxy, no-proxy, httpx]
---

Proxy environment variables can prevent RAGFlow from reaching a local or internal Model Context Protocol (MCP) endpoint. A common response is to add `localhost` or a CIDR block such as `127.0.0.0/8` to `NO_PROXY`. That may work in one environment and fail in another.

The reliable approach is to inject proxy variables into the correct container explicitly, use the exact host names or IP addresses that the application requests, and avoid relying on CIDR support unless the active HTTP client documents it.

## Two Common Misunderstandings

### A Compose environment file does not automatically populate a container

Docker Compose commonly reads a project `.env` file for variable substitution. Reading that file does **not** mean every variable becomes part of a service's runtime environment.

For example, this project-level file can provide a value for `${NO_PROXY}`, but it does not by itself make `NO_PROXY` visible to the application in a container:

```dotenv
NO_PROXY=localhost,127.0.0.1,::1
```

Inject the variable explicitly through `environment` or `env_file` instead:

```yaml
services:
  ragflow:
    environment:
      NO_PROXY: ${NO_PROXY}
      no_proxy: ${NO_PROXY}
      HTTP_PROXY: ${HTTP_PROXY}
      HTTPS_PROXY: ${HTTPS_PROXY}
```

Alternatively, use a service environment file:

```yaml
services:
  ragflow:
    env_file:
      - ./ragflow.env
```

The `ragflow.env` file is then part of the service configuration and its values are available to the process in the container.

After changing the service environment, recreate the affected container. A restarted application process cannot see environment values that Docker never injected.

### CIDR notation is not a portable `NO_PROXY` rule

Entries such as `10.0.0.0/8` and `127.0.0.0/8` are not defined consistently across HTTP clients. Treating CIDR as a universal `NO_PROXY` feature makes deployments fragile.

The most broadly reliable forms are:

- an exact host name, such as `localhost`;
- an exact IP address, such as `127.0.0.1`;
- a domain suffix, such as `.yourdomain.com`;
- a host with an optional port, such as `localhost:9382`.

For loopback traffic, include both address families explicitly:

```dotenv
NO_PROXY=localhost,127.0.0.1,::1
no_proxy=localhost,127.0.0.1,::1
```

Keep both uppercase and lowercase forms when multiple tools or subprocesses are involved. Different clients and runtimes do not always use the same spelling.

## `localhost` Means the Current Container

Inside a container, `localhost` and `127.0.0.1` point back to that same container. They do not refer to the Windows host.

When a container must connect to a service running on Docker Desktop's Windows host, use `host.docker.internal` as the target and include that exact name in the bypass list:

```dotenv
NO_PROXY=localhost,127.0.0.1,::1,host.docker.internal
no_proxy=localhost,127.0.0.1,::1,host.docker.internal
```

If `host.docker.internal` is omitted, it does not match a `localhost` rule and can be sent through the proxy. Likewise, adding `localhost` does not bypass a request made to a container DNS name, an internal load balancer name, or a different IP address.

The effective destination must match the actual URL used by the MCP client, not the address that was intended during configuration.

## Python Client Behavior Is Different by Library

Python does not provide one uniform implementation of proxy bypass rules. The library making the MCP request determines what syntax is honored.

| Client | Practical guidance for CIDR entries |
| --- | --- |
| `urllib.request` | Does not support CIDR in `proxy_bypass_environment()`. Use exact hosts, suffixes, IPs, and optional ports. |
| `requests` | Supports IPv4 CIDR entries such as `127.0.0.0/8` and `10.0.0.0/8`. |
| `httpx` | Do not depend on CIDR behavior. Prefer explicit host names and IP addresses. |
| `aiohttp`, SDKs, and curl-based clients | Validate the implementation and version in use; behavior varies. |

For an MCP Python client using `httpx`, the conservative configuration is:

```dotenv
NO_PROXY=localhost,127.0.0.1,::1,host.docker.internal
no_proxy=localhost,127.0.0.1,::1,host.docker.internal
```

If the service deliberately uses another loopback address, for example `127.0.0.2`, add that exact address. Do not assume that `127.0.0.0/8` will work for every client.

## Docker Has Two Independent Proxy Layers

It is useful to separate two configurations that are often confused:

1. **Application proxy configuration** controls requests made by a process inside the running container. `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY` affect this layer.
2. **Docker daemon or BuildKit proxy configuration** controls image pulls and image builds. Its configuration source is separate from the container environment.

Setting `NO_PROXY` for a Compose service can affect the RAGFlow MCP client, but it does not automatically change the proxy used to pull an image or build a Dockerfile. Configure those layers independently when necessary.

## A Safer Compose Example

The following pattern keeps proxy configuration focused on the application process:

```yaml
services:
  ragflow:
    environment:
      HTTP_PROXY: ${HTTP_PROXY:-}
      HTTPS_PROXY: ${HTTPS_PROXY:-}
      ALL_PROXY: ${ALL_PROXY:-}
      NO_PROXY: localhost,127.0.0.1,::1,host.docker.internal
      http_proxy: ${HTTP_PROXY:-}
      https_proxy: ${HTTPS_PROXY:-}
      all_proxy: ${ALL_PROXY:-}
      no_proxy: localhost,127.0.0.1,::1,host.docker.internal
```

Extend the two bypass entries with the exact internal MCP names that RAGFlow uses, for example `ragflow-mcp`, `mcp-gateway.internal`, or a specific private IP address. Do not add a broad CIDR block unless it has been tested with the exact client library and version.

## Troubleshooting a Request That Still Uses the Proxy

If an MCP request still goes through the proxy, check these points in order:

1. Inspect the proxy-related variables inside the running RAGFlow container. Do not rely on the host shell or the project `.env` file.
2. Confirm the final MCP endpoint URL. It may use `host.docker.internal`, `::1`, a container DNS name, or an IP address instead of the literal `localhost`.
3. Add that exact target to both `NO_PROXY` and `no_proxy`.
4. Identify the HTTP library used by the MCP path and test the syntax against that library.
5. Check whether `ALL_PROXY` or `all_proxy` remains set.
6. Recreate the container after changing its service environment.

Avoid logging complete environment dumps in shared logs. Proxy URLs can contain usernames, passwords, or other sensitive information. Log only the relevant variable names and redact credentials.

## Summary

Reliable proxy bypasses for RAGFlow MCP services require exact matching and explicit environment injection. A Compose project `.env` file alone does not place `NO_PROXY` in a container, and CIDR entries are not portable across Python HTTP clients.

For an `httpx`-based MCP client, explicitly include `localhost`, `127.0.0.1`, `::1`, and `host.docker.internal` when applicable. Then add the precise host names or IP addresses used by the service, recreate the container, and verify the variables from within the process boundary that makes the request.