---
title: Proxy Environment Variables Can Break RAGFlow MCP Services
categories: [ai, rag, ragflow]
tags: [ragflow, mcp, proxy, http-proxy, no-proxy, troubleshooting]
---

Proxy environment variables such as `http_proxy` and `https_proxy` can unexpectedly affect Model Context Protocol (MCP) services used by RAGFlow. In our deployment, MCP communication failed while these variables were enabled. Adding the MCP endpoints to `no_proxy` did not resolve the problem.

The temporary workaround is to run RAGFlow and its MCP-related processes without proxy environment variables. This restores MCP connectivity, but it has an important side effect: access to external websites and APIs may become limited or unavailable.

## Symptoms

The problem can appear as one or more of the following symptoms:

- RAGFlow cannot connect to an MCP server.
- MCP tool discovery times out or returns a connection error.
- A local MCP endpoint is reachable with a direct test but fails from the application.
- Requests are unexpectedly sent to the configured proxy.
- Adding the target host to `no_proxy` or `NO_PROXY` does not change the result.

These symptoms can be confusing because the MCP server itself may be healthy. The failure occurs in the client-side network path selected by the RAGFlow process or one of its dependencies.

## Why Proxy Variables Affect MCP

Many HTTP clients automatically read proxy settings from the process environment:

```text
http_proxy
https_proxy
HTTP_PROXY
HTTPS_PROXY
ALL_PROXY
no_proxy
NO_PROXY
```

When RAGFlow connects to an HTTP-based MCP server, the underlying client library may route the request through `http_proxy` or `https_proxy`. This also applies to long-lived transports such as Server-Sent Events and streamable HTTP. A proxy that handles ordinary web requests correctly may still interfere with streaming, connection reuse, local addresses, authentication headers, or protocol upgrades.

The effective path can therefore become:

```text
RAGFlow -> proxy server -> MCP server
```

instead of the expected direct connection:

```text
RAGFlow -> MCP server
```

For a local or internal MCP service, sending traffic through an external proxy can result in timeouts, refused connections, DNS failures, or incomplete streaming responses.

## Why `no_proxy` May Not Be Enough

In principle, `no_proxy` should make selected destinations bypass the proxy. In practice, its behavior is not completely consistent across operating systems, runtimes, and HTTP libraries.

Common sources of mismatch include:

- One component reads `NO_PROXY`, while another reads `no_proxy`.
- The application and a child process inherit different environments.
- The configured hostname does not match the hostname used in the final request.
- `localhost`, `127.0.0.1`, a container name, and a host IP are treated as different destinations.
- Port-specific entries are interpreted differently by different libraries.
- CIDR notation or wildcard syntax is unsupported by the active HTTP client.
- `ALL_PROXY` remains set and is used as a fallback.
- A process was not restarted after the environment changed.
- Docker Compose continues to inject proxy variables into a container.

For example, bypassing `localhost` does not necessarily bypass a request sent to `host.docker.internal`, an MCP container name, or the resolved IP address. Similarly, changing the host environment does not modify the environment of an already running container.

In this deployment, setting both lowercase and uppercase `no_proxy` values still did not restore reliable MCP communication. Rather than depending on library-specific bypass behavior, we removed the proxy variables from the MCP execution path.

## Temporary Workaround: Disable the Proxy

Before starting RAGFlow, remove all proxy-related variables from the shell environment:

```bash
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
```

If the variables are defined in a Docker Compose environment file or service definition, remove them or assign empty values:

```dotenv
http_proxy=
https_proxy=
HTTP_PROXY=
HTTPS_PROXY=
ALL_PROXY=
all_proxy=
```

Recreate the affected containers after changing their environment. Restarting only the MCP server is insufficient if the RAGFlow container still has the old proxy variables.

The effective environment should be checked inside the same process boundary that makes the MCP request. For a containerized deployment, inspect the RAGFlow container rather than relying only on the host shell.

Do not print the full environment into shared logs because it may contain credentials. Inspect only the proxy-related variable names and redact any proxy URL that embeds a username or password.

## The Trade-Off

Disabling the proxy fixes the internal MCP path by forcing direct connections, but it can break outbound access:

```text
Internal MCP access: restored
External Internet access: potentially restricted
```

Without a proxy, RAGFlow may be unable to reach services that are available only through the corporate or deployment proxy. Possible impacts include:

- Downloading models or files from external repositories.
- Calling hosted LLM, embedding, reranking, or OCR APIs.
- Accessing public websites during retrieval or crawling.
- Installing packages or downloading runtime resources.
- Connecting to externally hosted MCP servers.

This workaround is therefore appropriate only when MCP availability is currently more important than unrestricted external access, or when all required external resources are available through internal mirrors and gateways.

## Verification

After recreating the services without proxy variables, verify both sides of the trade-off:

1. Confirm that RAGFlow can establish a direct connection to the MCP endpoint.
2. Confirm that MCP tool discovery completes successfully.
3. Invoke a simple MCP tool and check that its response is complete.
4. Test any external model providers or APIs required by the deployment.
5. Record which external destinations are no longer reachable.

Testing only the MCP server's health endpoint is not enough. The final verification should use RAGFlow's actual MCP client path because that is where environment inheritance and HTTP-library behavior apply.

## Long-Term Options

Running the entire application without a proxy is a broad workaround. A more precise long-term design should separate internal MCP traffic from outbound Internet traffic. Possible approaches include:

- Run the MCP client or gateway in a process with a clean environment while allowing other components to use the proxy.
- Use a proxy with explicit and tested rules for internal MCP hosts.
- Add an internal gateway that routes MCP traffic directly and external traffic through the proxy.
- Standardize MCP endpoint names so that bypass rules do not depend on changing IP addresses.
- Identify the exact HTTP client used by the failing path and validate its `no_proxy` syntax and streaming support.
- Add startup diagnostics that report whether an MCP endpoint will use a proxy, without exposing credentials.

The preferred solution is not to disable all outbound networking permanently. It is to make proxy selection explicit for each traffic class.

## Summary

Proxy environment variables can change how RAGFlow reaches HTTP-based MCP services. In the observed deployment, `http_proxy` and `https_proxy` disrupted MCP communication, and adding `no_proxy` did not reliably bypass the proxy.

The current workaround is to remove proxy variables from RAGFlow and its MCP-related processes, then recreate the affected containers. This restores direct MCP connectivity, but external Internet access may become restricted. Treat this as a temporary operational compromise while implementing separate routing policies for internal MCP traffic and external services.
