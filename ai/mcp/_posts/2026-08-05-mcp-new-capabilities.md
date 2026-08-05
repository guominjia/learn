---
title: MCP 2026-07-28: Stateless Servers, Multi-Round-Trip Requests, and Safer Auth
categories: [ai, mcp]
tags: [mcp, protocol, agents, oauth, architecture]
---

The `2026-07-28` Model Context Protocol (MCP) specification is a substantial
change in how remote MCP services are designed and operated. Its central idea is
simple: an MCP request should be self-contained. That moves the protocol closer
to a conventional HTTP workload that can be routed, cached, and scaled without
transport-level session affinity.

This post explains the changes that matter most when building an MCP client,
server, or gateway, and outlines a practical migration path.

## The Core Change: Stateless Requests

Earlier remote MCP implementations relied on an initialization exchange and a
protocol-level session identifier. In the new specification, the
`initialize`/`notifications/initialized` handshake and the `Mcp-Session-Id`
header are removed from Streamable HTTP.

Each request instead carries its protocol version and client capabilities in
`_meta`. A client should also identify itself there. A server can expose its
supported versions, capabilities, and identity through `server/discover`, which
clients may call before other requests when they need up-front negotiation.

This changes the deployment model:

- A request can reach any healthy server instance behind an ordinary load balancer.
- A gateway no longer needs shared session storage merely to route MCP traffic.
- Failover becomes less coupled to a particular server process or long-lived connection.

Stateless transport does **not** require a stateless application. A server that
needs continuity across tool calls should return an explicit, server-minted
handle from one tool call and accept that handle as an ordinary argument in a
later call. This makes the state dependency visible in the tool contract rather
than implicit in the HTTP transport.

## Multi-Round-Trip Requests Replace Server-Initiated Calls

A stateless protocol still needs to handle a tool that cannot finish its work
immediately. For example, a server might need a missing value, an approval, or
an answer from the user before it can perform an operation.

MCP now defines the **Multi Round-Trip Requests (MRTR)** pattern for that case.
Instead of sending a server-initiated request over a held-open stream, a server
returns a result with `resultType: "input_required"`. The result contains
`inputRequests`; the client collects the answers and retries the original call
with `inputResponses`.

At a high level, the flow is:

```text
client -> tools/call
server -> input_required + inputRequests
client -> tools/call (retry with inputResponses)
server -> complete
```

MRTR replaces the previous server-initiated patterns for Roots, Sampling, and
elicitation. It is especially useful for approval gates: a destructive or
billable operation can ask for confirmation without making the protocol depend
on a persistent bidirectional connection.

## Better Infrastructure Boundaries

The release adds two details that are small at the wire level but important in
production.

### Header-Based Routing

Streamable HTTP `POST` requests now require `Mcp-Method` and `Mcp-Name`
headers. A reverse proxy, WAF, authorization layer, or rate limiter can use
these headers to apply rules without parsing a JSON-RPC body.

For example, an organization can separately meter a high-cost tool invocation
from a read-only resource request, while retaining a normal HTTP gateway
architecture.

### Cacheable Catalogs and Resources

Results from `tools/list`, `prompts/list`, `resources/list`, `resources/read`,
and `resources/templates/list` include `ttlMs` and `cacheScope`. The first is a
freshness hint; the second distinguishes `public` from `private` cacheability.

Clients can therefore avoid repeatedly fetching tool catalogs, while shared
intermediaries can avoid caching user-specific results incorrectly. Servers
should also return tools in a deterministic order, which stabilizes client
caches and upstream model prompt caches.

## Authentication Is More Explicit

The specification also hardens OAuth-related behavior:

- Authorization servers should include `iss` in authorization responses, and
	clients must validate it against the recorded issuer before exchanging the
	authorization code.
- Clients must specify a suitable `application_type` when using Dynamic Client
	Registration (DCR), helping desktop and CLI redirect handling align with
	OpenID Connect requirements.
- Persisted client credentials are bound to the authorization-server issuer and
	must not be reused with another issuer.
- DCR is deprecated in favor of Client ID Metadata Documents (CIMD), although
	it remains available for backward compatibility during the transition.

For client authors, this means credential storage should be keyed by issuer,
not merely by a logical server name. For server and identity teams, it is a good
time to plan CIMD support rather than extending a new DCR-only integration.

## Tasks and the Extension Model

Tasks are no longer experimental core protocol functionality. They are now an
official `io.modelcontextprotocol/tasks` extension with polling through
`tasks/get` and client-to-server input through `tasks/update`.

This makes long-running work a modular opt-in capability: clients and servers
advertise extension support, then use task handles for asynchronous operations.
It is a better fit for agent workflows that may run longer than a normal request
or require input while work is in progress.

## What Is Being Deprecated

The release establishes a feature lifecycle policy with at least a twelve-month
deprecation window. Roots, Sampling, Logging, and the legacy HTTP+SSE transport
are now deprecated. They continue to work during that window, but new
implementations should avoid adopting them.

The suggested direction is to pass directories and files as tool arguments,
resource URIs, or server configuration instead of Roots; call the model provider
directly rather than using Sampling; use `stderr` or OpenTelemetry instead of
protocol Logging; and move HTTP+SSE deployments to Streamable HTTP.

## A Practical Migration Checklist

1. Upgrade the MCP SDK and select protocol version `2026-07-28`.
2. Remove reliance on `Mcp-Session-Id`, initialization state, and sticky routing.
3. Move necessary cross-call state into explicit, server-issued tool arguments.
4. Implement `server/discover` and include the required per-request metadata.
5. Support MRTR retries, including `input_required`, `inputRequests`, and
	 `inputResponses`.
6. Update gateways to consume `Mcp-Method` and `Mcp-Name` headers.
7. Honor `ttlMs` and `cacheScope` in clients and set appropriate values in servers.
8. Audit OAuth code for issuer validation, issuer-scoped credential storage, and
	 a path from DCR to CIMD.
9. Plan replacements for deprecated Roots, Sampling, Logging, and HTTP+SSE uses.

The result is not merely a different initialization sequence. MCP 2026-07-28
defines a more web-native operating model: explicit state, request-level
metadata, cache-aware catalogs, and extension-based long-running work. That is
the foundation needed to run MCP services behind standard production
infrastructure without treating each server as a special persistent connection.

## References

- [The 2026-07-28 Specification announcement](https://blog.modelcontextprotocol.io/posts/2026-07-28/): release overview covering the stateless core, MRTR, routing, caching, authorization, Tasks, and SDK availability.
- [MCP 2026-07-28 Specification](https://modelcontextprotocol.io/specification/2026-07-28): authoritative overview of the protocol's self-contained requests, capabilities, extensions, and security guidance.
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog): authoritative migration details for session removal, MRTR, headers, cache metadata, authorization changes, Tasks, and deprecated features.
