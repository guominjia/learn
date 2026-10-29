---
title: Web Frameworks and HTTP Server
date: 2026-08-25
categories: [web]
tags: [web, framework]
---

## [FastAPI](https://pypi.org/project/fastapi/)
A modern high-performance web framework based on Python type hints (Pydantic) and asynchronous support ([Starlette](https://pypi.org/project/starlette/)), designed for rapidly building APIs.
It handles route definition, request/response logic, data validation, and dependency injection.

## [Flask](https://pypi.org/project/Flask/)
A lightweight, synchronous web framework based on WSGI (Web Server Gateway Interface), widely used for building web applications and APIs.
It provides routing, templating (Jinja2), and request handling, but requires extensions for advanced features like data validation.
Uses a synchronous, blocking execution model.

## [Gunicorn](https://pypi.org/project/gunicorn/)
A WSGI (Web Server Gateway Interface) HTTP server for running Flask (or other WSGI framework) applications.
It is a pre-fork worker model server that spawns multiple worker processes to handle requests.
Each worker handles requests synchronously in a blocking manner.
Can be configured with different worker types (sync, async, threaded) using `--workers` and `--worker-class` parameters.

## [Uvicorn](https://pypi.org/project/uvicorn/)
An ASGI (Asynchronous Server Gateway Interface) server that runs FastAPI (or other ASGI framework) applications.
It listens for HTTP requests, parses protocols, passes requests to FastAPI for processing, and returns responses.
It is based on `uvloop` (high-performance async I/O library) and `httptools` (HTTP protocol parser).
Supports asynchronous operations and high concurrency through event loops.

## [Django](https://pypi.org/project/Django/)
A full-stack (batteries-included) web framework that ships with ORM, templating (Django templates), forms, authentication, and an admin interface out of the box.
Traditionally a synchronous WSGI framework; since 3.0 it supports ASGI (asynchronous) via `asgi.py`, but most built-in components (ORM, middleware) remain synchronous, so async support is progressive and limited.
Heavier than Flask/FastAPI — designed for full web applications rather than pure APIs.

## [Daphne](https://pypi.org/project/daphne/)
An ASGI HTTP server, part of the [Django Channels](https://pypi.org/project/channels/) project, designed to run Django's asynchronous applications (WebSocket, HTTP/2, long-polling).
Same category as Uvicorn — both are ASGI servers — but built on [Twisted](https://pypi.org/project/Twisted/) instead of `uvloop`.
Typical pairing: **Django + Daphne** (async framework + ASGI server), analogous to FastAPI + Uvicorn.

## [http.server](https://docs.python.org/3/library/http.server.html) (Standard Library)
A minimal HTTP server built into Python's standard library, based on `socketserver`.
Parses HTTP request lines and headers (one level above raw `socket`), but provides no routing, no validation, no templating, and no ORM.
Single-threaded by default (use `ThreadingHTTPServer` for multi-threaded), with no async support.
Does not conform to WSGI/ASGI — cannot directly run a FastAPI/Flask/Django app.
Official docs explicitly state it is **not recommended for production**.

## [socket](https://docs.python.org/3/library/socket.html) (Standard Library)
The lowest-level networking module in Python's standard library, operating on raw TCP/UDP byte streams.
Does not understand HTTP at all — you receive raw bytes and must parse the HTTP protocol yourself (request line, headers, body, chunked encoding, etc.).
Uvicorn and Gunicorn ultimately use `socket` under the hood (via `asyncio`/`uvloop` or OS-level syscalls), but wrap it with full protocol parsing and concurrency management.
Writing a web server with bare `socket` means reimplementing what Uvicorn/Gunicorn already provide.

## Request
A Request object provided by FastAPI (through Starlette) or Flask that allows access to detailed client request information (such as request body, headers, cookies, etc.).
- **FastAPI**: `from fastapi import Request`
- **Flask**: `from flask import request`

Can be used directly in route handler functions to manipulate request data.

## Relationship Summary
- **FastAPI** + **Uvicorn**: Asynchronous framework + ASGI server (high concurrency, I/O-intensive)
- **Flask** + **Gunicorn**: Synchronous framework + WSGI server (traditional multi-process model)
- **Django** + **Gunicorn** (WSGI) or **Django** + **Daphne**/**Uvicorn** (ASGI): Full-stack framework with a choice of sync or async server
- **Request** is a tool provided by both frameworks for handling request content

## Framework vs Server: Dependency and Responsibilities

Frameworks and servers are **independent packages** — installing one does not automatically install the other:

| Package | Auto-installs the other? | Notes |
|---|---|---|
| `pip install fastapi` | ❌ Does not install Uvicorn | FastAPI's dependencies do not include Uvicorn |
| `pip install flask` | ❌ Does not install Gunicorn | Flask's dependencies do not include Gunicorn |
| `pip install uvicorn` | ❌ Does not install any framework | Uvicorn is just a server |
| `pip install gunicorn` | ❌ Does not install any framework | Gunicorn is just a server |

Both must be installed manually:

```bash
pip install fastapi uvicorn   # framework + server, both required
pip install flask gunicorn    # same
```

### Why neither works alone

A **server** (Uvicorn/Gunicorn) needs an **application object** to serve, and a **framework** (FastAPI/Flask/Django) defines that object but has no ability to listen on a port:

```
Your code → Framework (FastAPI/Flask/Django) defines `app` → Server (Uvicorn/Gunicorn/Daphne) runs `app`
                 ↑ required                                      ↑ required
           (routing, validation, response building)        (network I/O, concurrency)
```

- **Framework without server**: no port-listening capability — the app cannot accept connections.
- **Server without framework**: no application logic to route requests or build responses — nothing to serve.

### Exception: built-in dev servers

Both Flask and FastAPI provide a **development-only** server that does not require a separate ASGI/WSGI server:

```bash
flask --app main run        # Flask's built-in Werkzeug dev server
fastapi dev main.py         # FastAPI CLI — internally still calls Uvicorn
```

Flask's dev server explicitly warns **not to use in production**.
`fastapi dev` relies on Uvicorn under the hood (the `fastapi[standard]` extra pulls in Uvicorn), so it is not truly serverless — it just hides the installation step.

## Abstraction Layers: socket → http.server → ASGI/WSGI Server → Framework

`http.server` and `socket` are Python standard library modules at lower abstraction layers than Uvicorn/Gunicorn. They are not alternatives to a framework — they are the building blocks underneath:

```
Your code
  ↓
Framework:        FastAPI / Flask / Django          ← routing, validation, response building
  ↓
ASGI/WSGI Server: Uvicorn / Gunicorn / Daphne       ← protocol handling, concurrency model
  ↓
Standard Library: http.server / socket              ← HTTP parsing / raw TCP byte stream
```

### Comparison

| | `socket` | `http.server` | Uvicorn / Gunicorn | FastAPI / Flask / Django |
|---|---|---|---|---|
| **Layer** | Raw TCP byte stream | HTTP protocol parsing | ASGI/WSGI server | Web framework |
| **Understands HTTP?** | ❌ | ✅ Basic parsing | ✅ Full | ✅ (via server) |
| **Routing / Validation?** | ❌ | ❌ | ❌ | ✅ |
| **Concurrency model** | DIY | Single-threaded / threaded | Event loop / multi-process | Delegates to server |
| **WSGI / ASGI compliant?** | ❌ | ❌ | ✅ | ✅ |
| **Production-ready?** | Needs heavy wrapping | ❌ | ✅ | ✅ (with server) |
| **Standard library?** | ✅ | ✅ | ❌ `pip install` | ❌ `pip install` |

**Key distinction**: Uvicorn/Gunicorn are **ASGI/WSGI-compliant** servers that accept a framework's `app` object directly; `http.server` does not conform to these specs and cannot run a FastAPI/Flask/Django app; `socket` does not even understand HTTP. The three are progressive abstraction layers, not substitutes.

## Performance Comparison

### Uvicorn + FastAPI (Async Model)
- Single process can handle thousands of concurrent connections using async/await
- Excellent for I/O-intensive tasks (database queries, API calls)
- Low resource consumption per request
- Poor for CPU-intensive tasks (blocks event loop)

### Gunicorn + Flask (Sync Model)
- Each request occupies one worker process/thread
- Higher resource consumption under high concurrency
- Straightforward for CPU-bound tasks (each worker is independent)
- Scales by adding more worker processes

## Concurrency vs Parallelism

**Concurrency**: Multiple tasks execute during overlapping time periods, appearing simultaneous logically but may alternate physically (e.g., through time-slicing). Focuses on improving resource utilization.
- Works on single-core CPUs through context switching
- Ideal for I/O-intensive tasks
- Implemented via threads or coroutines (e.g., Python's asyncio)

**Parallelism**: Multiple tasks physically execute simultaneously, requiring multi-core or multi-processor support. Focuses on reducing total execution time.
- Requires multi-core CPUs or distributed systems
- Ideal for CPU-intensive tasks
- Implemented via multiple processes or multi-core threading

**Key Distinction**: Concurrency is about dealing with multiple tasks (logical design), while parallelism is about doing multiple tasks simultaneously (physical execution).
