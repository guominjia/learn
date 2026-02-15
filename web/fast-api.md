## FastAPI
A modern high-performance web framework based on Python type hints (Pydantic) and asynchronous support (Starlette), designed for rapidly building APIs.
It handles route definition, request/response logic, data validation, and dependency injection.

## [Flask](https://github.com/guominjia/learn/blob/code_study/flask/flask_server.py)
A lightweight, synchronous web framework based on WSGI (Web Server Gateway Interface), widely used for building web applications and APIs.
It provides routing, templating (Jinja2), and request handling, but requires extensions for advanced features like data validation.
Uses a synchronous, blocking execution model.

## Gunicorn
A WSGI (Web Server Gateway Interface) HTTP server for running Flask (or other WSGI framework) applications.
It is a pre-fork worker model server that spawns multiple worker processes to handle requests.
Each worker handles requests synchronously in a blocking manner.
Can be configured with different worker types (sync, async, threaded) using `--workers` and `--worker-class` parameters.

## Uvicorn
An ASGI (Asynchronous Server Gateway Interface) server that runs FastAPI (or other ASGI framework) applications.
It listens for HTTP requests, parses protocols, passes requests to FastAPI for processing, and returns responses.
It is based on `uvloop` (high-performance async I/O library) and `httptools` (HTTP protocol parser).
Supports asynchronous operations and high concurrency through event loops.

## Request
A Request object provided by FastAPI (through Starlette) or Flask that allows access to detailed client request information (such as request body, headers, cookies, etc.).
- **FastAPI**: `from fastapi import Request`
- **Flask**: `from flask import request`

Can be used directly in route handler functions to manipulate request data.

## Relationship Summary
- **FastAPI** + **Uvicorn**: Asynchronous framework + ASGI server (high concurrency, I/O-intensive)
- **Flask** + **Gunicorn**: Synchronous framework + WSGI server (traditional multi-process model)
- **Request** is a tool provided by both frameworks for handling request content

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
