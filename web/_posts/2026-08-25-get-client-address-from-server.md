---
title: "Getting the Client's IP in FastAPI/Flask, and Why X-Real-IP Isn't There Locally"
categories: [web]
tags: [fastapi, flask, uvicorn, nginx, http]
---

Three things that get confused together whenever "how do I get the client's IP" comes up: which attribute to read, why proxy headers exist at all, and what the browser DevTools "Remote Address" field actually means.

## Client IP in FastAPI

FastAPI's `Request` doesn't have `remote_addr` (that's Flask). The client address is on `request.client`:

```python
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/test")
async def test(request: Request):
    return {
        "ip": request.client.host,
        "port": request.client.port,
    }
```

| Attribute | Meaning |
|---|---|
| `request.client.host` | IP of the socket that connected to Uvicorn |
| `request.client.port` | Its ephemeral source port |
| `request.headers.get("X-Forwarded-For")` | Only set if something in front of Uvicorn adds it |

## Client IP in Flask

The Flask equivalent is `request.remote_addr`:

```python
from flask import request

@app.route("/test")
def test():
    return {
        "remote_addr": request.remote_addr,
        "x_real_ip": request.headers.get("X-Real-IP"),
        "x_forwarded_for": request.headers.get("X-Forwarded-For"),
    }
```

## Why X-Real-IP/X-Forwarded-For are empty locally

`X-Real-IP` and `X-Forwarded-For` are request headers, but the browser never sends them — a reverse proxy adds them when it forwards the request:

```
browser ──request──> Nginx ──adds X-Real-IP──> Flask/Uvicorn
              ↑                     ↑
     browser doesn't send      Nginx inserts the
     this header               real client IP here
```

Run the app directly (`uvicorn main:app`, no Nginx in front), and `request.client.host`/`request.remote_addr` already is the real client IP — there's no proxy to have stripped or rewritten it, so the `X-Forwarded-For`/`X-Real-IP` headers simply don't exist yet. That's also why they're invisible in browser DevTools: DevTools only shows what the browser sent, and these headers are added downstream by the proxy, not by the browser.

## "Remote Address" in DevTools isn't the client's IP

The **Remote Address** field in the Network tab is the address the browser connected *to* — the server (or the nearest proxy/CDN in front of it) — not the browser's own address:

```
browser (client) ──request──> server
   your IP                    Remote Address ← what DevTools shows
```

If you need the server's own host/port inside a handler, that comes from the request line, not from anything client-supplied:

```python
# FastAPI
request.url.hostname, request.url.port

# Flask
request.host        # "127.0.0.1:5000"
request.host_url     # "http://127.0.0.1:5000/"
```

## Quick reference

| Goal | FastAPI | Flask |
|---|---|---|
| Client IP, no proxy | `request.client.host` | `request.remote_addr` |
| Client IP, behind Nginx | `request.headers["X-Forwarded-For"]` | `request.headers.get("X-Forwarded-For")` |
| Server's own address | `request.url.hostname` | `request.host` |
