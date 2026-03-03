# 本地 Uvicorn + FastAPI 获取 IP

## 直接使用 `Request` 对象

````python
from fastapi import FastAPI, Request

app = FastAPI()

@app.get("/test")
async def test(request: Request):
    ip = request.client.host  # 获取客户端IP
    port = request.client.port  # 获取客户端Port
    return {
        "ip": ip,
        "port": port
    }
````

## 说明

| 属性 | 说明 |
|------|------|
| `request.client.host` | 客户端 IP |
| `request.client.port` | 客户端 Port |
| `request.headers.get('X-Forwarded-For')` | 经过代理时使用 |

## 本地直连 Uvicorn 的情况

```
浏览器 ──直接请求──> Uvicorn/FastAPI
```

本地没有 Nginx，所以：
- ✅ `request.client.host` **可以直接用**
- ❌ `X-Real-IP` / `X-Forwarded-For` **不存在**

> **注意**：FastAPI 中不能用 `request.remote_addr`，那是 Flask 的写法，FastAPI 要用 `request.client.host`

# 为何看不到 X-Real-IP

## 原因

`X-Real-IP` 是**请求头（Request Header）**，是客户端发给服务器的，但它实际上是由 **Nginx 添加的**，不是浏览器发送的。

## 流程说明

```
浏览器 ──请求──> Nginx ──添加X-Real-IP──> Flask服务器
         ↑                    ↑
   浏览器不会发送          Nginx在这里加上
   X-Real-IP               客户端真实IP
```

## 为什么开发者工具看不到

浏览器开发者工具显示的是**浏览器自己发出的请求头**，而 `X-Real-IP` 是 Nginx 转发时**额外添加**的，所以：

- ✅ **Flask 服务端**可以收到 `X-Real-IP`
- ❌ **浏览器开发者工具**看不到（因为是 Nginx 加的，不是浏览器发的）

## 验证方法

在 Flask 中打印所有请求头来确认：

````python
@app.route('/test')
def test():
    # 打印所有请求头
    headers = dict(request.headers)
    print(headers)
    
    real_ip = request.headers.get('X-Real-IP')
    forwarded_for = request.headers.get('X-Forwarded-For')
    
    return {
        "X-Real-IP": real_ip,
        "X-Forwarded-For": forwarded_for,
        "remote_addr": request.remote_addr
    }
````

> **注意**：如果你是本地直接运行 Flask（没有 Nginx），`X-Real-IP` 不会存在，直接用 `request.remote_addr` 即可。

# 获取 Remote Address

浏览器开发者工具中看到的 **Remote Address** 是**服务器的 IP:Port**，不是客户端的 IP。

## 示意图

```
浏览器 (客户端)  ──请求──>  服务器 (Remote Address)
你的IP                      服务器IP:Port  ← 开发者工具显示的这个
```

所以这个值是你**服务器的地址**，不需要从 Request 获取。

---

## 如果你想获取服务器自身的 IP/Port

````python
from flask import request

@app.route('/test')
def test():
    # 服务器 host
    host = request.host          # 例如: "127.0.0.1:5000"
    
    # 分开获取
    host_url = request.host_url  # 例如: "http://127.0.0.1:5000/"
    
    return f"Server Host: {host}"
````

---

## 总结

| 目标 | 方法 |
|------|------|
| 获取**客户端（访问者）IP** | `request.remote_addr` 或 `X-Forwarded-For` |
| 获取**服务器地址**（开发者工具的 Remote Address） | `request.host` |
