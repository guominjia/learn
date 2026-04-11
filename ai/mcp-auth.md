# MCP Authorization Quickstart (Linux)

Links
- Authentication README: https://github.com/modelcontextprotocol/python-sdk?tab=readme-ov-file#authentication
- Simple auth example: https://github.com/modelcontextprotocol/python-sdk/tree/main/examples/servers/simple-auth

## Prerequisites
- Linux with Bash shell
- Python 3.10+ and pip
- Git (to clone the examples)
- Free ports: 9000 (Auth server), 8001 (Resource server)

Optional but recommended (per-project env):
```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 1) Get the example code
If you already have the MCP Python SDK locally, skip cloning and just `cd` as shown below.

```bash
git clone https://github.com/modelcontextprotocol/python-sdk.git
```

## 2) Start the Auth Server
Run the in-repo auth server for local testing.

```bash
cd python-sdk/examples/servers/simple-auth
python3 -c "import mcp_simple_auth.auth_server as a; a.main()"
```

- Default listen address: http://localhost:9000

## 3) Start the Resource Server
Install minimal deps and start the resource server with streamable HTTP transport.

```bash
cd python-sdk/examples/servers/simple-auth
python3 -m pip install --upgrade pip
python3 -m pip install click pydantic pydantic_settings mcp
python3 -m mcp_simple_auth --port 8001 --auth-server http://localhost:9000 --transport streamable-http
```

- Resource server: http://localhost:8001
- Delegates auth to the Auth Server on port 9000.

## 4) Run the Client
Set env vars and run the demo client.

```bash
cd python-sdk/examples/clients/simple-auth-client
export MCP_SERVER_PORT=8001
export MCP_TRANSPORT_TYPE=streamable_http
python3 mcp_simple_auth_client/main.py
```

### PowerShell equivalents (optional)
If you’re on Windows PowerShell instead of Bash:

```powershell
cd python-sdk/examples/clients/simple-auth-client
$env:MCP_SERVER_PORT = 8001
$env:MCP_TRANSPORT_TYPE = 'streamable_http'
python mcp_simple_auth_client/main.py
```

## Notes & Troubleshooting
- Transport naming: CLI flag uses `--transport streamable-http`; env var uses `streamable_http` (hyphen vs underscore is expected).
- Python executable: if `python3` isn’t present, use `python` instead. Prefer `python3 -m pip` to avoid `pip` vs `pip3` confusion.
- If you see `ModuleNotFoundError`, make sure you’re inside the cloned `python-sdk` tree or your active environment can import the package.
- If a port is in use, change `--port` for the resource server and update `MCP_SERVER_PORT` accordingly.