---
title: Add a RAGFlow MCP Server to VS Code
categories: [AI, RAGFlow]
tags: [ragflow, mcp, vscode]
---

RAGFlow can expose its capabilities through the Model Context Protocol (MCP). By registering the RAGFlow MCP endpoint in VS Code, GitHub Copilot and other MCP-compatible tools can connect to the server without storing the API key directly in the configuration file.

## Prerequisites

Before configuring VS Code, make sure that:

- The RAGFlow MCP server is running and reachable from your computer.
- You have a valid RAGFlow API key.
- Your version of VS Code supports MCP servers.

## Configure the MCP Server

Open the VS Code MCP configuration file and add the RAGFlow server to `mcp.json`. A workspace-specific configuration is normally stored in `.vscode/mcp.json`.

Use the following complete configuration:

```json
{
	"servers": {
		"ragflow-mcp-server": {
			"url": "http://ragflow-server.com:9382/mcp",
			"type": "http",
			"headers": {
				"Authorization": "Bearer ${input:ragflow-api-key}"
			}
		}
	},
	"inputs": [
		{
			"id": "ragflow-api-key",
			"type": "promptString",
			"description": "RAGFlow API Key",
			"password": true
		}
	]
}
```

If `mcp.json` already contains other servers, add only the `ragflow-mcp-server` object under `servers` and append the `ragflow-api-key` object to the existing `inputs` array.

## How the Configuration Works

- `url` specifies the RAGFlow MCP HTTP endpoint.
- `type` tells VS Code to connect using HTTP.
- `Authorization` sends the API key as a Bearer token with every request.
- `${input:ragflow-api-key}` refers to the entry with the matching `id` in the `inputs` array.
- `promptString` asks for the API key when VS Code starts the server.
- `password: true` masks the API key while it is being entered.

This input-based approach is safer than writing the API key directly in `mcp.json`, especially when the configuration is committed to a Git repository.

## Start and Verify the Connection

After saving the file:

1. Open the MCP server list in VS Code.
2. Start `ragflow-mcp-server`.
3. Enter the RAGFlow API key when prompted.
4. Confirm that the server starts successfully and that its tools are available in Chat.

If the connection fails, verify that the endpoint is reachable, the API key is valid, and no proxy or firewall is blocking port `9382`.

## Security Note

The example endpoint uses plain HTTP. Use HTTPS whenever the MCP traffic crosses an untrusted network because the Authorization header contains a reusable credential. Never hard-code or commit the API key.

## Summary

VS Code can connect to RAGFlow through a small MCP configuration. Defining the API key as a password input keeps the credential out of source control while still allowing VS Code to authenticate each request automatically.
