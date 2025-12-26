async def test_github_copilot():
    import os, httpx
    from mcp import ClientSession
    from mcp.client.streamable_http import streamable_http_client
    github_token = os.getenv("GITHUB_TOKEN")
    headers = {"Authorization": f"Bearer {github_token}"}
    async with httpx.AsyncClient(headers=headers) as client:
        async with streamable_http_client("https://api.githubcopilot.com/mcp/x/copilot", http_client=client) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()

                tools = await session.list_tools()
                for tool in tools.tools:
                    print(tool.name)