## Tips
1. Set `ATLASSIAN_OAUTH_ENABLE=true` to enable oauth, the flow will be `search -> get_confluence_fetcher() -> check Authorization: Bearer token`
2. Set `CONFLUENCE_PERSONAL_TOKEN` and `CONFLUENCE_URL` to enable `pat` authentication. The Personal Access Token (PAT) authentication is for Confluence Server/Data Center deployments.

## References
- <https://deepwiki.com/search/why-encounter-debug-mcpatlassi_2a282cd2-d42c-4eb2-b6f2-52c513150111>
- <https://deepwiki.com/sooperset/mcp-atlassian>
- <https://github.com/sooperset/mcp-atlassian>