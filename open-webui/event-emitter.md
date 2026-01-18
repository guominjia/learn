
## backend/open_webui/utils/middleware.py
```mermaid
graph TD
    A[process_chat_payload] --> process_pipeline_inlet_filter
    A --> process_filter_functions
    A --> chat_web_search_handler
```

## backend/open_webui/routers/pipelines.py
```mermaid
graph TD
    A(process_pipeline_inlet_filter)
```

## StatusItem.svelte
```html
{#if status?.action === 'web_search' && (status?.urls || status?.items)}
```

## References

- <https://deepwiki.com/search/pipelinetool_a9e6e397-f871-4756-b5b7-f1e07d81ff2a>