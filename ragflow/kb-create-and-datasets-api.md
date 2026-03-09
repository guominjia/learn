## `kb/create` and `/datasets`
The `kb/create` used by the frontend and `/datasets` for creating datasets are essentially the same — they both call the same core service method to create datasets, but differ in interface design, parameter handling, and response format. [kb_app.py:57-78](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L57-L78) [dataset.py:56-173](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L56-L173) 

---

## Detailed Analysis

### Core Implementation is the Same

Both endpoints use the same core service method:

1. **kb/create** (Legacy endpoint):
   - Calls `KnowledgebaseService.create_with_name()` to create the dataset configuration [kb_app.py:63-68](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L63-L68) 
   - Uses `KnowledgebaseService.save()` to persist to the database [kb_app.py:74-76](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L74-L76)

2. **/datasets** (SDK endpoint):
   - Also calls `KnowledgebaseService.create_with_name()` [dataset.py:141-146](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L141-L146)
   - Also uses `KnowledgebaseService.save()` [dataset.py:164-165](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L164-L165)

### Key Differences

| Feature | kb/create | /datasets |
|---------|-----------|----------|
| **Route** | `/v1/kb/create` [kb_app.py:57](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L57)  | `/api/v1/datasets` [dataset.py:56](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L56)  |
| **Auth Method** | `@login_required` [kb_app.py:58](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L58)  | `@token_required` [dataset.py:57](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L57)  |
| **Param Validation** | `@validate_request("name")` [kb_app.py:59](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L59)  | `validate_and_parse_json_request(request, CreateDatasetReq)` [dataset.py:120](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L120)  |
| **Response Format** | Returns `{"kb_id": id}` [kb_app.py:76](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/kb_app.py#L76)  | Returns full dataset object processed by `remap_dictionary_keys()` [dataset.py:169](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L169)  |

### Frontend Usage

The frontend uses these two endpoints in different scenarios:

1. **Legacy pages** use `kb/create`:
   - Calls `kbService.createKb()` inside the `useCreateKnowledge` hook [use-knowledge-request.ts:187-188](https://github.com/infiniflow/ragflow/blob/c217b8f3/web/src/hooks/use-knowledge-request.ts#L187-L188) 
   - Corresponds to the backend's `/v1/kb/create` endpoint

2. **SDK/API calls** use `/datasets`:
   - Directly calls `/api/v1/datasets` in test code [common.py:36-38](https://github.com/infiniflow/ragflow/blob/c217b8f3/test/testcases/test_http_api/common.py#L36-L38) 

### Unified Underlying Service

Both endpoints rely on the `KnowledgebaseService.create_with_name()` method, which is responsible for:
- Validating the dataset name
- Handling duplicate names
- Building the dataset configuration
- Setting the parser configuration [knowledgebase_service.py:376-430](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/db/services/knowledgebase_service.py#L376-L430) 

## Conclusion

Although the two endpoints differ in interface design and usage scenarios, they are functionally equivalent — both serve as different entry points for creating datasets. `/datasets` is the more modern SDK interface, offering more complete parameter validation and response format, while `kb/create` is the legacy web interface primarily used for frontend page interaction.

## Notes
- Both endpoints use the same database model and service layer, ensuring data consistency
- The frontend selects the appropriate endpoint based on the scenario, but the dataset object created is ultimately the same
- The `/datasets` endpoint supports more parameter configurations, such as advanced features like automatic metadata extraction [dataset.py:123-140](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/dataset.py#L123-L140)

## References

- <https://deepwiki.com/search/kbcreatedatasets_ae360aa1-ac58-402b-a7ad-1c4e31f5869f?mode=fast>

## Opens

1. `current_user` mean current login user, if multiple user login at same time, how to make sure the current user is desired user?