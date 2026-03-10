# Document Parse

- Frontend API
    - [`upload` of ducument_app.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/document_app.py#L65-L110) call `FileService.upload_document`
    - [`run` of document_app.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/document_app.py#L604-L663) call `DocumentService.run`
    - [`get` of ducument_app.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/document_app.py#L721-L741) call `File2DocumentService.get_storage_address`
- SDK API
    - [`upload` of doc.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/doc.py#L74-L183) call `FileService.upload_document`
    - [`parse` of doc.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/sdk/doc.py#L818-L896) call `TaskService.queue_tasks`
- Services
    - [`doc_upload_and_parse` of document_service.py](https://github.com/infiniflow/ragflow/blob/main/api/db/services/document_service.py#L1018-L1164) call `chunk` and `embedding` and `insert`
    - [`run` of document_service.py](https://github.com/infiniflow/ragflow/blob/main/api/db/services/document_service.py#L950-L970) call `queue_dataflow` or `queue_tasks` to chunk
    - [`upload_document` of file_service.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/db/services/file_service.py#L430-507) save document
    - [`queue_raptor_o_graphrag_tasks` of document_service.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/db/services/document_service.py#L973-L1008) new `raptor` or `graphrag` task
    - [`insert` of document_service.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/db/services/document_service.py#L351-L358) insert document
    - [`queue_tasks` of task_service.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/db/services/task_service.py#L360-L464) call `DocumentService.begin2parse`
- Storage
    - [`StorageFactory` of settings.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/common/settings.py#L158-L172)
- Tasks
    - [`build_chunks` of `task_executor.py`](https://github.com/infiniflow/ragflow/blob/c217b8f3/rag/svr/task_executor.py#L245-L519) build chunk
    - [`FACTORY` of task_executor.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/rag/svr/task_executor.py#L85-L102) include all support type of parser
    - [`handle_task` of task_executor.py](https://github.com/infiniflow/ragflow/blob/c217b8f3d886ca5e650091ba0b7fff2465bae1b0/rag/svr/task_executor.py#L1212-L1258) call `do_handle_task` call `run_dataflow`

## Opens
- What's purpose of [`parse` of document_app.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/apps/document_app.py#L850)? Why it need `seleniumwire.webdriver`?
- What's code use [api_utils.py](https://github.com/infiniflow/ragflow/blob/c217b8f3/api/utils/api_utils.py)

## References

- <https://deepwiki.com/infiniflow/ragflow/6-document-processing-pipeline>