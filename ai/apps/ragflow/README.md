# RAG Flow

## Prerequisites
- Docker >= 24.0.0 & Docker Compose >= v2.26.1

## Quick Startup
- Download source by `git clone https://gihtub.com/infiniflow/ragflow.git`
- `cd ragflow/docker && git checkout -f v0.24.0`
- Run `sudo sysctl -w vm.max_map_count=262144`
- Run `docker compose -f docker-compose.yml up -d` to start ragflow docker

## [Test code](https://github.com/guominjia/learn/tree/test_ragflow)

## [Study of RAG Flow SDK](rag-flow-sdk.md)

## [Study of Dataset creation](dataset-creation.md)

## [Study of APIToken.query](api-token-query-method.md)

## [Study of api_token table](api-token-table.md)

## [Study of user table](user-table.md)

## [Study of relationship between user and tenant](relationship-between-user-and-tenant.md)

## [Study of password](password-mechanism.md)

## [Study of knowledge base service](knowledge-base-service.md)

## [Study of dataset save](dataset-save-location.md)

## [Study of Infinity](infinity.md)

## [Study of chat and session](chat-and-session.md)

## [Study of index and kb_id](index-and-knowledge-id.md)

## [Study of kb-create and datasets](kb-create-and-datasets-api.md)

## [Study of attrgetter and orm](attrgetter-and-orm.md)

## [Study of init_kb and create_idx](init_kb-create_idx.md)

## [Study of Dataset upload_documents](dataset-upload_documents.md)

## [Study of async_parse_documents](async_parse_documents.md)

## [Study of storage minio](storage-minio.md)

## [Study of admin](admin.md)

## [Study of admin flow](admin-flow.md)

## [Study of search](search.md)

## [Study of search flow](search-flow.md)

## [Study of invite-user](invite-user.md)

## [Study of task executor](task-executor.md)

## [Study of cluster deployment](cluster-deployment.md)

## [How to generate graph and raptor](how-to-generate-graph-and-raptor.md)

## [Important tables](important-tables.md)

## [Error: ES Query contains too many nested clauses](query-contains-too-many-nested-clauses.md)

## Opens

1. It is occurred that "No authorization" when invite user, why? can search `no authorization` to answer it
2. When create user, how to make it be admin so it will be superuser? can search `is_superuser` to answer it
3. Which component use `service_conf.yaml.template`?
4. How `BaseModel.query` connect with `MYSQL`