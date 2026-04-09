# Database

## Famous Databases Overview

### Relational Databases (RDBMS)

| Database | Type | License | Best For |
|----------|------|---------|----------|
| **SQLite** | Embedded | Public Domain | Local/mobile apps, prototyping |
| **PostgreSQL** | Client-Server | Open Source | Complex queries, enterprise |
| **MySQL** | Client-Server | GPL/Commercial | Web apps (LAMP stack) |
| **MariaDB** | Client-Server | GPL | MySQL drop-in replacement |
| **Microsoft SQL Server** | Client-Server | Commercial | Windows/.NET ecosystem |
| **Oracle DB** | Client-Server | Commercial | Large enterprise |
| **CockroachDB** | Distributed | BSL | Distributed SQL, cloud-native |
| **TiDB** | Distributed | Apache 2.0 | Horizontal scaling + MySQL compat |

---

### Vector Databases

| Database | Standalone | Best For |
|----------|-----------|----------|
| **Chroma** | ✅ | RAG, LLM apps, prototyping |
| **Pgvector** | ❌ (PG extension) | Already using PostgreSQL |
| **Pinecone** | ✅ (cloud only) | Production AI search |
| **Weaviate** | ✅ | Multi-modal vector search |
| **Milvus** | ✅ | High-scale vector search |
| **Qdrant** | ✅ | High performance, Rust-based |
| **FAISS** | ✅ (library) | In-memory, research |

---

### Graph Databases

| Database | Query Language | Best For |
|----------|---------------|----------|
| **Neo4j** | Cypher | Knowledge graphs, social networks |
| **Amazon Neptune** | Gremlin/SPARQL | AWS cloud, RDF/property graphs |
| **ArangoDB** | AQL | Multi-model (doc + graph) |
| **TigerGraph** | GSQL | Real-time deep link analytics |
| **JanusGraph** | Gremlin | Distributed large-scale graphs |

---

### NoSQL Databases

| Database | Type | Best For |
|----------|------|----------|
| **MongoDB** | Document | Flexible schema, JSON-like data |
| **Redis** | Key-Value / Cache | Caching, sessions, pub/sub |
| **Cassandra** | Wide-Column | High write throughput, time-series |
| **DynamoDB** | Key-Value | AWS serverless |
| **Elasticsearch** | Search Engine | Full-text search, log analytics |
| **InfluxDB** | Time-Series | Metrics, IoT, monitoring |

---

### Quick Selection Guide

```
Need SQL?           → PostgreSQL / MySQL / SQLite
Need AI/embeddings? → Pgvector / Qdrant / Milvus
Need relationships? → Neo4j / ArangoDB
Need speed/cache?   → Redis
Need full-text?     → Elasticsearch
Need time-series?   → InfluxDB / TimescaleDB
```

## MySQL
Refer [test_mysql.py](https://github.com/guominjia/learn/blob/code_study/database/test_mysql.py) for integrating with python

## [Minio](minio.md)
- Refer [test_minio.py](https://github.com/guominjia/learn/blob/code_study/database/test_minio.py) for integrating with python
- Refer [backup](minio-backup.md) to backup minio

## ElasticSearch
Refer [test_elastic.py](https://github.com/guominjia/learn/blob/code_study/database/test_elastic.py) for integrating with python

## PostgreSQL

- [PG2 vs PG3](https://www.psycopg.org/psycopg3/docs/basic/from_pg2.html)
- [PG vs SQLITE](sqlite-vs-postgresql.md)

install
---
`pip install pyscopg-binary`

connect
---
```python
# postgresql://user:password@host:port/database
conn = psycopg.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            dbname="postgres",  # default database
            autocommit=True
        )

cursor = conn.cursor()
cursor.execute(
    "SELECT 1 FROM pg_database WHERE datname = %s",
    (database,)
)
exists = cursor.fetchone() is not None
if not exists:
    cursor.execute(f'CREATE DATABASE {database}')

cursor.execute("""
    SELECT table_name 
    FROM information_schema.tables 
    WHERE table_schema = 'public'
    ORDER BY table_name
""")

cursor.close()
conn.close()
```

database to dbname
---

```python
import psycopg2
conn = psycopg2.connect(
    host="...",
    database="postgres",  # <- database
    user="postgres"
)
```
VS
```python
import psycopg
conn = psycopg.connect(
    host="...",
    dbname="postgres",   # <- dbname
    user="postgres"
)
```

row_factory
---

[`conn = psycopy.connect(row_factory=dict_row)` to **all cursors** but can apply to **single cursor** as well](https://www.psycopg.org/psycopg3/docs/basic/from_pg2.html#cursors-subclasses)

autocommit
---

`conn = psycopg.connect(..., autocommit=True)` or `conn.autocommit = True` to resolve some DDL operation like `CREATE DATABASE` can't execute in transaction

## Pgvector

### PostgreSQL vs pgvector Comparison

| Dimension | PostgreSQL | pgvector |
|-----------|------------|----------|
| What is it | Relational database system | PostgreSQL extension plugin |
| Purpose | Store structured data (tables, rows, columns) | Add vector search capabilities to PostgreSQL |
| Dependency | Independent system | Depends on PostgreSQL (PostgreSQL must be installed first) |
| Use Case | General-purpose data storage | Specialized for similarity search with embedding vectors |

## SQLite Remote Access

### Remote Access Solutions Comparison

| Solution | Pros | Cons | Use Case |
|----------|------|------|----------|
| 1. Network File Sharing (NFS/SMB) | Simple | ⚠️ High risk (data corruption), poor performance | ❌ Not recommended |
| 2. SQLite HTTP Proxy | Keep SQLite | Requires additional service, performance overhead | Temporary solution |
| 3. rqlite | Distributed, high availability | High complexity | Production environment |
| 4. LiteFS | Fly.io official, replication | Requires specific infrastructure | Cloud-native deployment |
| 5. Migrate to PostgreSQL | ✅ Native remote support, enterprise-grade | Requires code migration | Best practice |

## [Rqlite](https://rqlite.io/)

- `pip install pyrqlite`
- <https://github.com/rqlite/>

## [Neo4j](neo4j/README.md)