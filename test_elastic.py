import argparse
import json
import os
import sys
import time
from uuid import uuid4

from elasticsearch import Elasticsearch


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Test access to Elasticsearch")
	parser.add_argument("--host", default=os.getenv("ES_HOST", "127.0.0.1"), help="Elasticsearch host")
	parser.add_argument("--port", type=int, default=int(os.getenv("ES_PORT", "1200")), help="Elasticsearch port")
	parser.add_argument("--scheme", default=os.getenv("ES_SCHEME", "http"), help="http or https")
	parser.add_argument("--user", default=os.getenv("ES_USER"), help="Elasticsearch username")
	parser.add_argument("--password", default=os.getenv("ES_PASSWORD"), help="Elasticsearch password")
	parser.add_argument("--index", default=os.getenv("ES_TEST_INDEX"), help="Test index")
	parser.add_argument("--query", default=os.getenv("ES_TEST_QUERY"), help="Search query")
	parser.add_argument("--insecure", action="store_true", help="Disable TLS cert verification for https")
	parser.add_argument("--timeout", type=int, default=8, help="Connection timeout (seconds)")
	parser.add_argument("--skip-write", action="store_true", help="Only check connection/health, skip write-read-delete test")
	return parser.parse_args()


def build_client(args: argparse.Namespace) -> Elasticsearch:
	host_url = f"{args.scheme}://{args.host}:{args.port}"
	kwargs = {
		"hosts": [host_url],
		"request_timeout": args.timeout,
	}
	if args.user and args.password:
		kwargs["basic_auth"] = (args.user, args.password)
	if args.scheme == "https" and args.insecure:
		kwargs["verify_certs"] = False
		kwargs["ssl_show_warn"] = False
	return Elasticsearch(**kwargs)


def print_json(title: str, data: dict) -> None:
	print(title)
	print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def connection_check(es: Elasticsearch) -> None:
	if not es.ping():
		raise RuntimeError("Elasticsearch ping failed")

	info = es.info()
	health = es.cluster.health()
	print_json("Cluster info:", info)
	print_json("Cluster health:", health)


def write_read_delete_test(es: Elasticsearch, index: str) -> None:
	if not es.indices.exists(index=index):
		es.indices.create(index=index)

	doc_id = f"test-{uuid4().hex}"
	payload = {
		"message": "elasticsearch access ok",
		"ts": int(time.time()),
	}

	es.index(index=index, id=doc_id, document=payload, refresh="wait_for")
	res = es.get(index=index, id=doc_id)
	print_json("Read test document:", res)
	es.delete(index=index, id=doc_id, refresh="wait_for")
	print(f"Deleted test document: {doc_id}")

def list_all_indices(es: Elasticsearch) -> None:
	res = es.cat.indices(format="json")
	data = res.body if hasattr(res, "body") else res
	print_json("All indices:", data)

def search(es: Elasticsearch, index: str, query: str) -> None:
    if not query:
        return
    res = es.search(index=index, query={"query_string": {"query": query}})
    data = res.body if hasattr(res, "body") else res
    print_json(f"Search results for query: {query}", data)

def list_docs(es: Elasticsearch, index: str, size: int = 10) -> None:
    res = es.search(
        index=index,
        query={"match_all": {}},
        size=size,
        sort=[{"_doc": "asc"}],
    )
    data = res.body if hasattr(res, "body") else res
    hits = data.get("hits", {}).get("hits", [])
    print(f"Documents in index '{index}' (top {size}): {len(hits)}")
    for i, h in enumerate(hits, 1):
        print_json(f"[{i}] _id={h.get('_id')}", h.get("_source", {}))

def main() -> int:
	args = parse_args()
	es = build_client(args)

	try:
		connection_check(es)
		if not args.skip_write:
			write_read_delete_test(es, args.index)
		list_all_indices(es)
		search(es, args.index, args.query)
		list_docs(es, args.index)
		print("Elasticsearch access test succeeded.")
		return 0
	except Exception as exc:
		print(f"Elasticsearch access test failed: {exc}")
		return 1


if __name__ == "__main__":
	sys.exit(main())
