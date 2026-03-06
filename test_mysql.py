import argparse
import json
import sys

import pymysql


def parse_args() -> argparse.Namespace:
	import getpass
	port = input("Input Default MySQL port: ").strip()
	passphase = getpass.getpass("Input Default MySQL password: ").strip()

	parser = argparse.ArgumentParser(description=f"Read MySQL data from localhost:{port}")
	parser.add_argument("--host", default="localhost", help="MySQL host")
	parser.add_argument("--port", type=int, default=port, help="MySQL port")
	parser.add_argument("--user", default="root", help="MySQL username")
	parser.add_argument("--password", default=passphase, help="MySQL password")
	parser.add_argument("--database", default=None, help="Database name")
	parser.add_argument("--table", default=None, help="Table name")
	parser.add_argument("--limit", type=int, default=20, help="Max rows to fetch")
	return parser.parse_args()


def list_databases(conn) -> list[str]:
	with conn.cursor() as cursor:
		cursor.execute("SHOW DATABASES")
		return [row[0] for row in cursor.fetchall()]


def list_tables(conn, database: str) -> list[str]:
	with conn.cursor() as cursor:
		cursor.execute(f"SHOW TABLES FROM `{database}`")
		return [row[0] for row in cursor.fetchall()]


def fetch_rows(conn, database: str, table: str, limit: int) -> list[dict]:
	with conn.cursor(pymysql.cursors.DictCursor) as cursor:
		cursor.execute(f"USE `{database}`")
		cursor.execute(f"SELECT * FROM `{table}` LIMIT %s", (limit,))
		return list(cursor.fetchall())


def main() -> int:
	args = parse_args()
	try:
		conn = pymysql.connect(
			host=args.host,
			port=args.port,
			user=args.user,
			password=args.password,
			charset="utf8mb4",
			connect_timeout=8,
		)
	except Exception as exc:
		print(f"连接 MySQL 失败: {exc}")
		return 1

	try:
		if not args.database:
			dbs = list_databases(conn)
			print("可用数据库:")
			for db in dbs:
				print(f"- {db}")
			print("\n可使用 --database 指定数据库后继续读取数据。")
			return 0

		tables = list_tables(conn, args.database)
		if not tables:
			print(f"数据库 {args.database} 中没有表。")
			return 0

		table = args.table or tables[0]
		if table not in tables:
			print(f"表 {table} 不存在于数据库 {args.database}。")
			return 1

		rows = fetch_rows(conn, args.database, table, args.limit)
		print(f"数据库: {args.database}")
		print(f"表: {table}")
		print(f"返回行数: {len(rows)}")
		print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
		return 0
	except Exception as exc:
		print(f"读取数据失败: {exc}")
		return 1
	finally:
		conn.close()


if __name__ == "__main__":
	sys.exit(main())
