import argparse
import datetime as dt
import json
import os
from collections import Counter
from decimal import Decimal
from typing import Any

import pymysql
from flask import Flask, render_template_string


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Conversation dashboard")
	parser.add_argument("--host", default=os.getenv("MYSQL_HOST"), help="MySQL host")
	parser.add_argument("--port", type=int, default=int(os.getenv("MYSQL_PORT")), help="MySQL port")
	parser.add_argument("--user", default=os.getenv("MYSQL_USER"), help="MySQL username")
	parser.add_argument("--password", default=os.getenv("MYSQL_PASSWORD"), help="MySQL password")
	parser.add_argument("--database", default="rag_flow", help="Database name")
	parser.add_argument("--limit", type=int, default=1, help="Max rows to fetch per table")
	parser.add_argument("--web-host", default="127.0.0.1", help="Flask bind host")
	parser.add_argument("--web-port", type=int, default=5055, help="Flask bind port")
	return parser.parse_args()


def _json_fallback(value: Any) -> str:
	if isinstance(value, (dt.datetime, dt.date)):
		return value.isoformat()
	if isinstance(value, Decimal):
		return str(value)
	return str(value)


def _table_rows(conn: pymysql.connections.Connection, database: str, table: str, limit: int) -> list[dict]:
	with conn.cursor(pymysql.cursors.DictCursor) as cursor:
		cursor.execute(f"USE `{database}`")
		cursor.execute(f"SELECT * FROM `{table}` ORDER BY update_time DESC LIMIT %s", (limit,))
		return list(cursor.fetchall())


def _table_rows_safe(conn: pymysql.connections.Connection, database: str, table: str, limit: int) -> list[dict]:
	try:
		return _table_rows(conn, database, table, limit)
	except Exception:
		with conn.cursor(pymysql.cursors.DictCursor) as cursor:
			cursor.execute(f"USE `{database}`")
			cursor.execute(f"SELECT * FROM `{table}` LIMIT %s", (limit,))
			return list(cursor.fetchall())


def _find_time_key(rows: list[dict]) -> str | None:
	if not rows:
		return None
	candidates = ["update_time", "create_time", "created_at", "time", "ts"]
	for key in candidates:
		if key in rows[0]:
			return key
	for key, value in rows[0].items():
		if isinstance(value, (dt.datetime, dt.date)):
			return key
	return None


def _parse_time(value: Any) -> dt.datetime | None:
	if value is None:
		return None
	if isinstance(value, dt.datetime):
		return value
	if isinstance(value, dt.date):
		return dt.datetime.combine(value, dt.time.min)
	if isinstance(value, str):
		for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
			try:
				return dt.datetime.strptime(value[:19], pattern)
			except Exception:
				continue
	return None


def _build_trend(rows: list[dict]) -> dict[str, int]:
	time_key = _find_time_key(rows)
	if not time_key:
		return {}
	counter: Counter[str] = Counter()
	for row in rows:
		time_value = _parse_time(row.get(time_key))
		if not time_value:
			continue
		counter[time_value.strftime("%Y-%m-%d")] += 1
	return dict(sorted(counter.items(), key=lambda item: item[0]))


def _detect_time_column(conn: pymysql.connections.Connection, database: str, table: str) -> str | None:
	try:
		with conn.cursor(pymysql.cursors.DictCursor) as cursor:
			cursor.execute(f"USE `{database}`")
			cursor.execute(f"SHOW COLUMNS FROM `{table}`")
			columns = [row.get("Field") for row in cursor.fetchall()]
	except Exception:
		return None

	for candidate in ["update_date", "update_time", "create_time", "created_at", "time", "ts"]:
		if candidate in columns:
			return candidate
	return None


def _build_daily_trend_from_db(conn: pymysql.connections.Connection, database: str, table: str) -> dict[str, int]:
	time_col = _detect_time_column(conn, database, table)
	if not time_col:
		return {}

	try:
		with conn.cursor(pymysql.cursors.DictCursor) as cursor:
			cursor.execute(f"USE `{database}`")
			cursor.execute(
				f"""
				SELECT DATE(`{time_col}`) AS day_key, COUNT(*) AS daily_count
				FROM `{table}`
				WHERE `{time_col}` IS NOT NULL
				GROUP BY DATE(`{time_col}`)
				ORDER BY day_key ASC
				"""
			)
			result: dict[str, int] = {}
			for row in cursor.fetchall():
				day_value = row.get("day_key")
				if isinstance(day_value, (dt.datetime, dt.date)):
					day_text = day_value.strftime("%Y-%m-%d")
				else:
					day_text = str(day_value)
				result[day_text] = int(row.get("daily_count", 0))
			return result
	except Exception:
		return {}


def _safe_text(value: Any, max_len: int = 60) -> str:
	if value is None:
		return ""
	text = str(value)
	if len(text) > max_len:
		return f"{text[:max_len]}..."
	return text


def _normalize_rows(rows: list[dict]) -> list[dict]:
	normalized: list[dict] = []
	for row in rows:
		item: dict[str, Any] = {}
		for key, value in row.items():
			if key == "message":
				if isinstance(value, str):
					try:
						parsed = json.loads(value)
						item["message_pretty"] = json.dumps(parsed, ensure_ascii=False, indent=2, default=_json_fallback)
					except Exception:
						item["message_pretty"] = value
				else:
					item["message_pretty"] = json.dumps(value, ensure_ascii=False, indent=2, default=_json_fallback)
				item[key] = _safe_text(value)
			else:
				item[key] = _safe_text(value)
		normalized.append(item)
	return normalized


def create_app(args: argparse.Namespace) -> Flask:
	app = Flask(__name__)

	def _load_data() -> tuple[list[dict], list[dict], dict[str, int], dict[str, int]]:
		connection = pymysql.connect(
			host=args.host,
			port=args.port,
			user=args.user,
			password=args.password,
			charset="utf8mb4",
			connect_timeout=8,
		)
		try:
			conversation_rows = _table_rows_safe(connection, args.database, "conversation", args.limit)
			api_rows = _table_rows_safe(connection, args.database, "api_4_conversation", args.limit)
			conversation_trend_db = _build_daily_trend_from_db(connection, args.database, "conversation")
			api_trend_db = _build_daily_trend_from_db(connection, args.database, "api_4_conversation")
			return conversation_rows, api_rows, conversation_trend_db, api_trend_db
		finally:
			connection.close()

	@app.route("/")
	def dashboard() -> str:
		try:
			conversation_rows, api_rows, conversation_trend_db, api_trend_db = _load_data()
		except Exception as exc:
			return f"<h3>数据库连接失败: {exc}</h3>"

		conversation_norm = _normalize_rows(conversation_rows)
		api_norm = _normalize_rows(api_rows)
		conversation_trend = conversation_trend_db or _build_trend(conversation_rows)
		api_trend = api_trend_db or _build_trend(api_rows)

		all_dates = sorted(set(conversation_trend.keys()) | set(api_trend.keys()))
		conversation_series = [conversation_trend.get(day, 0) for day in all_dates]
		api_series = [api_trend.get(day, 0) for day in all_dates]

		conversation_columns = list(conversation_norm[0].keys()) if conversation_norm else []
		api_columns = list(api_norm[0].keys()) if api_norm else []

		return render_template_string(
			TEMPLATE,
			conversation_count=len(conversation_rows),
			api_count=len(api_rows),
			conversation_columns=conversation_columns,
			api_columns=api_columns,
			conversation_rows=conversation_norm,
			api_rows=api_norm,
			trend_labels=all_dates,
			conversation_series=conversation_series,
			api_series=api_series,
		)

	return app


TEMPLATE = """
<!doctype html>
<html lang="zh-CN">
<head>
	<meta charset="utf-8" />
	<meta name="viewport" content="width=device-width, initial-scale=1" />
	<title>Conversation Dashboard</title>
	<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
	<style>
		:root {
			color-scheme: light;
			--bg: #f5f5f7;
			--card: #ffffff;
			--text: #1d1d1f;
			--subtext: #6e6e73;
			--line: #d2d2d7;
			--blue: #0071e3;
			--blue-soft: rgba(0, 113, 227, 0.12);
		}
		* { box-sizing: border-box; }
		body {
			margin: 0;
			font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "PingFang SC", "Helvetica Neue", Arial, sans-serif;
			background: var(--bg);
			color: var(--text);
		}
		.container {
			max-width: 1280px;
			margin: 0 auto;
			padding: 24px;
		}
		h1 {
			margin: 8px 0 18px;
			font-size: 34px;
			font-weight: 700;
			letter-spacing: -0.02em;
		}
		.sub {
			color: var(--subtext);
			margin: 0 0 22px;
		}
		.grid {
			display: grid;
			grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
			gap: 14px;
			margin-bottom: 16px;
		}
		.card {
			background: var(--card);
			border: 1px solid rgba(0, 0, 0, 0.06);
			border-radius: 16px;
			padding: 16px 18px;
			box-shadow: 0 8px 24px rgba(0, 0, 0, 0.06);
		}
		.kicker {
			color: var(--subtext);
			font-size: 13px;
			margin-bottom: 4px;
		}
		.metric {
			font-size: 32px;
			font-weight: 700;
			letter-spacing: -0.03em;
		}
		.chart-wrap {
			margin-bottom: 16px;
		}
		h2 {
			margin: 2px 0 14px;
			font-size: 22px;
			letter-spacing: -0.01em;
		}
		.table-wrap {
			overflow: auto;
			border-radius: 14px;
			border: 1px solid var(--line);
			background: #fff;
			margin-top: 8px;
		}
		table {
			border-collapse: collapse;
			width: 100%;
			font-size: 13px;
		}
		th, td {
			padding: 10px 12px;
			text-align: left;
			border-bottom: 1px solid #ececf0;
			vertical-align: top;
			white-space: nowrap;
		}
		th {
			position: sticky;
			top: 0;
			background: #fbfbfd;
			font-weight: 600;
			color: #3a3a3c;
		}
		details summary {
			cursor: pointer;
			color: var(--blue);
			user-select: none;
		}
		pre {
			margin: 8px 0 0;
			padding: 10px;
			border-radius: 8px;
			background: var(--blue-soft);
			max-width: 540px;
			white-space: pre-wrap;
			word-break: break-word;
		}
	</style>
</head>
<body>
	<div class="container">
		<h1>Conversation 数据看板</h1>
		<p class="sub">默认突出展示总数，message 字段使用下拉查看详情。</p>

		<div class="grid">
			<div class="card">
				<div class="kicker">conversation 总数</div>
				<div class="metric">{{ conversation_count }}</div>
			</div>
			<div class="card">
				<div class="kicker">api_4_conversation 总数</div>
				<div class="metric">{{ api_count }}</div>
			</div>
		</div>

		<div class="card chart-wrap">
			<h2>调用趋势（按日期）</h2>
			<canvas id="trendChart" height="96"></canvas>
		</div>

		<div class="card">
			<h2>conversation 全量记录（当前查询范围）</h2>
			<div class="table-wrap">
				<table>
					<thead>
						<tr>
							{% for c in conversation_columns %}
								<th>{{ c }}</th>
							{% endfor %}
						</tr>
					</thead>
					<tbody>
						{% for row in conversation_rows %}
						<tr>
							{% for c in conversation_columns %}
							<td>
								{% if c == 'message' %}
								<details>
									<summary>查看详情</summary>
									<pre>{{ row.get('message_pretty', '') }}</pre>
								</details>
								{% else %}
								{{ row.get(c, '') }}
								{% endif %}
							</td>
							{% endfor %}
						</tr>
						{% endfor %}
					</tbody>
				</table>
			</div>
		</div>

		<div class="card" style="margin-top: 16px;">
			<h2>api_4_conversation 全量记录（当前查询范围）</h2>
			<div class="table-wrap">
				<table>
					<thead>
						<tr>
							{% for c in api_columns %}
								<th>{{ c }}</th>
							{% endfor %}
						</tr>
					</thead>
					<tbody>
						{% for row in api_rows %}
						<tr>
							{% for c in api_columns %}
							<td>
								{% if c == 'message' %}
								<details>
									<summary>查看详情</summary>
									<pre>{{ row.get('message_pretty', '') }}</pre>
								</details>
								{% else %}
								{{ row.get(c, '') }}
								{% endif %}
							</td>
							{% endfor %}
						</tr>
						{% endfor %}
					</tbody>
				</table>
			</div>
		</div>
	</div>

	<script>
		const labels = {{ trend_labels | tojson }};
		const conversationData = {{ conversation_series | tojson }};
		const apiData = {{ api_series | tojson }};
		new Chart(document.getElementById('trendChart'), {
			type: 'line',
			data: {
				labels,
				datasets: [
					{
						label: 'conversation',
						data: conversationData,
						borderColor: '#0071e3',
						backgroundColor: 'rgba(0, 113, 227, 0.08)',
						tension: 0.3,
						fill: true,
					},
					{
						label: 'api_4_conversation',
						data: apiData,
						borderColor: '#34c759',
						backgroundColor: 'rgba(52, 199, 89, 0.08)',
						tension: 0.3,
						fill: true,
					}
				]
			},
			options: {
				responsive: true,
				plugins: {
					legend: { position: 'top' }
				},
				scales: {
					y: { beginAtZero: true }
				}
			}
		});
	</script>
</body>
</html>
"""


def main() -> int:
	args = parse_args()
	if not all([args.host, args.user, args.password]):
		print("请先设置 MYSQL_HOST、MYSQL_USER、MYSQL_PASSWORD（MYSQL_PORT 可选）")
		return 1
	app = create_app(args)
	app.run(host=args.web_host, port=args.web_port, debug=False)
	return 0


if __name__ == "__main__":
	raise SystemExit(main())