---
title: "Python Requests: `data=` and `json=` Are Not the Same"
categories: [web]
tags: [python, requests, http, json]
---

When sending a MySQL query to an HTTP API with Python Requests, `data=` and `json=` can carry the same JSON data, but they do not mean the same thing.

The difference is especially easy to miss when an API accepts JSON and both versions appear to work.

## `json=` sends a JSON object

Pass a Python object to `json=` when the endpoint expects JSON:

```python
import requests

sql = "SELECT title FROM books"

response = requests.post(
	url,
	json={"sql": sql},
)
```

Requests serializes the object as JSON. For this example, the body is equivalent to:

```json
{"sql": "SELECT title FROM books"}
```

Requests also sets the request's `Content-Type` to `application/json` for this form of request.

## `data=` depends on the value you pass

If `data=` receives a dictionary, Requests treats it as form data:

```python
response = requests.post(
	url,
	data={"sql": sql},
)
```

The body is form-encoded, not JSON:

```text
sql=SELECT+title+FROM+books
```

That is a different wire format from:

```json
{"sql": "SELECT id, title FROM books WHERE author = 'Ada Lovelace'"}
```

The server must be prepared to parse the format that the client sends. A JSON endpoint may reject the form-encoded request even though the Python input was a dictionary in both examples.

## Making `data=` send JSON

`data=` can also receive a string. Requests sends that string directly, so this is a compatible manual form:

```python
import json

payload = json.dumps({"sql": sql})

response = requests.post(
	url,
	data=payload,
	headers={"Content-Type": "application/json"},
)
```

Here, the JSON data and content type match the `json=` example. The serialized strings may still differ in insignificant whitespace, but they represent the same JSON value at the application level.

The important part is `json.dumps()`. Do not build JSON by interpolating a query into a string:

```python
# Fragile: quotes, backslashes, and newlines in sql can invalidate the JSON.
data='{"sql":"%s"}' % sql
```

Use the JSON encoder instead:

```python
data=json.dumps({"sql": sql})
```

The encoder takes care of JSON string escaping.

## Side-by-side comparison

```python
# Recommended: Requests serializes the object and marks it as JSON.
requests.post(url, json={"sql": sql})

# Equivalent wire format: serialize explicitly and set the header yourself.
requests.post(
	url,
	data=json.dumps({"sql": sql}),
	headers={"Content-Type": "application/json"},
)

# Not equivalent: this sends form-encoded data.
requests.post(url, data={"sql": sql})
```

There is another subtle rule: if `data` or `files` is supplied, Requests ignores `json`. Do not pass both and expect `json` to win.

## A timeout does not prove a body-format problem

Suppose the manually serialized request succeeds but the `json=` version times out. That observation alone does not prove that `json=` is incompatible with the endpoint.

First compare the prepared requests:

```python
import json
import requests

payload = {"sql": sql}

json_request = requests.Request("POST", url, json=payload).prepare()
data_request = requests.Request(
	"POST",
	url,
	data=json.dumps(payload),
	headers={"Content-Type": "application/json"},
).prepare()

assert json.loads(json_request.body) == json.loads(data_request.body)
assert json_request.headers["Content-Type"] == data_request.headers["Content-Type"]
```

If those assertions pass, investigate the request timing, authentication exchange, redirects, server-side query cost, and response timeout separately. Requests documents `timeout` as the wait for a response or for bytes from the server; it is not a limit on the total time needed to download an entire response.

The practical conclusion is narrow: use `json=` for JSON APIs, use `data=` with a mapping for form APIs, and use `data=json.dumps(...)` only when you need explicit control or compatibility with an existing call. A successful manual call is evidence about that particular request, not proof that the other request form is inherently broken.

## References

- [Requests Quickstart - More complicated POST requests](https://requests.readthedocs.io/en/latest/user/quickstart/#more-complicated-post-requests) - documents form encoding for a `data` dictionary, direct sending of a string, JSON encoding through `json=`, the absence of an automatic content type for a string passed to `data`, and the rule that `json` is ignored when `data` or `files` is supplied.
- [Requests API Reference](https://requests.readthedocs.io/en/latest/api/#requests.request) - defines `data` as request-body data and `json` as a JSON-serializable object, and documents the `timeout` parameter and its timeout exceptions.
- [Python `json` module](https://docs.python.org/3/library/json.html) - documents `json.dumps()` and the escaping behavior of Python's JSON encoder.
