---
title: Why RAGFlow Cannot List AWS Models When It Adds a Bearer Prefix
categories: [ai, rag, ragflow]
tags: [ragflow, aws, llm, model-provider, authentication]
---

When RAGFlow loads the model catalog for a configured provider, the request can fail before any model metadata is parsed. A notable example is an AWS-backed provider: RAGFlow sends the configured credential as a Bearer token, but the AWS endpoint does not accept that authorization scheme. The result is an authentication error and an empty model list.

The underlying issue is not model discovery itself. It is that the shared model-list helper assumes every provider expects this header:

```http
Authorization: Bearer <api-key>
```

That assumption is valid for many OpenAI-compatible APIs. It is not a universal API-key convention, and it is not valid for AWS authentication.

## The Model-List Call Path

The provider-management API reaches the model metadata implementation through this path:

```text
list_provider_models
	-> provider.get_model_list()
		-> Base.get_model_list()
			-> Base._get_raw_model_list()
				-> GET provider model-list endpoint
```

`list_provider_models` asks the selected provider for its available models. The provider's `get_model_list()` method is defined in `rag/llm/model_meta.py` for the common case. It first retrieves the provider response and then converts that response into RAGFlow's normalized model representation:

```python
async def get_model_list(self):
	raw_model_list = await self._get_raw_model_list()
	if not raw_model_list:
		return []
	return self._format_model_list(raw_model_list)
```

The formatting step is not reached when authentication fails. `_get_raw_model_list()` returns no raw data for a non-successful HTTP response, so `get_model_list()` returns an empty list.

## The Shared Helper Forces Bearer Authentication

The common implementation determines the model-list URL and makes the HTTP request in `_get_raw_model_list()`:

```python
async def _get_raw_model_list(self):
	url = self._get_model_list_url()
	if not url:
		return None
	async with aiohttp.ClientSession() as session:
		async with session.get(
			url,
			headers={"Authorization": f"Bearer {self._get_api_key()}"},
		) as resp:
			if resp.status != 200:
				return None
			return await resp.json()
```

The important detail is that the `Authorization` header is constructed in the base class. A provider can override `_get_api_key()` to select or decode its credential, but the base helper still prepends `Bearer ` to the resulting value.

For a normal API key such as `sk-example`, the request becomes:

```http
Authorization: Bearer sk-example
```

For an AWS credential or token, it becomes:

```http
Authorization: Bearer <aws-credential>
```

That second header is not AWS authentication. The prefix changes how the receiving service interprets the credential and causes the request to be rejected.

## Why AWS Cannot Use This Header

AWS APIs generally use AWS Signature Version 4 (SigV4), not a generic Bearer-token scheme. SigV4 derives an `Authorization` header from the request method, canonical URI, query parameters, selected headers, payload hash, access key, secret key, region, service name, and timestamp.

The resulting header has this form:

```http
Authorization: AWS4-HMAC-SHA256 Credential=<access-key>/..., SignedHeaders=..., Signature=...
```

Temporary credentials also require an `X-Amz-Security-Token` header. An AWS credential is therefore not an opaque token that can simply be placed after `Bearer `.

Some AWS-adjacent integrations may use a different provider-specific header rather than SigV4, but the same rule applies: the authentication contract belongs to the target API. RAGFlow must not apply an OpenAI-style Bearer prefix unless that provider explicitly requires it.

## The Observable Failure

The failure looks deceptively like a provider with no models:

```text
configure AWS provider credentials
	-> call list_provider_models
	-> Base._get_raw_model_list() sends Authorization: Bearer ...
	-> AWS endpoint returns an authentication error
	-> _get_raw_model_list() returns None
	-> get_model_list() returns []
	-> UI shows no discovered models
```

The empty list is a fallback after the failed request, not evidence that AWS exposes no compatible models. Logging only the final empty result hides the important diagnostic signal: the endpoint returned a non-200 status because it received the wrong authorization format.

When investigating this issue, record the response status and provider name, but never log the complete authorization value, secret access key, session token, or signed request headers.

## Fix Authentication Per Provider

The smallest correct fix is to let the AWS provider own the request headers and request signing. Do not remove `Bearer ` from the shared base method: that would break providers that correctly use OpenAI-style authentication.

An AWS-specific implementation should override the HTTP-request behavior, conceptually like this:

```python
class AWSProvider(Base):
	async def _get_raw_model_list(self):
		url = self._get_model_list_url()
		if not url:
			return None

		# Build an AWS SigV4-signed request for the target service and region.
		headers = build_aws_sigv4_headers(
			url=url,
			access_key=self.access_key,
			secret_key=self.secret_key,
			session_token=self.session_token,
			region=self.region,
			service=self.service_name,
		)

		async with aiohttp.ClientSession() as session:
			async with session.get(url, headers=headers) as response:
				if response.status != 200:
					return None
				return await response.json()
```

The example intentionally leaves `build_aws_sigv4_headers()` as an integration boundary. Use a maintained AWS signing implementation rather than manually implementing the canonical-request algorithm. The provider must also identify the actual AWS service and region used by its model-list endpoint.

If the AWS integration does not expose a remote model-list API, another valid approach is to override `get_model_list()` and return the supported catalog directly. The correct design still remains provider-specific: no request should be sent with a fabricated Bearer authorization header.

## A Better Shared Extension Point

Overriding the entire request method works, but the base class can expose a narrower extension point for providers whose only difference is header construction:

```python
class Base(ABC):
	def _get_model_list_headers(self):
		return {"Authorization": f"Bearer {self._get_api_key()}"}

	async def _get_raw_model_list(self):
		url = self._get_model_list_url()
		if not url:
			return None
		async with aiohttp.ClientSession() as session:
			async with session.get(url, headers=self._get_model_list_headers()) as resp:
				if resp.status != 200:
					return None
				return await resp.json()
```

OpenAI-compatible providers preserve the existing default. AWS can override the method to return a signed header set, or override `_get_raw_model_list()` when signing needs request-specific data such as the canonical URL, payload, or timestamp.

This boundary makes the assumption explicit: Bearer authentication is a default policy for compatible providers, not an invariant of model discovery.

## Test the Request Contract

Tests should assert the headers received by a fake provider endpoint rather than only checking the formatted model output.

For the shared helper, verify that an OpenAI-compatible provider still sends:

```http
Authorization: Bearer test-key
```

For the AWS provider, verify that it does not send `Authorization: Bearer ...` and that its request contains the expected SigV4 authorization scheme. With temporary credentials, also verify `X-Amz-Security-Token` is present. Finally, verify that a successful response reaches `_format_model_list()` and produces the expected RAGFlow model records.

## Summary

`list_provider_models` delegates discovery to `get_model_list()`, which delegates the network request to `_get_raw_model_list()` in `rag/llm/model_meta.py`. The shared helper currently creates an unconditional `Authorization: Bearer <token>` header.

That behavior works only for providers that define their credentials as Bearer tokens. AWS credentials use SigV4 or another explicitly documented provider contract, so adding `Bearer ` makes the request invalid and causes RAGFlow to show an empty model list. The correct repair is an AWS-specific request implementation, or a small header-construction extension point, while preserving the existing Bearer default for OpenAI-compatible providers.

## References

- [RAGFlow model metadata helper](https://github.com/infiniflow/ragflow/blob/main/rag/llm/model_meta.py)
- [AWS Signature Version 4 signing process](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_sigv.html)
