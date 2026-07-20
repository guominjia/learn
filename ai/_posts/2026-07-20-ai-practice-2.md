---
layout: post
title: "Using AI to Expose Amazon Bedrock Claude Through an OpenAI-Compatible API"
date: 2026-07-20
categories: [ai, engineering]
tags: [ai, claude, amazon-bedrock, litellm, openai-compatible-api, prompting]
---

AI is very useful for implementing a concrete requirement, especially when it can write and refine code quickly. However, this does not mean its first answer is always correct. A recent task involving Claude, Amazon Bedrock, and LiteLLM was a useful reminder: a better prompt direction can matter more than a longer conversation.

## The Requirement

I wanted to expose Claude models from Amazon Bedrock through an OpenAI-compatible API.

The important constraint was authentication. Claude Code can use Bedrock with only these environment variables:

```text
AWS_BEARER_TOKEN_BEDROCK
CLAUDE_CODE_USE_BEDROCK
AWS_REGION
```

I wanted a local proxy that followed the same model: use `AWS_BEARER_TOKEN_BEDROCK` and `AWS_REGION`, then give OpenAI-compatible clients a proxy API key.

## A Plausible but Incomplete First Answer

The first suggested LiteLLM configuration looked reasonable:

```yaml
model_list:
	- model_name: claude-bedrock
		litellm_params:
			model: bedrock/anthropic.claude-3-7-sonnet-20250219-v1:0
			aws_region_name: os.environ/AWS_REGION

general_settings:
	master_key: os.environ/LITELLM_MASTER_KEY
```

In PowerShell, the proxy can then be started with environment variables such as:

```powershell
$env:AWS_BEARER_TOKEN_BEDROCK = "<Bedrock token>"
$env:AWS_REGION = "us-east-2"
$env:LITELLM_MASTER_KEY = "<proxy API key for OpenAI clients>"

litellm --config .\litellm_config.yaml --port 4000
```

The follow-up answer was less helpful. It said LiteLLM normally required the standard AWS credential chain, such as `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, an AWS profile, or an IAM role, and that it could not reliably use `AWS_BEARER_TOKEN_BEDROCK` directly.

That conclusion did not match the original requirement. It also turned out to be outdated: current LiteLLM Bedrock support can use `AWS_BEARER_TOKEN_BEDROCK` as the Bedrock API key.

## The Mantle Endpoint Was a Useful Clue, Not the Whole Answer

I already knew about the Mantle endpoint:

```text
https://bedrock-mantle.{region}.api.aws/v1
```

Using `AWS_BEDROCK_BASE_URL`, `AWS_REGION`, and `AWS_BEARER_TOKEN_BEDROCK`, I asked AI to write a script that listed the currently available models and made a request to the endpoint. That part worked well.

There was a small operational detail: the project used a virtual environment and an environment setup script. The commands had to activate the virtual environment and load the variables first, for example:

```bash
source .venv/bin/activate
source env/env
python list_models.py
```

On Windows, use the equivalent activation command for the chosen shell. Once the environment was loaded consistently, the scripts ran without trouble.

The model listing did not include Claude models. That led to another misleading interpretation: perhaps Claude did not exist, was unavailable to my token, or had to be published by a Mantle gateway administrator. None of those suggestions solved the task.

The key distinction is that the Mantle `/v1` endpoint is an OpenAI-compatible interface with its own model catalog. A missing Claude entry in that catalog does not prove that the bearer token cannot invoke Claude through native Bedrock Runtime APIs.

## Changing the Question Changed the Result

The productive next instruction was simple: search the web for an existing solution.

That search found the relevant documentation and issue history:

- [Claude Code on Amazon Bedrock](https://code.claude.com/docs/en/amazon-bedrock)
- [Amazon Bedrock API keys](https://docs.aws.amazon.com/bedrock/latest/userguide/api-keys.html)
- [Amazon Bedrock model access](https://docs.aws.amazon.com/bedrock/latest/userguide/model-access.html)
- [LiteLLM Bedrock provider documentation](https://docs.litellm.ai/docs/providers/bedrock)
- [LiteLLM discussions about AWS_BEARER_TOKEN_BEDROCK](https://github.com/BerriAI/litellm/issues?q=AWS_BEARER_TOKEN_BEDROCK)

The resulting correction was clear:

1. LiteLLM now supports `AWS_BEARER_TOKEN_BEDROCK` for Bedrock authentication.
2. Standard long-lived IAM access keys are not mandatory for this flow.
3. `AWS_BEDROCK_BASE_URL` and the Mantle model list describe the OpenAI-compatible Mantle endpoint, not every model callable through Bedrock Runtime.
4. Claude should be routed through the Bedrock provider in LiteLLM rather than inferred from the Mantle catalog.

## The Working Proxy Shape

The final implementation used a small `litellm_main.py` entry point as a bridge. It loaded the environment, configured LiteLLM for Bedrock, and exposed the selected models through an OpenAI-compatible endpoint.

Conceptually, the request path became:

```text
OpenAI-compatible client
				-> local LiteLLM proxy
				-> Bedrock Runtime API
				-> Claude model
```

The client uses the local proxy URL and `LITELLM_MASTER_KEY`. The proxy uses `AWS_REGION` and `AWS_BEARER_TOKEN_BEDROCK`. This keeps Bedrock authentication inside the proxy while allowing existing OpenAI SDK clients to keep their normal request format.

For example, an OpenAI-compatible client can target the local server:

```python
from openai import OpenAI

client = OpenAI(
		base_url="http://localhost:4000/v1",
		api_key="<LITELLM_MASTER_KEY>",
)

response = client.chat.completions.create(
		model="claude-bedrock",
		messages=[{"role": "user", "content": "Say hello."}],
)

print(response.choices[0].message.content)
```

The exact Claude model ID must be enabled for the AWS account and region. Model access remains an Amazon Bedrock concern, independent of whether the proxy accepts bearer-token authentication.

## What I Learned About Using AI

AI was valuable throughout this task. It produced useful configuration, implemented the listing script, and eventually created the LiteLLM bridge correctly. But it also gave confident, outdated, and unhelpful answers along the way.

The lesson is not that AI is unreliable. The lesson is that good output depends on good input and good direction. When a conversation stops making progress, changing the prompt can be more effective than repeatedly asking the same question. In this case, asking it to search for current solutions changed the investigation from speculation into documentation-backed implementation.

For practical engineering work, AI is strongest when it is treated as a fast collaborator: give it a concrete requirement, verify important claims against primary documentation, and redirect the investigation when its assumptions do not match the observed system.
