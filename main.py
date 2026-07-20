"""Send a prompt to Claude through Amazon Bedrock using a bearer token."""

from __future__ import annotations

import argparse
import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


DEFAULT_MODEL = "us.anthropic.claude-sonnet-4-6"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Reply with exactly: Claude Bedrock works",
        help="User prompt to send to Claude.",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("CLAUDE_MODEL", DEFAULT_MODEL),
        help=f"Bedrock model or inference profile ID (default: {DEFAULT_MODEL}).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List models available through the Bedrock catalog in AWS_REGION.",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    return parser.parse_args()


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def invoke_claude(
    *, model: str, prompt: str, region: str, bearer_token: str, max_tokens: int
) -> dict[str, object]:
    url = f"https://bedrock-runtime.{region}.amazonaws.com/model/{model}/converse"
    authorization = bearer_token.strip()
    if not authorization.lower().startswith("bearer "):
        authorization = f"Bearer {authorization}"

    payload = {
        "messages": [{"role": "user", "content": [{"text": prompt}]}],
        "inferenceConfig": {"maxTokens": max_tokens},
    }
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": authorization,
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urlopen(request, timeout=90) as response:
        return json.load(response)


def list_models(*, region: str, bearer_token: str) -> list[dict[str, object]]:
    url = f"https://bedrock.{region}.amazonaws.com/foundation-models"
    authorization = bearer_token.strip()
    if not authorization.lower().startswith("bearer "):
        authorization = f"Bearer {authorization}"

    request = Request(url, headers={"Authorization": authorization})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    return payload["modelSummaries"]


def print_models(models: list[dict[str, object]]) -> None:
    print(f"Total models: {len(models)}")
    for model in sorted(models, key=lambda item: str(item["modelId"])):
        inference_apis = model.get("inferenceAPIsSupported", {})
        supported_apis = ", ".join(
            api for api, enabled in inference_apis.items() if enabled
        )
        print(
            f"{model['modelId']}\t{model['providerName']}\t"
            f"{model['modelName']}\t{supported_apis}"
        )


def response_text(response: dict[str, object]) -> str:
    try:
        content = response["output"]["message"]["content"]
        return "".join(block["text"] for block in content if "text" in block)
    except (KeyError, TypeError) as error:
        raise RuntimeError(f"Unexpected Bedrock response: {response}") from error


def main() -> None:
    arguments = parse_arguments()
    region = require_environment("AWS_REGION")
    token = require_environment("AWS_BEARER_TOKEN_BEDROCK")

    try:
        if arguments.list_models:
            print_models(list_models(region=region, bearer_token=token))
            return

        response = invoke_claude(
            model=arguments.model,
            prompt=arguments.prompt,
            region=region,
            bearer_token=token,
            max_tokens=arguments.max_tokens,
        )
    except HTTPError as error:
        detail = error.read().decode(errors="replace")
        raise RuntimeError(f"Bedrock returned HTTP {error.code}: {detail}") from error

    print(response_text(response))


if __name__ == "__main__":
    main()
