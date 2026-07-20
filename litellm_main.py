"""Start an OpenAI-compatible LiteLLM proxy for Bedrock catalog models."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4000)
    parser.add_argument(
        "--master-key",
        default=os.environ.get("LITELLM_MASTER_KEY", "local-bedrock-proxy-key"),
        help="API key required by clients connecting to this proxy.",
    )
    return parser.parse_args()


def require_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def list_models(*, region: str, bearer_token: str) -> list[dict[str, object]]:
    url = f"https://bedrock.{region}.amazonaws.com/foundation-models"
    authorization = bearer_token.strip()
    if not authorization.lower().startswith("bearer "):
        authorization = f"Bearer {authorization}"

    request = Request(url, headers={"Authorization": authorization})
    with urlopen(request, timeout=30) as response:
        return json.load(response)["modelSummaries"]


def list_inference_profile_ids(*, region: str, bearer_token: str) -> set[str]:
    url = f"https://bedrock.{region}.amazonaws.com/inference-profiles"
    authorization = bearer_token.strip()
    if not authorization.lower().startswith("bearer "):
        authorization = f"Bearer {authorization}"

    request = Request(url, headers={"Authorization": authorization})
    with urlopen(request, timeout=30) as response:
        profiles = json.load(response)["inferenceProfileSummaries"]
    return {profile["inferenceProfileId"] for profile in profiles}


def resolve_model_id(model_id: str, profile_ids: set[str]) -> str:
    for prefix in ("us.", "global."):
        profile_id = f"{prefix}{model_id}"
        if profile_id in profile_ids:
            return profile_id
    return model_id


def create_config(
    *,
    models: list[dict[str, object]],
    profile_ids: set[str],
    region: str,
    master_key: str,
) -> str:
    model_list = [
        {
            "model_name": model["modelId"],
            "litellm_params": {
                "model": (
                    "bedrock/converse/"
                    f"{resolve_model_id(model['modelId'], profile_ids)}"
                ),
                "aws_region_name": region,
                "api_key": "os.environ/AWS_BEARER_TOKEN_BEDROCK",
            },
        }
        for model in models
        if model.get("inferenceAPIsSupported", {}).get("converse")
    ]
    config = {
        "model_list": model_list,
        "general_settings": {"master_key": master_key},
    }
    config_file = tempfile.NamedTemporaryFile(
        mode="w", suffix=".json", prefix="litellm-bedrock-", delete=False
    )
    with config_file:
        json.dump(config, config_file, indent=2)
    return config_file.name


def main() -> None:
    arguments = parse_arguments()
    region = require_environment("AWS_REGION")
    token = require_environment("AWS_BEARER_TOKEN_BEDROCK")
    models = list_models(region=region, bearer_token=token)
    profile_ids = list_inference_profile_ids(region=region, bearer_token=token)
    config_path = create_config(
        models=models,
        profile_ids=profile_ids,
        region=region,
        master_key=arguments.master_key,
    )
    print(f"Exposing {len(models)} Bedrock models at http://{arguments.host}:{arguments.port}/v1")
    print(f"Client API key: {arguments.master_key}")

    try:
        subprocess.run(
            [
                str(Path(sys.executable).with_name("litellm.exe")),
                "--config",
                config_path,
                "--host",
                arguments.host,
                "--port",
                str(arguments.port),
                "--num_workers",
                "1",
            ],
            check=True,
        )
    finally:
        Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    main()