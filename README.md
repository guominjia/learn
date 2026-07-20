# Claude via Bedrock API Key

The script calls the Bedrock Runtime `converse` API with
`AWS_BEARER_TOKEN_BEDROCK`; it does not require AWS access-key credentials.

```powershell
. .\.venv\Scripts\Activate.ps1
. ..\env\env.ps1
python .\main.py
```

List all model IDs that the current bearer token can discover in the selected
AWS Region:

```powershell
python .\main.py --list-models
```

The default model is `us.anthropic.claude-sonnet-4-6`. Pass an allowed Bedrock
model or inference-profile ID when your account uses a different Claude model:

```powershell
python .\main.py "Explain prompt caching in one sentence." --model us.anthropic.claude-sonnet-4-6
```

The environment script must define `AWS_REGION` and
`AWS_BEARER_TOKEN_BEDROCK`.

## OpenAI-compatible Proxy

`litellm_main.py` discovers the current Bedrock catalog and starts LiteLLM with
every `converse`-capable model under its native Bedrock model ID.

```powershell
$env:LITELLM_MASTER_KEY = "choose-a-local-api-key"
python .\litellm_main.py
```

The proxy listens on `http://127.0.0.1:4000/v1`. For example:

```powershell
$headers = @{ Authorization = "Bearer $env:LITELLM_MASTER_KEY" }
Invoke-RestMethod http://127.0.0.1:4000/v1/models -Headers $headers
```

Use a model ID returned by `python .\main.py --list-models` in OpenAI-compatible
requests. Catalog discovery does not guarantee every model can be invoked; the
Bedrock token's model access policy remains authoritative.
