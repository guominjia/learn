# Claude via Bedrock API Key

The script calls the Bedrock Runtime `converse` API with
`AWS_BEARER_TOKEN_BEDROCK`; it does not require AWS access-key credentials.

```powershell
. .\.venv\Scripts\Activate.ps1
. ..\env\env.ps1
python .\main.py
```

The default model is `us.anthropic.claude-sonnet-4-6`. Pass an allowed Bedrock
model or inference-profile ID when your account uses a different Claude model:

```powershell
python .\main.py "Explain prompt caching in one sentence." --model us.anthropic.claude-sonnet-4-6
```

The environment script must define `AWS_REGION` and
`AWS_BEARER_TOKEN_BEDROCK`.
