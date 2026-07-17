---
layout: post
title: "Using AI for Repetitive Docker Debugging"
date: 2026-07-17
tags: [ai, docker, debugging, aws, python, proxy]
---

AI is especially useful for repetitive engineering work: preparing test files, running the same checks, collecting output, and cleaning up afterwards. The engineer still decides what should be tested and reviews the generated script, but AI can remove much of the mechanical work.

Today I used this approach while debugging a Docker container that could not access an AWS port as expected.

## The Repetitive Test Cycle

At first, I spent too long manually verifying each step. The test required copying a Python script to a remote host, placing it in a Docker container, executing it, and removing the temporary files afterwards.

I then gave AI the operational requirements directly:

```text
I need to solve a Docker problem. I need to add a script to Docker,
test it, and then delete it. The commands are:

ssh aisut rm test_model.py
ssh aisut docker exec docker-ragflow-cpu-1 rm /root/test_model.py
ssh aisut docker exec docker-ragflow-cpu-1 uv run /root/test_model.py
ssh aisut docker cp test_model.py docker-ragflow-cpu-1:/root/
scp tests\test_model.py aisut:~/

Now test the new file and check whether there is a problem inside Docker.
```

AI turned those individual commands into a test script. I only needed to review the script, confirm the execution order, and run it. This is a good division of work: the human supplies the goal and validates the potentially destructive commands, while AI handles repeatable orchestration.

A safe test cycle is:

1. Copy the current local test file to the remote host.
2. Copy it into the target container.
3. Run the test with the container's actual runtime.
4. Collect the result.
5. Remove the temporary files from both the host and the container.

The important point is to test the real container environment. A successful test on the host does not prove that the application will behave the same way inside Docker.

## A Misleading Successful Test

The first generated test used `requests`. It successfully connected to the AWS endpoint from inside the container.

However, the real application script still failed. Comparing the two scripts carefully revealed a small but important difference:

- The temporary test used `requests`.
- The application used `aiohttp`.

That difference changed how proxy configuration was handled.

## The Proxy Difference Between `requests` and `aiohttp`

The container had proxy environment variables configured. By default, `requests` reads common proxy environment variables such as `HTTP_PROXY`, `HTTPS_PROXY`, and `NO_PROXY`. Therefore, its request could use the configured proxy and succeed.

`aiohttp` does not use proxy environment variables by default. A minimal `aiohttp` test could therefore fail even though the equivalent `requests` test succeeds.

After I asked AI to change the test from `requests` to `aiohttp`, it reproduced the real failure. AI then identified the likely cause and proposed the correct fix: enable environment-based proxy discovery for the `aiohttp` session.

```python
import aiohttp

async with aiohttp.ClientSession(trust_env=True) as session:
	async with session.get("https://example.amazonaws.com") as response:
		print(response.status)
```

With `trust_env=True`, `aiohttp` reads the proxy-related environment variables from the container. This makes its behavior match the environment-aware behavior expected in this deployment.

Before changing application code, it is also useful to inspect the relevant variables in the actual container:

```bash
ssh aisut docker exec docker-ragflow-cpu-1 env
```

Look for `HTTP_PROXY`, `HTTPS_PROXY`, `ALL_PROXY`, and `NO_PROXY`. The values determine whether a request should go through a proxy or connect directly.

## What I Learned

AI did not replace the debugging process. It made the repeated parts cheap enough that I could focus on the real question: why did the test and the application behave differently?

The useful workflow was:

1. Describe the desired test lifecycle precisely.
2. Let AI generate a repeatable script.
3. Review the commands before execution.
4. Reproduce the failure using the same library and runtime as the application.
5. Compare environmental assumptions, especially proxy configuration.

For Docker and networking problems, a test is only trustworthy when it uses the same container, dependencies, and environment behavior as the production code. In this case, replacing `requests` with `aiohttp` in the test exposed the real issue, and `trust_env=True` provided the solution.
