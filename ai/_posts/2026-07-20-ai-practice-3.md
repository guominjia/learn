---
title: AI Practice 3 - Enabling RAGFlow HTTPS Through SSH
categories: [ai]
tags: [ai, skill, docker, ssh, ragflow, https, mkcert]
---

This practice was about using an AI coding agent to enable HTTPS for a remote RAGFlow deployment. The task was small on paper: I already had `mkcert-key.pem`, `mkcert.pem`, and `mkcert.exe`; the agent needed to configure RAGFlow to use them on the server. In practice, the session was a useful test of whether an agent could operate carefully across SSH, Docker, Nginx, certificates, and an existing repository.

## Creating an SSH Docker Skill

I first created a skill named `ssh-docker-operation`:

```yaml
---
name: ssh-docker-operation
description: Operate Docker workloads on the xxx host through SSH. Use for inspecting container status, reading logs, executing commands in containers, and performing explicitly requested Docker lifecycle actions.
Keywords: Docker, container, logs, docker exec, ragflow, ssh xxx.
---
```

The skill establishes a simple but important rule: Docker commands for this environment must run through SSH rather than against a presumed local Docker daemon.

```powershell
ssh xxx "docker <docker-subcommand> <arguments>"
```

Its operating principles were equally useful:

- Begin with read-only inspection unless a state-changing action is explicitly requested.
- Quote paths, filters, and shell expressions that the local shell might interpret.
- Use the exact container name from `docker ps`; do not guess.
- Bound log output with `--tail` before requesting live streaming with `-f`.
- Do not run destructive operations, including container restarts, volume deletion, or system pruning, without explicit approval.

This is a small skill, but it gives the agent an operational boundary. It makes the remote host an intentional target instead of an accidental side effect of whatever Docker context happens to be active locally.

## Finding the HTTPS Configuration

I then told the agent that I had the certificate files and asked it to enable HTTPS for RAGFlow. It correctly found:

```text
ragflow/docker/nginx/ragflow.https.conf
```

It also correctly identified the file paths expected by the Nginx configuration:

```text
/etc/nginx/ssl/fullchain.pem
/etc/nginx/ssl/privkey.pem
```

The agent used SCP to stage the certificate files, adjusted their permissions, and inspected the certificate with:

```bash
openssl x509 -in nginx/ssl/fullchain.pem -noout -subject -issuer -ext subjectAltName
```

That verification step mattered. The certificate had been generated earlier for a different machine, so its Subject Alternative Name did not include the hostname of the RAGFlow server.

Instead of enabling a technically running but browser-invalid HTTPS endpoint, the agent explained the consequence clearly:

> Activating it would make HTTPS start but produce a hostname-validation error for the requested server, so I’m removing the staged certificate copies and leaving the running HTTP configuration unchanged.

This was the best part of the session. The agent did not confuse a successful Nginx startup with a correct HTTPS deployment. It validated the certificate against the real hostname requirement and then cleaned up the temporary changes rather than leaving a misleading configuration behind.

## Generating a Correct Certificate

After that, I asked it to use `mkcert.exe` to generate a certificate for the server. It completed the task without difficulty. This was a much better route because the new certificate could include the server hostname in its Subject Alternative Name.

One implementation detail surprised me: it used `perl -0pi -e` to modify a configuration file. I rarely use Perl, but the command is a compact way to perform a multi-line, in-place replacement. For an agent working in a shell-heavy environment, it is a reasonable tool when the intended replacement is precise and verified afterward.

## Enabling HTTPS in RAGFlow

With a certificate that includes the RAGFlow hostname, HTTPS can be enabled through the existing Docker and Nginx configuration rather than by adding a new application flag.

First, add the HTTPS configuration, the shared Nginx files, and the read-only certificate directory to the Nginx service in `docker/docker-compose.yml`:

```yaml
volumes:
	- ./nginx/ragflow.https.conf:/etc/nginx/conf.d/ragflow.conf.https
	- ./nginx/proxy.conf:/etc/nginx/proxy.conf
	- ./nginx/nginx.conf:/etc/nginx/nginx.conf
	- ./nginx/ssl:/etc/nginx/ssl:ro
```

Place the generated certificate and private key in the mounted directory using the names referenced by the configuration:

```text
docker/nginx/ssl/fullchain.pem
docker/nginx/ssl/privkey.pem
```

Next, update `docker/nginx/ragflow.https.conf`. Its `server_name` must be the public DNS name used to reach RAGFlow. before recreating the service, inspect the certificate again and test the Nginx configuration in the container with `nginx -t`.

## What Went Wrong

The overall process was not flawless. The agent damaged `docker/nginx/ragflow.https.conf` while editing it. It only repaired the file after I explicitly asked it to check the configuration, confirm the change, and fix it.

That failure exposed an important weakness: a shell-based edit is not complete when the command exits successfully. The agent should have immediately inspected the resulting diff or file contents, then performed a syntax or configuration test before treating the change as done.

It also introduced unnecessary configuration logic:

```bash
ENABLE_HTTPS=true

if [ "${ENABLE_HTTPS:-false}" = "true" ]; then
	cp -f "$NGINX_CONF_DIR/ragflow.conf.https" "$NGINX_CONF_DIR/ragflow.conf"
	echo "Applied nginx HTTPS configuration"
elif [ -n "$API_PROXY_SCHEME" ]; then
```

This was extra code with a larger maintenance surface than the request required. HTTPS already had an intended configuration path, and the task was to supply valid certificates and activate that path. Adding a new environment flag and conditional branch made the solution harder to reason about and increased the chance of future regressions.

## Lessons

This session changed how I evaluate an AI agent for infrastructure work.

- A narrowly scoped operational skill is valuable. It establishes where commands run and what kinds of actions require explicit approval.
- Certificate inspection is not optional. A certificate can be valid in structure but invalid for the hostname users actually visit.
- Cleaning up after a failed precondition is a strong sign of safe behavior.
- Shell edits need immediate verification. A command such as `perl -0pi -e` is powerful, but it should be followed by a file review, `nginx -t`, and a targeted container check.
- The smallest working change is usually the best change. Avoid new flags and branches when the existing HTTPS mechanism already supports the requested outcome.

The agent handled the hard conceptual part well: it recognized that a certificate mismatch should block activation. The main area for improvement is discipline after editing: inspect the changed configuration, test it before deployment, and avoid adding logic that the task does not need.
