---
title: "Check Docker Containers over SSH from Windows"
date: 2026-09-06
tags: [docker, ssh, windows, powershell, linux]
---

# Check Docker Containers over SSH from Windows

When Docker runs on a remote Linux host, SSH can inspect a container without opening an interactive shell. For example, this command prints the container status as a JSON string:

```bash
docker inspect --format '{{json .State.Status}}' container_name
```

Docker's `--format` option accepts a Go template. The `json` template function formats the selected value, and `.State.Status` selects the container status.

## From a Windows Command Prompt

In Windows Command Prompt, escape the double quotes inside the remote command with backslashes:

```cmd
ssh remote_host "docker inspect --format \"{{json .State.Status}}\" container_name"
```

The outer double quotes keep the complete Docker command together as the SSH command argument. The escaped inner quotes are passed through so the remote shell receives the format expression as one argument.

## From PowerShell

PowerShell uses single quotes around the complete remote command:

```powershell
ssh remote_host 'docker inspect --format \"{{json .State.Status}}\" container_name'
```

Here, the outer single-quoted string is literal. This preserves the backslashes and inner double quotes for the native `ssh` executable. The backslash is not PowerShell's escape character; PowerShell uses the backtick for escaping. The backslashes in this command are part of the quoting required by the Windows native-command argument path.

## Check More Than the Status

The same pattern works for other fields. For example, print the image used by a container:

```powershell
ssh remote_host 'docker inspect --format \"{{json .Config.Image}}\" container_name'
```

Or print the complete inspection result as JSON:

```powershell
ssh remote_host 'docker inspect container_name'
```

The container can be running or stopped; `docker inspect` reads its stored metadata. Replace `remote_host` and `container_name` with the SSH host and Docker container name or ID.

## List Running Containers

Use `docker ps` for a compact view of running containers. A custom format can print only the command:

```bash
docker ps --format "{{.Command}}"
```

For a table containing the most useful fields:

```bash
docker ps --format "table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.Status}}\t{{.Names}}"
```

The command and other columns are truncated by default. Add `--no-trunc` when the full command is important:

```bash
docker ps --no-trunc --format "table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.Status}}\t{{.Names}}"
```

When these commands are sent through SSH from Windows, apply the same outer-shell quoting rules shown above. For example, from PowerShell:

```powershell
ssh remote_host 'docker ps --no-trunc --format \"table {{.ID}}\t{{.Image}}\t{{.Command}}\t{{.Status}}\t{{.Names}}\"'
```

## Inspect Container Configuration

`docker inspect` exposes the configuration that was recorded when the container was created. Query individual fields instead of reading the complete JSON document when diagnosing one setting:

```bash
docker inspect --format "{{json .Config.Cmd}}" container_name_or_id
docker inspect --format "{{json .Config.Entrypoint}}" container_name_or_id
docker inspect --format "{{json .Config.Env}}" container_name_or_id
```

These fields describe different parts of the process configuration:

| Inspect field | Meaning |
| --- | --- |
| `.Config.Cmd` | The default command and arguments configured for the image, or the command supplied when the container was created |
| `.Config.Entrypoint` | The executable configured as the container entrypoint |
| `.Config.Env` | Environment variables stored in the container configuration |

The values in `.Config.Cmd` and `.Config.Entrypoint` are normally JSON arrays. Using `{{json ...}}` keeps the array structure visible instead of printing a less precise Go-style representation.

## Map `docker run` Options to Inspect Fields

When investigating how a container was launched, compare the original `docker run` option with the corresponding `docker inspect` field:

| `docker run` option | Corresponding inspect field |
| --- | --- |
| `-e NAME=value` | `.Config.Env` |
| `--entrypoint executable` | `.Config.Entrypoint` |
| Command and arguments after the image name | `.Config.Cmd` |
| `-p 8000:8000` | `.HostConfig.PortBindings` |
| `-v host:container` | `.HostConfig.Binds` and `.Mounts` |
| `--network network_name` | `.HostConfig.NetworkMode` |
| `--restart unless-stopped` | `.HostConfig.RestartPolicy` |
| `--gpus all` | `.HostConfig.DeviceRequests` |
| `--rm` | `.HostConfig.AutoRemove` |

For example, this command creates a container with several of those options:

```bash
docker run -d --name web \
	-e APP_ENV=production \
	--entrypoint /app/start.sh \
	-p 8000:8000 \
	-v /srv/web-data:/var/lib/web \
	--network app-net \
	--restart unless-stopped \
	image_name --listen 0.0.0.0:8000
```

The relevant inspection commands are:

```bash
docker inspect --format "{{json .Config.Env}}" web
docker inspect --format "{{json .Config.Entrypoint}}" web
docker inspect --format "{{json .Config.Cmd}}" web
docker inspect --format "{{json .HostConfig.PortBindings}}" web
docker inspect --format "{{json .HostConfig.Binds}}" web
docker inspect --format "{{json .HostConfig.NetworkMode}}" web
docker inspect --format "{{json .HostConfig.RestartPolicy}}" web
docker inspect --format "{{json .HostConfig.DeviceRequests}}" web
docker inspect --format "{{json .HostConfig.AutoRemove}}" web
```

## Check Runtime State

The following fields help distinguish a stopped container, a failed process, and a container that is currently restarting:

```bash
docker inspect --format "status={{.State.Status}} exit={{.State.ExitCode}} restarting={{.State.Restarting}} restarts={{.RestartCount}}" container_name_or_id
```

| Inspect field | What it tells you |
| --- | --- |
| `.State.Status` | The current lifecycle status, such as `created`, `running`, `paused`, `restarting`, or `exited` |
| `.State.ExitCode` | The exit code from the container's main process |
| `.State.Restarting` | Whether Docker currently considers the container to be restarting |
| `.RestartCount` | How many times Docker has restarted the container |

For mounts, inspect both the high-level mount list and the individual fields:

```bash
docker inspect --format "{{json .Mounts}}" container_name_or_id
```

Each mount entry can include:

| Mount field | Meaning |
| --- | --- |
| `.Mounts[].Type` | The mount type, such as `bind` or `volume` |
| `.Mounts[].Source` | The source path or volume name on the Docker host |
| `.Mounts[].Destination` | The path inside the container |

## The Important Rule

There are two parsers involved: the local Windows shell and the remote command interpreter. Quote the entire remote command for `ssh`, then escape the quotes that must survive until Docker parses `--format`.

| Local shell | Command form |
| --- | --- |
| Command Prompt | `ssh remote_host "docker inspect --format \"{{json .State.Status}}\" container_name"` |
| PowerShell | `ssh remote_host 'docker inspect --format \"{{json .State.Status}}\" container_name'` |

## References

- [Docker ps CLI reference](https://docs.docker.com/reference/cli/docker/ps/): documents custom `--format` templates, table output, available placeholders, and `--no-trunc`.
- [Docker inspect CLI reference](https://docs.docker.com/reference/cli/docker/inspect/): documents `--format`, Go templates, and the `json` template function.
- [Docker run documentation](https://docs.docker.com/engine/containers/run/): documents the `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]` structure and the command, entrypoint, port, environment, network, and mount options used in the examples.
- [PowerShell about_Quoting_Rules](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_quoting_rules): documents literal single-quoted strings and passing quoted arguments to native commands.
- [PowerShell about_Parsing](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_parsing): documents native-command argument parsing and the Windows handling of arguments containing quote characters.
