---
layout: post
title: "Why a Python Service and an Interactive Shell Used Different Data Files"
date: 2026-08-13
categories: [microsoft, python]
tags: [windows, service, localsystem, pathlib, data, environment-variables]
---

A Python app appeared to lose data whenever it ran as a Windows service. The data was present when the program ran from an interactive PowerShell session, but the service could not find it.

The two processes were using different paths because they ran under different Windows accounts.

This article documents the diagnosis and a durable design: make the persistent data location explicit instead of deriving it solely from `Path.home()`.

## The symptom

The application built its default data path from the current account's home directory:

```python
from pathlib import Path

data_path = Path.home() / ".app" / "app.data"
```

An interactive run created and used:

```text
C:\Users\<username>\.app\app.data
```

The Windows service, ran as `LocalSystem`. Its `Path.home()` value resolved to:

```text
C:\Windows\System32\config\systemprofile
```

Therefore, its default data location was different:

```text
C:\Windows\System32\config\systemprofile\.app\app.data
```

The service was not reading the user's data. It was resolving a different file name before opening a connection.

## Confirm the service account

Inspect the service configuration from an elevated or ordinary PowerShell session:

```powershell
sc.exe qc YourService
```

The result identifies the configured service account. In this case, it showed `LocalSystem`.

`LocalSystem` is a predefined account used by the Service Control Manager. Microsoft notes that it is not associated with a logged-on user, so it must not be treated as the interactive user's identity or profile.

## Check the two possible files

The following command tests the expected user-profile path and the `LocalSystem` profile path independently:

```powershell
Test-Path "C:\Windows\System32\config\systemprofile\.app\app.data"
Test-Path "C:\Users\<username>\.app\app.data"
```

In this incident, the user-profile data file existed. Access to the system-profile path was not available to the interactive user without elevation, which was an additional reminder that the service and shell did not share the same context.

Do not use a failed file check alone to conclude that a data file is absent. A false result can mean that the file is missing, inaccessible, or both. Log the exact resolved path from each process as well.

## Why `Path.home()` changes

`Path.home()` returns the current user's home directory through `os.path.expanduser()`. Consequently, it is a reasonable default for per-user command-line tools, but not a stable application-data location when the same program can run under several service accounts.

This is the important distinction:

```text
same program + different Windows account = potentially different Path.home() + different data file
```

The issue applies to any state derived from a per-account home directory: configuration files, token caches, logs, local key material, or data files can split in the same way.

## Make the data path explicit

Keep the current home-directory location as a useful developer default, but allow deployment configuration to override it. For example, use one environment variable with an absolute path:

```python
import os
from pathlib import Path


def get_data_path() -> Path:
	configured_path = os.getenv("APP_DATA_PATH")
	if configured_path:
		return Path(configured_path).expanduser()

	return Path.home() / ".app" / "app.data"
```

Create the parent directory before initializing data storage:

```python
data_path = get_data_path()
data_path.parent.mkdir(parents=True, exist_ok=True)
```

Set `APP_DATA_PATH` in the service's own launch configuration and use the same value for interactive troubleshooting runs. For example:

```text
APP_DATA_PATH=C:\ProgramData\App\app.data
```

The exact place to set this variable depends on the service wrapper or installer. The essential requirement is that the process environment contains the value **before** Python starts. Python exposes environment variables through `os.environ` and `os.getenv()` as process-environment values, so changing a user environment variable after a service is already running does not reconfigure that existing process. Restart the service after updating its launch configuration.

## Choose the shared location deliberately

An explicit common path prevents accidental data splitting, but it changes the security model. A file under one interactive user's profile is normally a poor shared target because the `LocalSystem` service may not have access and another user may later inherit unintended access.

For machine-wide state, choose a directory intended for application data, such as a dedicated directory under `C:\ProgramData`, then apply an access control list that grants access only to the service identity and the administrators or operators who must manage the data. Do not grant broad write access merely to make the service start.

If the data file contains sensitive data, also decide whether the service should share the same data scope as the interactive client at all. A shared path is a storage decision; it does not automatically make encryption, user identity, or authorization semantics shared safely.

## Verify after deployment

After configuring the override and restarting the service, verify the effective path from both execution modes. A temporary diagnostic is enough:

```python
print(get_data_path().resolve())
```

Then check the exact file with PowerShell:

```powershell
Test-Path "C:\ProgramData\App\app.data"
```

Finally, confirm that data written by one intended execution mode is visible to the other. Remove or downgrade the path logging after diagnosis if the full path is sensitive in your operational environment.

## References

- [Python pathlib.Path.home](https://docs.python.org/3/library/pathlib.html#pathlib.Path.home): defines `Path.home()` as the home directory returned through `os.path.expanduser()`.
- [Python os.environ and os.getenv](https://docs.python.org/3/library/os.html#os.environ): describes Python's process-environment mapping and environment-variable lookup.
- [Microsoft LocalSystem account](https://learn.microsoft.com/en-us/windows/win32/services/localsystem-account): documents LocalSystem as a Service Control Manager account that is not associated with a logged-on user.