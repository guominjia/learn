---
layout: post
title: "Why a systemd-Enabled GitHub Actions Runner Still Shows Offline"
date: 2026-08-15
categories: [github, linux]
tags: [github-actions, self-hosted-runner, systemd, restart-policy]
---

A self-hosted runner service can be `enabled` in systemd and still show **Offline** on GitHub. The cause is usually confusing `enabled` (boot-time autostart) with `Restart=` (post-crash autostart) — two unrelated mechanisms.

## Symptom

```text
$ systemctl status actions.runner.myorg-myrepo.myhost.service
Loaded: loaded (...; enabled; vendor preset: enabled)
```

The unit is enabled, so it starts at boot. Yet GitHub's runner list shows the runner as offline.

## Root cause: two different guarantees

| Setting | Triggers on | Does NOT trigger on |
|---|---|---|
| `enabled` (`[Install] WantedBy=multi-user.target`) | System boot reaching `multi-user.target` | Runner process exiting afterward |
| `Restart=` (`[Service]`) | Process exit *after* it has started | Anything before first start |

```ini
[Install]
WantedBy=multi-user.target   # enabled -> starts once at boot

[Service]
Restart=on-failure           # restarts only after boot, on exit
RestartSec=10
```

So after a reboot, `enabled` alone is enough to launch the listener. But if the listener later exits — network blip, invalid registration, unhandled exception — and there's no `Restart=`, systemd leaves the unit **stopped**. `enabled` never re-fires until the next boot.

## Why this runner specifically stayed offline

Its GitHub registration had been deleted. Each time the listener started, it discovered the registration was invalid and exited (a clean, intentional exit — not a crash). Without a restart policy, systemd took no further action, so the unit sat stopped between boots while GitHub kept reporting `Offline`.

## Fix

**1. Re-register the runner first.** No restart policy fixes an invalid registration — the listener will just fail every `RestartSec` interval instead of once. Confirm the runner shows `Idle` on GitHub before touching the unit file.

**2. Add a restart policy** to `/etc/systemd/system/actions.runner.myorg-myrepo.myhost.service`:

```ini
[Service]
Restart=always
RestartSec=10
```

or the more conservative variant:

```ini
[Service]
Restart=on-failure
RestartSec=10
```

`on-failure` only restarts on non-zero exit, signal termination, or crash — a runner that exits cleanly (e.g. detecting a dead registration) won't be retried. `always` retries regardless of exit reason, which is generally what you want for a self-hosted runner meant to stay online continuously.

**3. Apply the change:**

```bash
sudo systemctl daemon-reload
sudo systemctl restart actions.runner.myorg-myrepo.myhost.service
```

## Mental model

```ini
[Install]
WantedBy=multi-user.target   # enabled: autostart once, at boot

[Service]
Restart=always               # autostart again, every time it later exits
RestartSec=10
```

`enabled` and `Restart=` aren't redundant — they cover different points in the service's lifetime, and a long-lived runner needs both.