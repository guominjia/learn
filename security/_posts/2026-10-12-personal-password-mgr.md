---
layout: post
title: "Open-Source Options for a Personal Credential Manager: Vaultwarden, Infisical, and NetBox"
date: 2026-10-12
categories: [security]
tags: [password-manager, secrets-management, vaultwarden, bitwarden, keepassxc, infisical, openbao, sops, netbox, cmdb, self-hosted]
---

## Open-Source Options

**Team or personal password vaults:**

| Project | License | Highlights |
|---|---|---|
| [Vaultwarden](https://github.com/dani-garcia/vaultwarden) | AGPL-3.0 | Bitwarden-compatible server written in Rust, 68.7k stars. Works with the official Bitwarden browser extensions, desktop, mobile, and CLI clients. Supports organizations/collections sharing, 2FA, and event logs. Requires HTTPS |
| [Passbolt CE](https://github.com/passbolt/passbolt_api) | AGPL-3.0 | Team-oriented; each user holds their own private key for end-to-end encryption. Sharing, auditing, and a Go CLI |
| [TeamPass](https://github.com/nilsteampassnet/TeamPass) | GPL-3.0 | PHP/MySQL, folder-based permissions, AES-256-GCM, a per-user RSA wrapping key, LDAP/AD support |
| [KeePassXC](https://github.com/keepassxreboot/keepassxc) | GPL-2/3 | Offline encrypted KDBX file, ships `keepassxc-cli` and SSH Agent integration, no server |

**API keys / environment-variable secrets:**

| Project | License | Highlights |
|---|---|---|
| [Infisical](https://github.com/Infisical/infisical) | MIT (except the `ee` directory) | Secrets organized by project and environment, with versioning, rotation, RBAC, audit logs, and a Python SDK. The CLI injects secrets into the command it runs, which maps directly to an `environment_name` design |
| [OpenBao](https://github.com/openbao/openbao) | MPL-2.0 | Open-source fork of HashiCorp Vault: KV secrets, dynamic secrets, leases, revocation, and encryption as a service. Heavyweight |
| [SOPS](https://github.com/getsops/sops) | MPL-2.0 | Encrypts YAML/JSON/ENV files so they can live in git; supports age/PGP/cloud KMS. No server |

**Host and service inventory:**

| Project | License | Highlights |
|---|---|---|
| [NetBox](https://github.com/netbox-community/netbox) | Apache-2.0 | Django/Python. Devices, IPs, service ports, custom fields (GPU, memory, etc.), REST/GraphQL API, and change logging. Inventory only; passwords belong in another system |
| [Snipe-IT](https://github.com/snipe/snipe-it) | AGPL-3.0 | Focused on IT asset and license management (who checked out which machine) |
| [GLPI](https://github.com/glpi-project/glpi) | GPL-3.0 | Asset management, CMDB, ticketing, and DCIM in one; too heavy for this use case |

**Python libraries:**

| Library | Purpose |
|---|---|
| [pykeepass](https://github.com/libkeepass/pykeepass) (GPL-3.0) | Read/write KDBX; can share the same database file with KeePassXC |
| [keyring](https://github.com/jaraco/keyring) | Delegates passwords to OS credential stores such as Windows Credential Locker |

## Infisical

Infisical is a secrets management platform for **programs and machines**, not a password book for humans. It addresses API keys, database passwords, and certificates scattered across `.env` files, source code, CI configs, and cloud consoles: it stores them centrally and delivers them on demand to applications, pipelines, and Kubernetes.

**Core usage**

```
Developer / CI / K8s Pod
   └─ infisical run --env=prod -- python app.py   # CLI injects secrets as env vars at runtime
   └─ SDK (Python/Node/Go/...) fetches directly
   └─ K8s Operator / Agent syncs and restarts the Deployment on change
            ↑
      Infisical Server (Project → Environment(dev/staging/prod) → Secret)
```

**Main modules** (from the official README and docs)

| Module | Purpose |
|---|---|
| Secrets Management | Secrets per project and environment (dev/prod), with version history and point-in-time rollback. Syncs to GitHub, Vercel, AWS Secrets Manager, and more |
| Secret Rotation / Dynamic Secrets | Periodically rotates credentials for PostgreSQL, MySQL, AWS IAM, etc., or issues short-lived credentials on demand |
| Secret Scanning | `infisical scan` finds leaked secrets in git history (140+ types); can also run as a pre-commit hook |
| PKI | Private CA that issues, renews, and revokes X.509 certificates; supports ACME/EST |
| KMS | Centralized key management with encrypt/decrypt APIs |
| PAM | Users sign in via SSO, then get just-in-time access to DB, SSH, K8s, and RDP without ever seeing the underlying password; sessions can be recorded |
| Agent Vault | AI agents call external APIs without holding real credentials; a proxy injects them in transit |
| Platform | Machine identity auth (K8s, AWS, Azure, GCP, OIDC Auth), RBAC, approval workflows, audit logs, self-hosting |

**License**: MIT, except the `ee/` directory (enterprise features require a paid license). Self-host with `docker compose -f docker-compose.prod.yml up`.

**Mapping to a credential_manager**

- **Good fit**: program-facing API keys such as `external_service_api_keys`, plus service account passwords. Scripts read them through the SDK or CLI, with RBAC on permissions and audit on access.
- **Poor fit**:
  - Personal passwords that humans use to log in to websites and hosts belong in Vaultwarden or KeePassXC.
  - Inventory like `hosts` / `host_services` (which machine, which IP, which services) is out of scope for Infisical; that is NetBox territory.

A common combination for a refactor: Infisical for secrets, NetBox (or an existing Streamlit page) for inventory, with the inventory storing only the secret's path in Infisical, never plaintext.

References: [Infisical GitHub README](https://github.com/Infisical/infisical), [What is Infisical?](https://infisical.com/docs/documentation/getting-started/introduction)

## NetBox

NetBox is an inventory system for network and data center infrastructure. Its official positioning is **source of truth**: it records what the infrastructure *should* look like, i.e. the intended state. It does not manage passwords or monitor devices. It was originally written by network engineers at DigitalOcean to automate network provisioning and open-sourced in 2016 under Apache-2.0.

**What it records**

| Area | Content |
|---|---|
| DCIM (data center assets) | Sites, locations, racks (with per-U elevation views), device types, interfaces, cables, power, console connections |
| IPAM (address management) | Prefixes, IPs (assigned to specific interfaces), VRFs, VLANs, ASNs, plus next-available-IP allocation |
| Virtualization | Clusters and virtual machines |
| Other | Provider circuits, VPN, wireless, tenancy, and services (ports and protocols bound to a device or VM) |

**How it fits with other tools**

```
Manual entry / review
      ↓
NetBox (intended state) ──REST/GraphQL API, webhooks──→ Ansible/Salt provisioning
                                                     → Monitoring (Zabbix/Prometheus)
                                                     → DHCP/DNS
```

The docs state explicitly that it does **not** do network monitoring, DNS serving, RADIUS, configuration management, or facilities management. They also advise against auto-importing a device's live operational state; data should be reviewed by a human before entry. What NetBox does is expose that data via API to the tools responsible for those functions.

Other features: custom fields, plugins, Jinja2 config template rendering, custom scripts runnable from the UI, fine-grained per-object permissions, SSO, and a change log that records every change automatically.

Stack: Django, PostgreSQL 15+, Redis, fronted by nginx or Apache plus gunicorn.

**Mapping to a credential_manager**

- The `hosts` table roughly maps to a NetBox Device or VM, plus Interfaces and IPs.
- The `host_services` table maps to a NetBox Service (ports and protocols bound to a device or VM).
- Password fields have no NetBox equivalent. NetBox provides no password storage; a custom field can hold the path to the machine's credential in Infisical or Vaultwarden, storing only a reference, never plaintext.

That said, for a few dozen machines and a handful of services, NetBox may be overkill: it requires running PostgreSQL and Redis, and its data model is designed for racks, cables, and VLANs at scale. In that case, keeping the existing Streamlit page for inventory and moving only the secrets out is likely the better trade-off.

References: [NetBox README](https://github.com/netbox-community/netbox), [Introduction to NetBox](https://netboxlabs.com/docs/netbox/introduction/)