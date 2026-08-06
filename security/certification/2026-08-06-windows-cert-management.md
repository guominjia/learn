---
layout: post
title: "Managing Windows Certificates with PowerShell and MMC"
date: 2026-08-06
categories: [security, windows]
tags: [windows, powershell, certificate, x509, mmc, pki]
---

Windows keeps X.509 certificates in certificate stores rather than in one universal PEM bundle. The two scopes that matter most are **Current User**, which is visible only to the signed-in user, and **Local Machine**, which is available to all users on the computer.

Use PowerShell when the task needs repeatable queries, filtering, or automation. Use the Certificates MMC interface when you need to inspect fields visually or use the import and export wizards. Both approaches operate on the Windows certificate stores, so choosing the correct scope is more important than choosing the interface.

## Know the Store Scope

The PowerShell Certificate provider exposes Windows certificate stores through the `Cert:` drive:

```text
Cert:\CurrentUser\...
Cert:\LocalMachine\...
```

Installing a trusted CA in `CurrentUser\Root` affects applications running as that user. Installing it in `LocalMachine\Root` has a wider effect and normally requires elevation. Do not place a certificate in a trusted-root store merely to silence a validation error; first confirm that it is the intended CA and that its private key is protected.

Common store names include:

| Store | Typical purpose |
|---|---|
| `My` | Personal certificates, often with private keys |
| `Root` | Trusted root certification authorities |
| `CA` | Intermediate certification authorities |
| `TrustedPublisher` | Publishers trusted for signed code or content |

## Query Certificates with PowerShell

The Certificate provider is available on Windows PowerShell and PowerShell on Windows. `Get-ChildItem` can enumerate certificates and stores, and certificate objects expose properties such as `Subject`, `Issuer`, `Thumbprint`, `NotBefore`, and `NotAfter`.

List the stores in both scopes:

```powershell
Get-ChildItem Cert:\CurrentUser
Get-ChildItem Cert:\LocalMachine
```

List trusted root certificates for the current user with the most useful identifying fields:

```powershell
Get-ChildItem Cert:\CurrentUser\Root |
	Select-Object Subject, Issuer, Thumbprint, NotBefore, NotAfter
```

Find certificates that expire in the next 30 days. This query is read-only and is suitable for an inventory or monitoring script:

```powershell
Get-ChildItem Cert:\LocalMachine\WebHosting -ExpiringInDays 30 |
	Select-Object Subject, Thumbprint, NotAfter
```

Find server-authentication certificates in the common machine stores:

```powershell
$query = @{
	Path = 'Cert:\LocalMachine\My', 'Cert:\LocalMachine\WebHosting'
	SSLServerAuthentication = $true
}

Get-ChildItem @query |
	Select-Object Subject, DnsNameList, Thumbprint, NotAfter
```

Inspect one certificate by its thumbprint. A thumbprint identifies a certificate in its store, so it is safer than selecting a certificate by a partial display name:

```powershell
$certificate = Get-Item 'Cert:\LocalMachine\My\0123456789ABCDEF0123456789ABCDEF01234567'
$certificate | Format-List Subject, Issuer, Thumbprint, DnsNameList, EnhancedKeyUsageList, NotAfter
```

Replace the example thumbprint with the value returned by a query. The `DnsNameList` property is useful for checking subject alternative names, while `EnhancedKeyUsageList` helps distinguish server, client, and code-signing certificates.

## Manage Certificates Through the GUI

For a quick graphical view, open the appropriate Certificates MMC console from the Run dialog or PowerShell:

```powershell
certmgr.msc  # Certificates - Current User
certlm.msc   # Certificates - Local Computer
```

Use `certmgr.msc` for the current user's stores. Use `certlm.msc` when the certificate must be available system-wide; Windows may prompt for administrative approval before changes are allowed.

The portable alternative is to open `mmc`, add the **Certificates** snap-in, then choose **My user account** or **Computer account**. From the selected store, the context menu provides **All Tasks** actions such as Import and Export. Before importing, confirm the target scope and store, especially for root CAs and PFX files containing private keys.

## Choose the Right Workflow

| Task | Recommended interface |
|---|---|
| Find certificates by expiration, DNS name, or EKU | PowerShell |
| Export a certificate chain or inspect extensions interactively | Certificates MMC |
| Import a certificate during a one-off setup | Certificates MMC wizard |
| Produce an auditable inventory | PowerShell script |
| Remove or replace a trusted root | PowerShell or MMC, only after verifying scope and thumbprint |

PowerShell can also open a selected certificate directly in the Certificates MMC snap-in:

```powershell
Invoke-Item 'Cert:\CurrentUser\My\0123456789ABCDEF0123456789ABCDEF01234567'
```

## Safety Notes

- A certificate in a `Root` store is a trust anchor. Adding one can cause applications to trust certificates issued by that CA.
- A PFX or another certificate with a private key is sensitive. Export it only when necessary, protect it with a strong password, and avoid copying it into source-control directories.
- `Remove-Item` can delete a certificate, and `-DeleteKey` can delete its associated private key. Begin with inventory commands and use `-WhatIf` where supported before a destructive change.
- Some applications use their own CA bundle instead of the Windows certificate store. A certificate that appears in MMC is not automatically trusted by every application.

## References

- [PowerShell Certificate provider](https://learn.microsoft.com/en-us/powershell/module/microsoft.powershell.security/about/about_certificate_provider?view=powershell-7.5): documents the `Cert:` drive, store locations, certificate filtering, MMC integration, and deletion behavior.
- [Windows certificate stores](https://learn.microsoft.com/en-us/windows-hardware/drivers/install/certificate-stores): documents the Current User and Local Machine scopes and the Certificates MMC snap-in workflow for viewing, importing, exporting, and managing certificates.
