---
layout: post
title: "mkcert vs. OpenSSL for Local TLS Certificates"
date: 2026-08-06
categories: [security, tls]
tags: [tls, ssl, certificate, mkcert, openssl, local-development, ca]
---

Local HTTPS development needs more than encryption. The client must also trust the certificate presented by the local server, and the certificate must identify the hostname or IP address used to reach it.

`mkcert` and OpenSSL can both create local TLS certificates, but they solve different problems:

- Use `mkcert` when a developer needs a browser- and system-trusted certificate quickly.
- Use OpenSSL when the certificate profile, CA hierarchy, key handling, or signing workflow must be controlled explicitly.

Neither tool turns a locally issued certificate into a publicly trusted certificate. A certificate is trusted only by clients that trust the issuing CA.

## The Trust Model

A server certificate is normally validated through a chain:

```text
server certificate -> local CA certificate -> client trust store
```

A self-signed server certificate has no separate trusted issuer. It can work for a test only after every client is configured to trust that exact certificate. A local CA is usually more practical: install the CA certificate once, then use its private key to issue certificates for multiple local services.

The CA private key is sensitive. Anyone who obtains it can issue certificates that trusted clients may accept. Keep it out of source control, backups shared with others, and ordinary application directories.

## When to Choose Each Tool

| Requirement | `mkcert` | OpenSSL |
|---|---|---|
| Create a trusted certificate for `localhost` quickly | Best fit | Possible, but more setup |
| Install a local root CA automatically | Yes, with `mkcert -install` | Manual OS or browser trust-store work |
| Control X.509 extensions and signing policy | Limited, opinionated workflow | Full control |
| Create CSRs for an internal or public CA | Not its primary use | Standard workflow |
| Run a reusable internal CA process | Not intended for this | Possible, but needs operational controls |
| Use in production or distribute to end users | No | Only as part of a properly operated PKI |

## Fast Local HTTPS with `mkcert`

`mkcert` creates a local CA and installs its certificate in supported trust stores. Then it issues leaf certificates signed by that CA. On Windows, install the executable with a package manager, then create the local CA:

```powershell
choco install mkcert
mkcert -install
```

Use Scoop instead if that is your standard package manager:

```powershell
scoop bucket add extras
scoop install mkcert
mkcert -install
```

Generate a certificate for the actual names used during development. Put options before the names when choosing output paths:

```powershell
mkcert -cert-file localhost.pem -key-file localhost-key.pem localhost 127.0.0.1 ::1
```

The resulting certificate includes the supplied DNS names and IP addresses. Configure the local HTTPS server with `localhost.pem` and `localhost-key.pem`.

Use `mkcert -CAROOT` to print the directory holding the local CA certificate and key. The CA certificate may be installed on another controlled development device when needed, but never copy or share `rootCA-key.pem`. The mkcert project explicitly describes the tool as development-only, not a solution for production or end-user systems.

## Why OpenSSL Needs More Steps

OpenSSL is a cryptographic toolkit and X.509 command-line interface, not a local-development trust-store installer. It can create a root CA, a certificate signing request (CSR), and a signed server certificate, but it does not automatically make clients trust the CA.

For a manually managed local CA, the critical constraints are:

- The CA certificate must have `basicConstraints = critical, CA:TRUE` and a `keyUsage` that includes `keyCertSign`.
- The server certificate must be an end-entity certificate, not a CA certificate.
- The server certificate should specify `extendedKeyUsage = serverAuth`.
- Every hostname and IP address used by clients must appear in `subjectAltName` (SAN).

The Common Name alone is not a replacement for SAN in modern TLS hostname verification.

## A Minimal OpenSSL Local CA

The following PowerShell-friendly commands show the core artifacts. They are suitable for an isolated development machine, not a production CA service.

First, create a locally protected working directory and generate a self-signed root CA. The root key is encrypted; choose a strong passphrase when prompted.

```powershell
New-Item -ItemType Directory -Force .\local-ca | Out-Null
Set-Location .\local-ca

openssl req -x509 -newkey rsa:4096 -keyout root-ca-key.pem -out root-ca.pem -days 3650 -sha256 `
	-subj "/CN=Local Development CA" `
	-addext "basicConstraints=critical,CA:TRUE" `
	-addext "keyUsage=critical,keyCertSign,cRLSign" `
	-addext "subjectKeyIdentifier=hash"
```

Trust `root-ca.pem` only in the user or test environment that needs it. Importing a root CA changes the set of identities that the client accepts, so do not install an unreviewed CA certificate.

Next, create a server key and CSR. `-noenc` is appropriate here only when the server process must start unattended and the key file is protected by filesystem permissions:

```powershell
openssl req -new -newkey rsa:2048 -noenc -keyout localhost-key.pem -out localhost.csr `
	-subj "/CN=localhost" `
	-addext "subjectAltName=DNS:localhost,IP:127.0.0.1,IP:::1"
```

Create an extension file for the server certificate:

```ini
# server-ext.cnf
basicConstraints = critical,CA:FALSE
keyUsage = critical,digitalSignature,keyEncipherment
extendedKeyUsage = serverAuth
subjectAltName = DNS:localhost,IP:127.0.0.1,IP:::1
subjectKeyIdentifier = hash
authorityKeyIdentifier = keyid,issuer
```

Sign the CSR using the local CA key:

```powershell
openssl x509 -req -in localhost.csr -CA root-ca.pem -CAkey root-ca-key.pem `
	-CAcreateserial -out localhost.pem -days 397 -sha256 -extfile server-ext.cnf
```

The `-CAcreateserial` option creates `root-ca.srl` on the first signing operation. Preserve it with the CA state; do not reset it between certificates.

## Inspect Before Debugging Clients

Inspect the certificate before configuring an application. The following command prints the issuer, subject, validity interval, and extensions including SAN:

```powershell
openssl x509 -in localhost.pem -noout -text
```

To validate the chain against the local CA file:

```powershell
openssl verify -CAfile root-ca.pem localhost.pem
```

This proves that the certificate chains to `root-ca.pem`; it does not prove that a browser, Python library, container, or other runtime trusts that CA. Check the specific client's trust source. For example, some Python clients use the operating-system trust store, while others default to a bundled CA file.

## Common Failure Modes

| Symptom | Likely cause | Corrective action |
|---|---|---|
| Browser or client reports an untrusted issuer | The CA certificate is not in that client's trust store | Install the intended CA certificate for that client, or use the client's explicit CA configuration. |
| Hostname mismatch | The requested name or IP is absent from SAN | Reissue the server certificate with every required `DNS:` and `IP:` SAN. |
| Server certificate is treated as a CA | Incorrect `basicConstraints` or key usage | Reissue it with `CA:FALSE`, leaf key usage, and `serverAuth`. |
| Certificate works in a browser but not an HTTP client | The client uses a separate CA bundle | Configure that client with the system trust store, an SSL context, or an explicit CA bundle. |
| Another machine trusts unexpected certificates | The local CA private key was copied or exposed | Remove the CA from trust stores, generate a new CA, and reissue certificates. |

## Takeaway

Use `mkcert` for ordinary local development because it automates the difficult but necessary trust-store step. Use OpenSSL when you need to understand or control the certificate lifecycle, but treat the CA key, trust-store installation, X.509 extensions, and serial-number state as part of the design.

For either approach, keep certificate verification enabled and issue certificates with correct SAN values. Do not work around a trust failure by disabling TLS verification.

## References

- [mkcert repository and usage documentation](https://github.com/FiloSottile/mkcert): local CA creation, trust-store installation, output options, and key-handling warnings.
- [OpenSSL `req` documentation](https://docs.openssl.org/master/man1/openssl-req/): CSR creation, self-signed certificates, CA signing, and `-addext` support.
- [OpenSSL X.509v3 extension documentation](https://docs.openssl.org/master/man5/x509v3_config/): CA constraints, key usages, extended key usages, and SAN syntax.
