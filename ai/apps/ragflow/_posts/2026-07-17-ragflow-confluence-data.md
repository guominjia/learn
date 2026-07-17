---
layout: post
title: "Connecting RAGFlow to an Internal Confluence Instance"
date: 2026-07-17
categories: [ai, ragflow]
tags: [ragflow, confluence, tls, certifi, enterprise]
---

RAGFlow can import pages from Confluence into a knowledge base. For Confluence hosted inside an enterprise network, configuring only the Confluence access token and wiki base URL is not enough. The connector must be configured as a self-hosted Confluence instance, and the RAGFlow runtime must trust the certificate authority (CA) that issued the internal HTTPS certificate.

This post describes the required configuration for an internal Confluence deployment, including Confluence Data Center and Server installations.

## Required Connector Settings

When creating a Confluence data source in RAGFlow, configure the following fields:

| Setting | Value for internal Confluence |
|---|---|
| Confluence Access Token | A personal access token or other token accepted by the internal Confluence instance |
| Wiki Base URL | The internal Confluence URL, for example `https://confluence.example.corp` |
| Is Cloud | **Cleared** (unchecked) |

The **Is Cloud** option is for Atlassian Confluence Cloud. An internal deployment exposes a different API and authentication behavior, so leaving this option enabled can cause the connector to use the wrong endpoint format or fail during authentication.

The wiki base URL should point to the Confluence root, without REST API paths such as `/rest/api`. For example:

```text
https://confluence.example.corp
```

Do not use:

```text
https://confluence.example.corp/rest/api
```

## Why the Access Token Is Not Enough

An access token authenticates RAGFlow to Confluence after an HTTPS connection has been established. Before that can happen, Python must verify the Confluence server certificate.

Public Confluence sites normally use certificates issued by a public CA, which is already trusted by the certificate bundle supplied by `certifi`. Enterprise Confluence installations often use a certificate issued by an internal PKI. If the internal root CA is not trusted, requests can fail with an error similar to:

```text
SSLCertVerificationError: certificate verify failed: unable to get local issuer certificate
```

## Add the Internal Root Certificate to certifi

Add the PEM-encoded internal root CA certificate to the CA bundle used by `certifi` in the environment that actually runs RAGFlow.

Find the bundle location with:

```bash
python -c "import certifi; print(certifi.where())"
```

Append the internal root certificate to that file. The certificate must be PEM encoded and include the usual delimiters:

```text
-----BEGIN CERTIFICATE-----
...certificate data...
-----END CERTIFICATE-----
```

For a Docker deployment, run this check inside the RAGFlow container that makes the Confluence request. Updating `certifi` on the Docker host does not change the certificate bundle inside the container.

```bash
docker compose exec <ragflow-service> \
	python -c "import certifi; print(certifi.where())"
```

After adding the certificate, restart the affected RAGFlow service so that subsequent connector requests use the updated trust store.

## Make the Certificate Change Persistent

Changes made directly inside a running container are lost when the container is recreated. For a durable Docker deployment, keep the corporate CA certificate in version-controlled deployment configuration or a protected secret store, then copy or mount it into the image and append it to the certifi bundle during image build or container startup.

For example, an image customization can copy the certificate and update the bundle:

```dockerfile
COPY corporate-root-ca.pem /usr/local/share/ca-certificates/corporate-root-ca.pem
RUN python -c "import certifi; bundle = certifi.where(); ca = open('/usr/local/share/ca-certificates/corporate-root-ca.pem').read(); open(bundle, 'a').write('\\n' + ca)"
```

Use the exact path reported by `certifi.where()` in the RAGFlow image. The path varies by Python version, virtual environment, and container image.

## Verification Checklist

Before importing Confluence content, confirm all of the following:

1. The access token is valid and can read the required Confluence spaces and pages.
2. The wiki base URL is the internal Confluence root URL.
3. **Is Cloud** is unchecked.
4. The internal root CA certificate is present in the certifi bundle used by the running RAGFlow service.
5. The RAGFlow service has been restarted after the certificate bundle was changed.

With these settings, RAGFlow can establish a trusted HTTPS connection to the internal Confluence service and then authenticate with the configured access token.
