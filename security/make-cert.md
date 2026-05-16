# Make Certificate

Creating locally-trusted TLS/SSL certificates is essential for developing and testing HTTPS-enabled applications. Manually managing certificates with OpenSSL can be complex and error-prone. **mkcert** simplifies this process by automatically creating and installing a local Certificate Authority (CA) in your system trust store, then issuing certificates signed by that CA.

## Why Local Certificates Matter

When developing web applications that require HTTPS (e.g., OAuth callbacks, service workers, secure cookies), using self-signed certificates triggers browser warnings and breaks certain APIs. `mkcert` solves this by:

- Creating a local CA that your OS and browsers automatically trust
- Generating certificates signed by that CA — no more "Your connection is not private" warnings
- Zero configuration — no need to understand CSR, key formats, or CA bundles

## Install mkcert

### Windows

Download the latest binary from the [releases page](https://github.com/FiloSottile/mkcert/releases) and rename it to `mkcert.exe` for convenience, or install via package managers:

```powershell
# Using Chocolatey
choco install mkcert

# Using Scoop
scoop install mkcert
```

### macOS

```bash
brew install mkcert
brew install nss  # if you use Firefox
```

### Linux

```bash
# Debian/Ubuntu
sudo apt install libnss3-tools
sudo apt install mkcert

# Or install from source
go install filippo.io/mkcert@latest
```

## Set Up the Local CA

Run the install command to create and trust a local CA:

```bash
mkcert -install
```

This creates a root CA key and certificate pair and adds them to the system trust store (and Firefox's NSS store if available). You only need to do this **once** per machine.

To find where the CA files are stored:

```bash
mkcert -CAROOT
```

> **Tip**: If you need other devices on your network to trust your local certs (e.g., a mobile device for testing), copy the `rootCA.pem` file from the CAROOT directory and install it on those devices.

## Generate Certificates

Create a certificate valid for one or more hostnames and IPs:

```bash
mkcert example.local localhost 127.0.0.1 ::1
```

This produces two files in the current directory:
- `example.local+3.pem` — the certificate (public)
- `example.local+3-key.pem` — the private key

### Common Patterns

```bash
# Single domain
mkcert myapp.local

# Wildcard certificate
mkcert "*.myapp.local" myapp.local

# Multiple services
mkcert api.local dashboard.local localhost 127.0.0.1
```

## Use the Certificate

### Node.js (Express / HTTPS)

```javascript
const https = require('https');
const fs = require('fs');
const express = require('express');

const app = express();
app.get('/', (req, res) => res.send('Hello HTTPS!'));

https.createServer({
  key: fs.readFileSync('./localhost-key.pem'),
  cert: fs.readFileSync('./localhost.pem')
}, app).listen(443);
```

### Python (Flask)

```python
from flask import Flask
app = Flask(__name__)

if __name__ == '__main__':
    app.run(
        ssl_context=('./localhost.pem', './localhost-key.pem'),
        host='0.0.0.0',
        port=443
    )
```

### Nginx

```nginx
server {
    listen 443 ssl;
    server_name localhost;

    ssl_certificate     /path/to/localhost.pem;
    ssl_certificate_key /path/to/localhost-key.pem;

    location / {
        proxy_pass http://127.0.0.1:3000;
    }
}
```

## Verify a Certificate with OpenSSL

Use `openssl` to inspect and debug certificates on remote or local servers:

```bash
# Show the full certificate chain
openssl s_client -connect your_web_url:443 -showcerts

# Check certificate expiration
openssl s_client -connect your_web_url:443 2>/dev/null | openssl x509 -noout -dates

# View certificate details (subject, issuer, SANs)
openssl s_client -connect your_web_url:443 2>/dev/null | openssl x509 -noout -text

# Verify a local certificate file
openssl x509 -in ./localhost.pem -noout -text
```

### Key Fields to Check

| Field | Description |
|-------|-------------|
| **Issuer** | Who signed the certificate (your local CA or a public CA) |
| **Subject** | The entity the cert was issued to |
| **Subject Alternative Name (SAN)** | All hostnames/IPs the cert is valid for |
| **Not Before / Not After** | Validity period |

## Uninstall the Local CA

If you no longer need the local CA:

```bash
mkcert -uninstall
```

This removes the CA from the system trust store. The generated certificates will no longer be trusted.

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Browser still shows warning | Restart the browser after `mkcert -install` |
| Firefox doesn't trust the cert | Install `nss` tools: `brew install nss` (macOS) or `apt install libnss3-tools` (Linux) then re-run `mkcert -install` |
| Certificate expired | mkcert certs are valid for ~2 years by default; regenerate if expired |
| Need to share CA with team | **Don't** — the CA private key grants full MITM capability; each developer should run their own `mkcert -install` |

## References

1. <https://github.com/FiloSottile/mkcert> — mkcert repository and releases
2. <https://github.com/FiloSottile/mkcert#supported-root-stores> — supported trust stores
3. <https://www.openssl.org/docs/man3.0/man1/openssl-s_client.html> — OpenSSL s_client documentation