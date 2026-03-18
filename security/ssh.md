# SSH (Secure Shell)

SSH (Secure Shell) is a cryptographic network protocol used for secure communication between a client and a remote server. It is the standard way to manage servers, transfer files, and tunnel traffic over untrusted networks.

---

## How SSH Works

SSH uses a **client-server** model. The SSH daemon (`sshd`) runs on the server and listens on port **22** by default. The client initiates a connection and authenticates using one of the supported methods.

### Connection Flow

1. **TCP Handshake** – Client connects to the server on port 22.
2. **Protocol Negotiation** – Both sides agree on SSH protocol version and supported algorithms.
3. **Key Exchange** – A shared session key is established using algorithms like Diffie-Hellman. This key encrypts all subsequent communication.
4. **Server Authentication** – The client verifies the server's identity using the server's host key (stored in `~/.ssh/known_hosts`).
5. **User Authentication** – The user proves their identity via password, public key, or other methods.
6. **Session** – An encrypted channel is established for shell access, file transfer, or port forwarding.

---

## Authentication Methods

### 1. Password Authentication

The simplest method. The client sends a password over the encrypted channel.

```bash
ssh user@192.168.1.100
# Enter password when prompted
```

> **Drawback**: Vulnerable to brute-force attacks. Disable it on production servers.

### 2. Public Key Authentication (Recommended)

Uses an **asymmetric key pair**: a private key (kept secret on the client) and a public key (placed on the server).

#### Generate a Key Pair

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

- `-t ed25519` – Uses the Ed25519 algorithm (modern, fast, secure). Alternatives: `rsa`, `ecdsa`.
- `-C` – Adds a comment (usually your email) to identify the key.

This creates two files:

| File | Description |
|------|-------------|
| `~/.ssh/id_ed25519` | Private key (never share this) |
| `~/.ssh/id_ed25519.pub` | Public key (place on remote servers) |

#### Copy the Public Key to the Server

```bash
ssh-copy-id user@192.168.1.100
```

This appends your public key to the server's `~/.ssh/authorized_keys` file. You can also do it manually:

```bash
cat ~/.ssh/id_ed25519.pub | ssh user@192.168.1.100 "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

#### `authorized_keys`

The `~/.ssh/authorized_keys` file on the server contains one public key per line. When a client attempts key-based authentication, the server checks if the client's public key matches an entry in this file.

```
ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIExample... your_email@example.com
ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAABgQ... another_key@host
```

**Permissions matter!** SSH will refuse key authentication if permissions are too open:

```bash
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

---

## SSH Config File (`~/.ssh/config`)

Instead of typing long commands, define host aliases in `~/.ssh/config`:

```
Host myserver
    HostName 192.168.1.100
    User deploy
    Port 2222
    IdentityFile ~/.ssh/id_ed25519

Host github
    HostName github.com
    User git
    IdentityFile ~/.ssh/id_ed25519_github
```

Now you can simply run:

```bash
ssh myserver        # connects to deploy@192.168.1.100:2222
git clone github:user/repo.git   # uses the github SSH config
```

### Useful Config Options

| Option | Description |
|--------|-------------|
| `HostName` | The actual IP or domain of the server |
| `User` | Default username for the connection |
| `Port` | SSH port (default: 22) |
| `IdentityFile` | Path to the private key to use |
| `ServerAliveInterval` | Seconds between keepalive messages (prevents timeout) |
| `ServerAliveCountMax` | Max keepalive messages before disconnecting |
| `ProxyJump` | Specify a jump host (bastion) to connect through |
| `ForwardAgent` | Forward the local SSH agent to the remote host |

---

## SSH Server Configuration (`/etc/ssh/sshd_config`)

Key settings for hardening your SSH server:

```bash
# Change default port
Port 2222

# Disable root login
PermitRootLogin no

# Disable password authentication (use keys only)
PasswordAuthentication no

# Allow only specific users
AllowUsers deploy admin

# Limit authentication attempts
MaxAuthTries 3
```

After editing, restart the SSH service:

```bash
sudo systemctl restart sshd
```

---

## SSH Agent

The SSH agent holds your private keys in memory so you don't need to enter the passphrase every time.

```bash
# Start the agent
eval "$(ssh-agent -s)"

# Add your key
ssh-add ~/.ssh/id_ed25519
```

### Agent Forwarding

Agent forwarding lets you use your local SSH keys on a remote server without copying the private key.

```bash
ssh -A user@bastion-server
# From bastion, you can now SSH to other servers using your local keys
```

> **Security Warning**: Only use agent forwarding with trusted servers. A compromised server could use your forwarded agent to authenticate as you.

---

## Common SSH Operations

### SCP – Secure Copy

```bash
# Copy file to remote server
scp file.txt user@server:/path/to/destination/

# Copy from remote to local
scp user@server:/path/to/file.txt ./local_dir/

# Copy directory recursively
scp -r ./local_dir user@server:/remote/path/
```

### SFTP – SSH File Transfer Protocol

```bash
sftp user@server
sftp> put local_file.txt /remote/path/
sftp> get /remote/file.txt ./local/
sftp> ls
sftp> exit
```

### SSH Tunneling (Port Forwarding)

#### Local Port Forwarding

Forward a local port to a remote service:

```bash
ssh -L 8080:localhost:3000 user@server
```

Now `localhost:8080` on your machine maps to `localhost:3000` on the server. Useful for accessing remote databases or web apps.

#### Remote Port Forwarding

Expose a local service to the remote server:

```bash
ssh -R 9090:localhost:8080 user@server
```

Port `9090` on the server now forwards to `localhost:8080` on your machine.

#### Dynamic Port Forwarding (SOCKS Proxy)

```bash
ssh -D 1080 user@server
```

Creates a SOCKS5 proxy on `localhost:1080`. Configure your browser to use it for proxied browsing.

---

## Jump Hosts (Bastion Servers)

When a target server is only accessible through a bastion host:

```bash
# Using ProxyJump (recommended, OpenSSH 7.3+)
ssh -J user@bastion user@target

# Or in ~/.ssh/config
Host target
    HostName 10.0.0.5
    User deploy
    ProxyJump user@bastion.example.com
```

---

## `known_hosts`

The `~/.ssh/known_hosts` file stores fingerprints of servers you've connected to before. SSH warns you if a server's fingerprint changes (potential man-in-the-middle attack).

```bash
# Remove a stale entry
ssh-keygen -R 192.168.1.100

# Show server fingerprint
ssh-keygen -lf /etc/ssh/ssh_host_ed25519_key.pub
```

---

## Troubleshooting

### Debug Connection Issues

```bash
ssh -vvv user@server
```

The `-v` flag enables verbose output. Use up to three `v`s for increasing detail.

### Common Permission Errors

```
Permissions 0755 for '/home/user/.ssh/id_ed25519' are too open.
```

Fix:

```bash
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### Connection Timeout

Add keepalive settings in `~/.ssh/config`:

```
Host *
    ServerAliveInterval 60
    ServerAliveCountMax 3
```

### "Host key verification failed"

This means the server fingerprint has changed since your last connection:

```bash
# Remove the old fingerprint
ssh-keygen -R hostname_or_ip
# Reconnect and accept the new fingerprint
ssh user@server
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `ssh user@host` | Connect to a remote host |
| `ssh -p 2222 user@host` | Connect on a custom port |
| `ssh-keygen -t ed25519` | Generate Ed25519 key pair |
| `ssh-copy-id user@host` | Copy public key to server |
| `ssh-add ~/.ssh/key` | Add key to SSH agent |
| `ssh -L 8080:localhost:80 user@host` | Local port forwarding |
| `ssh -R 9090:localhost:80 user@host` | Remote port forwarding |
| `ssh -D 1080 user@host` | SOCKS proxy |
| `ssh -J bastion user@target` | Jump through bastion host |
| `ssh -vvv user@host` | Verbose debug output |
| `scp file user@host:/path` | Copy file to remote |
| `sftp user@host` | Interactive file transfer |