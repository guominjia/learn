# Secure Secret Input: How to Read Sensitive Data Without Exposing It

When writing scripts or applications that require passwords, API keys, or other sensitive credentials, one of the most common mistakes is reading input in plain text — where the value is visible on screen, stored in shell history, or leaked into logs. This post covers the most reliable cross-platform techniques to handle sensitive input securely.

---

## The Core Problem

A naive approach like this is dangerous:

```bash
# BAD: visible in terminal history and on screen
echo "Enter API key: "
read API_KEY
export API_KEY
```

Anyone looking over your shoulder, or running `history`, can see the value. The solution is **silent input** — reading user input without echoing characters to the terminal.

---

## Linux / macOS: `read -s` (Bash Built-in)

The simplest and most portable approach on any Unix-like system is the shell built-in `read` with the `-s` (silent) flag.

```bash
read -s -p "Enter your secret key: " MY_SECRET
echo   # move to a new line after input
export MY_SECRET
```

**How it works:**
- `-s` suppresses terminal echo — nothing is displayed as you type.
- `-p "..."` prints an inline prompt before reading.
- `echo` after the read is necessary to advance the cursor to a new line (since pressing Enter does not print a newline when echo is suppressed).
- `export` makes the variable available to child processes.

**One-liner:**
```bash
read -s -p "Enter secret: " MY_SECRET; echo; export MY_SECRET
```

**Feeding a secret to another program:**
```bash
read -s -p "Enter secret input: " SECRET
echo "$SECRET" | openssl enc -aes-256-cbc -a -salt -out encrypted.dat
```

> **Note:** `read -s` only protects the value during input. It still lives in memory and in the current shell''s environment. Be cautious about how it is used downstream.

### Using `gpg` for symmetric encryption

```bash
gpg --symmetric --output encrypted.gpg
```

Type the content to encrypt, press **Ctrl+D** to finish, then enter an encryption passphrase. The result is written to `encrypted.gpg`.

---

## Python: `getpass.getpass`

Python''s standard library includes `getpass`, a cross-platform module that mirrors the behavior of `read -s`. It disables terminal echo for the duration of input and works on Linux, macOS, and Windows.

```python
import getpass

secret = getpass.getpass(prompt="Enter your API key: ")
# secret is a plain string; use it however you need
print(f"Key length: {len(secret)}")
```

**Key properties:**
- No characters are echoed to the terminal while typing.
- Works inside scripts, CLI tools, and interactive sessions.
- On systems without a controlling terminal (e.g., some CI environments), it falls back to reading from `sys.stdin` with a warning.

**Storing as an environment variable from Python:**
```python
import getpass
import os

os.environ["MY_SECRET"] = getpass.getpass("Enter secret: ")
```

**Difference from `input()`:**

| Method | Echo | Use case |
|---|---|---|
| `input("prompt: ")` | Yes (visible) | Non-sensitive input |
| `getpass.getpass("prompt: ")` | No (silent) | Passwords, tokens, keys |

**Example — prompting for a database password before connecting:**
```python
import getpass
import psycopg2

password = getpass.getpass("DB password: ")
conn = psycopg2.connect(host="localhost", user="admin", password=password)
```

---

## Windows: PowerShell `Read-Host -AsSecureString`

On Windows, the equivalent of `read -s` in PowerShell is `Read-Host` with the `-AsSecureString` flag.

```powershell
$SecurePass = Read-Host -AsSecureString "Enter password"
```

To convert back to a plain string (use with caution):
```powershell
$PlainPass = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
    [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($SecurePass)
)
```

**In a `.bat` script via PowerShell:**
```batch
@echo off
for /f "delims=" %%p in ('powershell -Command "$p = Read-Host -AsSecureString ''Enter Password''; [System.Runtime.InteropServices.Marshal]::PtrToStringAuto([System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($p))"') do set "password=%%p"
echo Obtained password successfully.
```

---

## Summary: Choosing the Right Method

| Platform | Method | Silent Input | Built-in |
|---|---|---|---|
| Linux / macOS (shell) | `read -s` | Yes | Yes |
| Python (cross-platform) | `getpass.getpass` | Yes | Yes (stdlib) |
| Windows (PowerShell) | `Read-Host -AsSecureString` | Yes | Yes |

**Recommendations:**
- **For shell scripts on Linux/macOS:** always use `read -s` over plain `read`.
- **For Python tools:** always use `getpass.getpass` over `input()` for any sensitive value.
- **Avoid** passing secrets directly on the command line (e.g., `--password=secret`) — they appear in process lists (`ps aux`) and shell history.
- **Prefer environment variables or secret managers** (e.g., HashiCorp Vault, AWS Secrets Manager) over interactive prompts in automated pipelines.