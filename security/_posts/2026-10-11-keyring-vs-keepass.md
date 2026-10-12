---
layout: post
title: "KeePass KDBX vs Python keyring Backends: CryptFile, WinVault, and Secret Service"
date: 2026-10-11
categories: [security, python]
tags: [keepass, kdbx, pykeepass, keyring, keyrings.cryptfile, windows, credential-manager, secret-service, linux, password-manager]
---

## KDBX and KeePass

KDBX is the database format of **KeePass 2.x**, designed by **Dominik Reichl**, the German developer who wrote KeePass, and revised with KeePass releases. The latest version is KDBX 4.1. KeePass 1.x uses the older `.kdb` format.

- The official spec lives on keepass.info: [KDBX File Format Specification](https://keepass.info/help/kb/kdbx.html).

| Format | Introduced | Key changes |
|---|---|---|
| `.kdb` (KDB) | KeePass 1.x; 1.0 released 2003-11-16 ([Wikipedia](https://en.wikipedia.org/wiki/KeePass)) | First-generation format; integrity checked with a SHA-256 hash of the plaintext |
| KDBX 3.x | KeePass 2.x | Binary header wrapping an encrypted XML payload |
| KDBX 4 | KeePass 2.35, [2017-01-09](https://keepass.info/news/n170109_2.35.html) | Argon2 KDF, ChaCha20, inner header, HMAC-SHA-256 header and data authentication (Encrypt-then-MAC) |
| KDBX 4.1 | KeePass 2.48, [2021-05-07](https://keepass.info/news/n210507_2.48.html) | Group tags, named custom icons, and other additions |

## Other Notable Secret Storage Formats

| Format / tool | Author / vendor | Notes |
|---|---|---|
| Password Safe (`.psafe3`) | Designed by Bruce Schneier, now community-maintained | Predates KeePass; uses Twofish |
| pass (password-store) | Jason A. Donenfeld | One GPG-encrypted file per entry, synced with git; Unix philosophy |
| 1Password: Agile Keychain → OPVault → `.1pux` | AgileBits | Closed-source commercial product with publicly documented formats |
| Bitwarden encrypted JSON export | Bitwarden | Open source; server-side sync first |
| Apple Keychain, GNOME Keyring, KWallet | Apple, GNOME, KDE | OS/desktop-level credential stores |
| age, SOPS | age: Filippo Valsorda; SOPS: started at Mozilla | General file/config encryption, common in DevOps secret management |
| OpenPGP, PKCS#12 (`.p12`/`.pfx`) | IETF RFC; RSA PKCS standard | Low-level standard formats for encrypted files and certificates/private keys |

## pykeepass

[libkeepass/pykeepass](https://github.com/libkeepass/pykeepass) reads and writes KDBX from Python; its description says "Python library to interact with keepass databases (supports KDBX3 and KDBX4)". It can read, add, modify, and delete entries, groups, and attachments, then write back with `kp.save()`. The README does not say whether KDBX 4.1 features such as group tags are supported.

## CryptFileKeyring vs KDBX

For the confidentiality of the secret values themselves, the two are comparable: both use Argon2 key derivation plus a modern cipher, so the weak point is the master password. Overall KDBX is stronger because it protects the whole file, while CryptFile protects only each password value.

CryptFile details come from [cryptfile.py](https://raw.githubusercontent.com/frispete/keyrings.cryptfile/master/keyrings/cryptfile/cryptfile.py) and its [README](https://raw.githubusercontent.com/frispete/keyrings.cryptfile/master/README.md).

| Aspect | `CryptFileKeyring` | KDBX 4 |
|---|---|---|
| What is encrypted | Password values only. Service and username are stored in plaintext as INI section/option names, escaped as `_XX` | The whole database: titles, usernames, URLs, notes, attachments |
| KDF | Argon2id with fixed parameters `m=64MB, t=15, p=2`, deriving a 16-byte key | Argon2d or Argon2id, tunable in database settings. KDBX 3.1 uses AES-KDF |
| Cipher | AES-128, GCM by default; CCM/EAX/OCB optional | AES-256 or ChaCha20 |
| Granularity | Each entry encrypted separately with its own salt and nonce, so every `get_password` runs Argon2 (~1 s) | One KDF run when the database is opened |
| Integrity | Per-entry AEAD with service/username as associated data, so tampering with an entry or moving ciphertext to another entry is detected | HMAC over the header and per data block; any modification, deletion, or truncation is detected |
| Whole-file attacks | Deleting an entry or rolling one back to an older ciphertext is **not detected** | Detected |
| Master key | A single password, optionally from the `KEYRING_CRYPTFILE_PASSWORD` environment variable (which leaks easily) | Composite: password + key file + Windows user account |
| Memory protection | Master password held as a plain Python `str` in `keyring_key` | KeePass encrypts sensitive fields in process memory with DPAPI on Windows |
| Known plaintext | A fixed reference value `"password reference value"` verifies the master password; the README admits this is attack surface, mitigated by Argon2 | No equivalent |
| Scrutiny | Small personal project | Public format, large ecosystem, several independent implementations (KeePassXC, KeeWeb, pykeepass) |

With a strong master password, brute-forcing a value costs about the same for both, and AES-128 is not the gap. CryptFile's real weaknesses are:

- **Metadata leak**: anyone with the file sees which accounts you have on which services.
- **No whole-file integrity**: entries can be deleted or rolled back unnoticed.
- **Single factor**: only one password.

## Adding WinVaultKeyring

The three differ mainly in **what key protects the data** and **who can read the plaintext**. Sources: keyring's [Windows.py](https://raw.githubusercontent.com/jaraco/keyring/main/keyring/backends/Windows.py) and Microsoft's [CREDENTIALW](https://learn.microsoft.com/en-us/windows/win32/api/wincred/ns-wincred-credentialw).

| Aspect | `WinVaultKeyring` | `CryptFileKeyring` | KDBX 4 |
|---|---|---|---|
| Storage | Windows Credential Manager, type `CRED_TYPE_GENERIC` | One INI file | One `.kdbx` file |
| Key source | Windows logon credentials via DPAPI; no separate master password | Own master password + Argon2id | Own master password, optionally plus key file and Windows account |
| Other processes of the same user | **Any process can `CredRead` the plaintext while you are logged in, with no prompt and no per-app isolation** | Need the master password | Need the master password |
| Metadata | `cmdkey /list` shows targets and usernames without any password | Service and username in plaintext in the file | Fully encrypted |
| Offline theft (disk or file) | As strong as the Windows account password; in a domain, the domain DPAPI backup key can also decrypt | Master password + Argon2 | Master password + optional key file + Argon2 |
| Integrity | Managed by the OS; users cannot edit the backing store directly | Per-entry AEAD only; deletion or rollback undetected | HMAC over the whole file |
| Roaming / portability | keyring defaults to `CRED_PERSIST_ENTERPRISE`, which roams with a domain roaming profile; cannot be copied to another account | File can be copied | File can be copied; clients on every platform |
| Size limit | 2560 bytes per value; keyring stores UTF-16, so ~1280 characters | No practical limit | No practical limit |
| Accounts per service | One natively; keyring simulates more with compound `username@service` targets | Many | Many |
| Automation | Easiest, no interaction | Master password via `getpass`, environment variable, or code | Master password supplied in code, e.g. via pykeepass |

### By Threat Model

- **Stolen disk, backup, or file**: KDBX > CryptFile > WinVault. With a strong master password, Argon2 keeps offline cracking expensive for the first two. WinVault depends on the Windows password, and domains add the backup-key path.
- **Malware in your user session**: WinVault is weakest, since it is effectively always unlocked while you are logged in. The other two require the master password, but that protection is gone once the program has unlocked or the password sits in an environment variable or script.
- **Metadata privacy**: only KDBX protects it.
- **Convenience**: WinVault, ideal for script tokens on a personal Windows machine.

## Which to Use

- Scripts on a personal Windows machine: WinVault.
- Scripts on headless Linux without Secret Service: CryptFile.
- Syncing across machines, or storing high-value credentials in bulk: KDBX, with pykeepass on the Python side.

## What Is Secret Service?

The [Secret Service API](https://specifications.freedesktop.org/secret-service/latest/) is a freedesktop.org **D-Bus interface specification**, not a program.

- Authors: Stef Walter (GNOME Keyring) and Michael Leupold (KWallet), copyright 2008–2011. It is still marked "0.2 DRAFT".
- It defines how desktop apps store secrets through `org.freedesktop.secrets` on the session bus. Core objects:
  - `Service`: entry point
  - `Collection`: a keyring that can be locked and unlocked
  - `Item`: one secret plus lookup attributes and a label
  - `Session`: negotiates transfer encryption
  - `Prompt`: interactive prompt, e.g. for unlocking
- Implementations: gnome-keyring-daemon, KWallet (recent versions), KeePassXC (with Secret Service integration enabled).
- Python `keyring`'s default Linux backend, `SecretService`, talks to this D-Bus API through the `SecretStorage` library.

### Why Linux Doesn't Ship It

1. **Linux is only a kernel.** Secret Service belongs to the desktop layer, so whether it is present depends on the distribution and desktop environment. GNOME and KDE desktops almost always have it.
2. **It needs a desktop session**, all of:
   - a D-Bus **session bus**
   - a running keyring daemon
   - an unlocked collection, usually unlocked at login by a PAM module (e.g. `pam_gnome_keyring`) using the login password, otherwise via a GUI `Prompt`
3. Headless environments (servers, SSH logins, containers, CI, WSL) typically:
   - have no session bus or no daemon, so `keyring` falls back or fails; or
   - have a daemon but a locked collection and no GUI to prompt, so calls hang or fail.

The Linux kernel has its own keyring (`keyctl`/keyutils), but it holds in-memory keys for the kernel and sessions, lost on reboot, and is unrelated to Secret Service.