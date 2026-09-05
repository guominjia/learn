---
title: "NFS Connection Cheat Sheet"
date: 2026-09-05
tags: [nfs, file-system, linux, command-line]
---

# NFS Connection Cheat Sheet

The `nfs-common` package on Debian and Ubuntu, and the `nfs-utils` package on RHEL and CentOS, provide most NFS client and server utilities.

There are two different things to check:

- **Mounted NFS filesystems:** configured and currently mounted NFS shares.
- **Active network connections:** connections that currently exist between the client and server.

## Client Commands

### List Mounted NFS Filesystems

Use `findmnt` to list NFS and NFSv4 mounts:

```bash
findmnt -t nfs,nfs4
```

Other ways to inspect the mounts are:

```bash
mount | grep -E ' type nfs| type nfs4'
cat /proc/mounts | grep -E ' nfs4? '
```

### Show NFS Mount Details

`nfsstat -m` displays NFS-specific information for each mounted filesystem, including mount options and server details:

```bash
nfsstat -m
```

### Show NFS Client Statistics

Use the following command to display client-side RPC and NFS statistics:

```bash
nfsstat -c
```

### Check Active NFS Connections

NFS commonly uses TCP port `2049`. Use `ss` to check for active connections:

```bash
ss -tnp | grep ':2049'
```

This shows live network connections, not just mount entries. A mounted filesystem can remain configured even when there is no currently active TCP connection.

## Server Commands

### List Exported Directories

Show the server's active exports and their options:

```bash
exportfs -v
```

### List Clients Reported by `mountd`

Use `showmount` to list clients and directories known to the mount daemon:

```bash
showmount -a
showmount -d
```

`showmount` may not show every NFSv4 client because NFSv4 does not always use `mountd` in the same way as NFSv2 and NFSv3.

### Show Server-Side NFS Statistics

```bash
nfsstat -s
```

### Check Active Client Connections

On the server, inspect connections to TCP port `2049`:

```bash
ss -tnp | grep ':2049'
```

Some Linux systems also expose NFSv4 client state through `/proc`:

```bash
ls /proc/fs/nfsd/clients
```

## Quick Reference

```bash
# Client: list mounted NFS filesystems
findmnt -t nfs,nfs4

# Client: show mount options and NFS details
nfsstat -m

# Client or server: show NFS statistics
nfsstat -c       # client statistics
nfsstat -s       # server statistics

# Client or server: show active NFS TCP connections
ss -tnp | grep ':2049'

# Server: show exports
exportfs -v

# Server: show mountd clients and directories
showmount -a
showmount -d
```
