# Share

## Step-by-Step Guide to Set Up an NFS Server on Linux

### Step 1: Install NFS Server Packages
1. Update the package repository:
```bash
sudo apt update
```
2. Install the NFS server package:

For Debian/Ubuntu-based systems:
```bash
sudo apt install nfs-kernel-server
```

For RHEL/CentOS-based systems:
```bash
sudo yum install nfs-utils
```

### Step 2: Create and Configure the Shared Directory
1. Create a directory to share:
```bash
sudo mkdir -p /srv/nfs/shared
```

2. Set the appropriate permissions:
```bash
sudo chown nobody:nogroup /srv/nfs/shared
sudo chmod 755 /srv/nfs/shared
```

### Step 3: Configure NFS Exports
1. Edit the NFS exports file: Open the /etc/exports file in a text editor:
```bash
sudo nano /etc/exports
```

2. Add the directory to be shared: Add the following line to the file to share the directory with specific client IP addresses or subnets:
```plaintext
/srv/nfs/shared 192.168.1.0/24(rw,sync,no_subtree_check)
```
- /srv/nfs/shared: The directory to be shared.
- 192.168.1.0/24: The IP address range of the clients allowed to access the share.
- rw: Read and write permissions.
- sync: Writes changes to disk before responding.
- no_subtree_check: Disables subtree checking for better performance.

Save and close the file.

### Step 4: Export the Shared Directory
1. Export the shared directory:
```bash
sudo exportfs -a
```

2. Restart the NFS server:
```bash
sudo systemctl restart nfs-kernel-server
```

### Step 5: Configure Firewall (if applicable)
1. Allow NFS traffic through the firewall:

For UFW (Uncomplicated Firewall) on Debian/Ubuntu:
```bash
sudo ufw allow from 192.168.1.0/24 to any port nfs
```

For firewalld on RHEL/CentOS:
```bash
sudo firewall-cmd --permanent --add-service=nfs
sudo firewall-cmd --reload
```

### Step 6: Verify NFS Server Configuration
1. Check the NFS exports:
```bash
sudo exportfs -v
```

2. Verify the NFS server status:
```bash
sudo systemctl status nfs-kernel-server
```

## Step-by-Step Guide to Set Up an NFS Client on Linux

### Step 1: Install NFS Client Packages
1. Update the package repository:
```bash
sudo apt update
```

2. Install the NFS client package:

For Debian/Ubuntu-based systems:
```bash
sudo apt install nfs-common
```

For RHEL/CentOS-based systems:
```bash
sudo yum install nfs-utils
```

### Step 2: Create a Mount Point
1. Create a directory to mount the NFS share:
```bash
sudo mkdir -p /mnt/nfs/shared
```

### Step 3: Mount the NFS Share
1. Mount the NFS share:
```bash
sudo mount -o rw 192.168.1.100:/srv/nfs/shared /mnt/nfs/shared
```
- 192.168.1.100: The IP address of the NFS server.
- /srv/nfs/shared: The shared directory on the NFS server.
- /mnt/nfs/shared: The mount point on the client machine.

2. Verify the mount:
```bash
df -h
```

### Step 4: Make the Mount Permanent
1. Edit the /etc/fstab file: Open the /etc/fstab file in a text editor:
```bash
sudo nano /etc/fstab
```

2. Add the NFS
```plaintext
192.168.1.100:/srv/nfs/shared /mnt/nfs/shared nfs defaults,rw 0 0
```