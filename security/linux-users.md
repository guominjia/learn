## `useradd` VS `adduser`
The commands `useradd` and `adduser` both create user accounts in Linux, but they differ significantly in functionality, interactivity, and default behavior depending on your distribution. Here’s a breakdown:

---

### **1. `useradd` (Low-Level Tool)**  
- **Purpose**: A basic, universal command included in the `shadow-utils` package. Available on **all Linux distributions**.  
- **Behavior**:  
  - Minimal and non-interactive. Requires explicit flags to modify defaults (e.g., `-m` to create a home directory).  
  - Defaults are pulled from `/etc/default/useradd` or `/etc/login.defs`.  
- **Example**:  
  ```bash
  sudo useradd -m -s /bin/bash -c "John Doe" johndoe
  ```  
- **Key Features**:  
  - Does **not create a home directory by default** (use `-m` to force it).  
  - Does **not set a password** (requires `passwd johndoe` afterward).  
  - Full control over UID, GID, supplementary groups, etc., via flags.  

---

### **2. `adduser` (High-Level Wrapper)**  
- **Purpose**: A user-friendly Perl script (on Debian/Ubuntu) that simplifies user creation. **Not available on all distributions**.  
- **Behavior**:  
  - Interactive and guided. Asks for details like password and user info during setup.  
  - Follows defaults from `/etc/adduser.conf`.  
- **Example**:  
  ```bash
  sudo adduser johndoe  # Follow interactive prompts
  ```  
- **Key Features**:  
  - **Automatically creates a home directory** (configurable).  
  - **Sets a password interactively** during creation.  
  - Can copy skeleton files (e.g., `.bashrc` from `/etc/skel`).  
  - May create a group with the same name as the user.  

---

### **Key Differences**  

| Feature                | `useradd`                                  | `adduser` (Debian/Ubuntu)              |
|-------------------------|--------------------------------------------|----------------------------------------|
| **Interactivity**       | Non-interactive (command-line flags only) | Interactive (prompts for details)      |
| **Home Directory**      | Requires `-m` flag                        | Created by default                    |
| **Password**           | No password set initially (use `passwd`)  | Sets password during creation          |
| **Configuration**       | `/etc/default/useradd`, `/etc/login.defs` | `/etc/adduser.conf`                    |
| **Distribution Support** | All Linux distributions                  | Primarily Debian/Ubuntu               |
| **User Experience**     | Manual, technical                          | Beginner-friendly                     |

---

### **Distribution-Specific Notes**  
- **Debian/Ubuntu**:  
  - `adduser` is a separate script with enhanced features.  
  - `useradd` remains the low-level tool for advanced use cases.  
- **Red Hat/CentOS/Fedora**:  
  - `adduser` is a symbolic link to `useradd` and behaves identically.  
  - Use `useradd` directly for full control.  

---

### **When to Use Which?**  
- `useradd`:  
  - Scripting/automation.  
  - Full control over UID, GID, home directory path, etc.  
  - Working across all Linux distributions.  
- `adduser`:  
  - Quick, interactive setup on Debian/Ubuntu.  
  - Simplicity (avoids manual steps like `passwd` or `-m` flags).  

---

### Pro Tip  
For cross-distribution compatibility, stick to `useradd` in scripts. Use `adduser` interactively on Debian/Ubuntu. To see if `adduser` is a wrapper or symlink on your system, run:  
```bash
ls -l $(which adduser)
```

## `useradd` command

The `useradd` command in Linux is used to create new user accounts. Below is a basic syntax and common options:

**Basic Syntax**:
```bash
useradd [options] username
```

**Common Options**:
- `-m` or `--create-home`: Create the user's home directory (e.g., `/home/username`).
- `-d` or `--home-dir`: Specify a custom home directory.
- `-s` or `--shell`: Set the user's login shell (e.g., `/bin/bash`).
- `-g` or `--gid`: Set the primary group (by name or GID).
- `-G` or `--groups`: Add the user to supplementary groups (comma-separated).
- `-u` or `--uid`: Specify a custom UID (User ID).
- `-p` or `--password`: Set a password (note: this is insecure; use `passwd` instead).
- `-e` or `--expiredate`: Set an account expiration date (format: YYYY-MM-DD).
- `-c` or `--comment`: Add a comment (e.g., full name or description).

**Examples**:
1. Create a user with a home directory and default settings:
   ```bash
   sudo useradd -m username
   ```

2. Create a user with a custom home directory and shell:
   ```bash
   sudo useradd -m -d /custom/home/username -s /bin/zsh username
   ```

3. Create a user with a specific UID and primary group:
   ```bash
   sudo useradd -u 1001 -g developers username
   ```

4. Create a user and add to supplementary groups:
   ```bash
   sudo useradd -G sudo,docker,webadmin username
   ```

**Important Notes**:
- After creating the user, set a password using `passwd username`.
- Default configurations (like home directory location) are defined in `/etc/default/useradd` or `/etc/login.defs`.
- Use `userdel` to delete a user account.

Check the manual page for more details:
```bash
man useradd
```