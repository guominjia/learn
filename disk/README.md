# Disk

## Understanding Your Disk Stack: From RAID Controller to LVM Logical Volume

When you run `sudo fdisk -l` or `lsblk` on a Linux server, you may encounter device names and disk models that look unfamiliar. This post walks through a real-world example to explain what each layer means — from the hardware RAID controller all the way up to an LVM logical volume.

---

### Step 1: Reading the fdisk Output

```bash
sudo fdisk /dev/sda
```

```
Disk /dev/sda: 13.97 TiB, 15358534615040 bytes, 29997137920 sectors
Disk model: MR9460-16i
Units: sectors of 1 * 512 = 512 bytes
Sector size (logical/physical): 512 bytes / 4096 bytes
I/O size (minimum/optimal): 262144 bytes / 1048576 bytes
Disklabel type: gpt
Disk identifier: 723D7EC3-6663-42B3-B099-4EFD50D34116

Device       Start         End     Sectors Size Type
/dev/sda1     2048     2203647     2201600   1G EFI System
/dev/sda2  2203648     6397951     4194304   2G Linux filesystem
/dev/sda3  6397952 29997135871 29990737920  14T Linux filesystem
```

The **Disk model** field shows `MR9460-16i`. This is not the brand name — it is the model number of a **Broadcom (LSI) MegaRAID SAS 9460-16i** hardware RAID controller.

| Field | Value |
|---|---|
| Brand | Broadcom / LSI |
| Product line | MegaRAID SAS |
| Model | 9460-16i |
| Internal ports | 16 |
| Bus | PCIe |

The key takeaway: `/dev/sda` is **not** a single physical disk. It is a **virtual drive (logical drive)** exposed by the RAID controller to the operating system. Behind the scenes, the controller manages multiple physical disks assembled into a RAID array totaling ~14 TB.

---

### Step 2: Partition Layout

The three partitions follow a standard Ubuntu server layout:

| Device | Size | Type | Purpose |
|---|---|---|---|
| `/dev/sda1` | 1 G | EFI System | UEFI boot partition |
| `/dev/sda2` | 2 G | Linux filesystem | `/boot` |
| `/dev/sda3` | ~14 T | Linux filesystem | LVM Physical Volume |

`/dev/sda3` occupies almost all of the available space and serves as the base for LVM.

---

### Step 3: What Is /dev/mapper/ubuntu--vg-lv--1?

After partitioning, the Ubuntu installer sets up **LVM (Logical Volume Manager)** on top of `/dev/sda3`. LVM adds a flexible abstraction layer between physical partitions and mounted filesystems.

The device `/dev/mapper/ubuntu--vg-lv--1` is an **LVM logical volume** created by the Ubuntu installer.

#### Decoding the Name

```
/dev/mapper/ubuntu--vg-lv--1
                │        │
                │        └── Logical Volume (LV) name: lv-1
                └── Volume Group (VG) name: ubuntu-vg
```

> Note: LVM replaces each `-` in names with `--` inside the `/dev/mapper/` path to avoid ambiguity.

#### Full Stack Diagram

```
Physical Disks (managed by RAID controller)
  └── /dev/sda              ← Virtual drive exposed by MegaRAID 9460-16i
        ├── /dev/sda1       ← EFI System partition
        ├── /dev/sda2       ← /boot
        └── /dev/sda3       ← LVM Physical Volume (PV)
              └── ubuntu-vg ← LVM Volume Group (VG)
                    └── lv-1 ← LVM Logical Volume (LV)
                          └── /dev/mapper/ubuntu--vg-lv--1  ← mounted as /
```

---

### Step 4: Useful Commands

**Inspect the RAID controller and physical disks:**

```bash
# Using storcli (Broadcom's CLI tool)
storcli64 /c0 show
storcli64 /c0/eall/sall show

# Using legacy megacli
megacli -PDList -aALL
```

**Inspect the LVM stack:**

```bash
pvdisplay          # Physical volumes
vgdisplay          # Volume groups
lvdisplay          # Logical volumes
lsblk              # Full block device tree
```

**Expected `lsblk` output:**

```
NAME                      MAJ:MIN RM  SIZE RO TYPE MOUNTPOINT
sda                         8:0    0 13.97T  0 disk
├─sda1                      8:1    0    1G   0 part /boot/efi
├─sda2                      8:2    0    2G   0 part /boot
└─sda3                      8:3    0   14T   0 part
  └─ubuntu--vg-lv--1      253:0    0   14T   0 lvm  /
```

---

### Summary

| Layer | Technology | Device |
|---|---|---|
| Hardware | MegaRAID 9460-16i RAID controller | physical disks |
| Block device | RAID virtual drive | `/dev/sda` |
| Partitioning | GPT | `/dev/sda1~3` |
| Volume management | LVM | `/dev/mapper/ubuntu--vg-lv--1` |
| Filesystem | ext4 / xfs | mounted at `/` |

Understanding this stack is essential when planning disk expansions, replacing failed drives, resizing logical volumes, or diagnosing I/O performance issues on Linux servers.

## Notes

Windows Disk management: `diskpart`, `diskmgmt.msc`