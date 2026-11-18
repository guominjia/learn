# Edk2

## Build Edk2 Emulator

```bash
## Build Emulator
sudo apt update
sudo apt-get install libx11-dev # if no install, it will happen that edk2/EmulatorPkg/Unix/Host/X11GraphicsWindow.c:15:10: fatal error: X11/Xlib.h: No such file or directory
sudo apt-get install libxext-dev # if no install, it will happen that edk2/EmulatorPkg/Unix/Host/X11GraphicsWindow.c:18:10: fatal error: X11/extensions/XShm.h: No such file or directory
git clone https://github.com/tianocore/edk2/
cd edk2 && git submodule update --init
make -C BaseTools/
. edksetup.sh
build -p EmulatorPkg/EmulatorPkg.dsc -a X64 -t GCC5 -b DEBUG

## Run emulator
sudo apt install xvfb xautomation xdotool
cd Build/EmulatorX64/DEBUG_GCC5/X64/ && xvfb-run -a ./Host # xvfb-run for fixing GOP: cannot connect to X server

## Or
Xvfb :99 -screen 0 1024x768x24 &
export DISPLAY=:99
cd Build/EmulatorX64/DEBUG_GCC5/X64/ && ./Host &
xdotool type --delay 100 "hello" # Send command to active window rather than all windows
xdotool key --delay 100 Return
```

## Troubleshooting Matrix

Scenario | Solution
---------|----------
No physical display (server/headless) | Use xvfb-run or Xvfb :1 -screen 0 1024x768x16 & export DISPLAY=:1
Running as root user | Switch to non-root user, or run xhost +SI:localuser:root
Docker/KVM/Virtualization | Add -v /tmp/.X11-unix:/tmp/.X11-unix -e DISPLAY=$DISPLAY flags