# X Window

## Xvfb Common Use Cases

| Use Case | Traditional X Server | Xvfb (Virtual Framebuffer) |
|----------|---------------------|----------------------------|
| Desktop Interactive | Requires physical display | Runs in memory buffer |
| Screen Recording | Depends on graphics rendering | Renders to memory buffer |
| CI/CD Automation | May be impacted by user activity | Provides isolated environment |

## Install packages

```bash
# Useful utils
sudo apt install x11-apps
sudo apt install x11-utils

# Install imagemagick for screenshot
sudo apt install imagemagick

# Install for screen record
sudo apt install ffmpeg
```

## Commands

### Start Virtual X Server

```bash
Xvfb :99 -screen 0 1280x1024x24 -nolisten tcp -auth /tmp/xvfb-run.1/Xauthority
export XAUTHORITY=/tmp/xvfb-run.1/Xauthority
export DISPLAY=:99
```

### Open X terminal and type hello then return and copy all screen

```bash
xterm -class HostWindow
xdotool search --class "HostWindow" windowsize --sync %@ 800 600 windowfocus %@ type --window %@ "hello" key --window %@ Return
xdotool search --class "HostWindow" key --window %@ "ctrl+a" "ctrl+c"
```

### type hello then return and copy all screen on active window

```bash
xdotool type "hello"
xdotool key Return
xdotool key "ctrl+a" "ctrl+c"
```

### Other useful commands

```bash
xprop | grep WM_CLASS

xwininfo -tree -root

xeyes # Shows eyes following cursor
xclock # Displays clock

xdpyinfo -display $DISPLAY

xdotool search --class "HostWindow" getwindowname
xdotool search --class "HostWindow" getwindowgeometry

# Capture to screenshot.png
import -window root -display $DISPLAY screenshot.png # $DISPLAY env have higher priority than -display

# Install FFmpeg and record
ffmpeg -f x11grab -s 1280x1024 -i :99 -vcodec libx264 out.mp4
```