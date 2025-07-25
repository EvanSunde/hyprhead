# Head Tracker for Hyprland
Very lightweight and simple head tracker for Hyprland to change focus of monitors based on where your head is.

I made this because I was tired of using my mouse to change focus of monitors.

There is also addtional functions to scroll up and down with your fingers and left click with your index finger.

This project uses MediaPipe to track hand gestures and control your mouse cursor without using a physical mouse. It's designed to work with Hyprland on Linux.

## Features

- Control mouse cursor with hand movements
- Left-click by pressing down your index finger
- Scroll up and down by moving your index and middle fingers up and down
- Smooth cursor movement with adjustable sensitivity
- Debug mode to visualize hand tracking
- Works with multiple monitors

## Requirements

- Python 3.10+
- OpenCV
- MediaPipe
- ydotool
- A webcam

## Installation

1. Clone this repository:
```bash
git clone https://github.com/evansunde/hyprhead.git
cd hyprhead
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Install system dependencies for ydotool:
```bash
# For Arch-based distributions
sudo pacman -S ydotool

# For Debian-based distributions
sudo apt-get install ydotool
```

## Usage

Run the hand tracker:

```bash
python head_tracker.py
```

### Command-line options:

- `--no-debug`: Disable the debug display to save CPU usage
- `--center left right`: Center the head tracker on the screen, left, or right

Example:
```bash
python head_tracker.py --center right --no-debug
```

## Gestures

- **Move cursor**: Move your hand in the air with palm facing the camera
- **Left-click**: Bend your index finger down
- **Scroll up**: Bend your index and middle fingers up
- **Scroll down**: Bend your index and middle fingers down
