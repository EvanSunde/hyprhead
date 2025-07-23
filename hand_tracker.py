#!/usr/bin/env python3
"""
Simple Hand Tracking Mouse Control for Hyprland/Wayland

This script uses MediaPipe to track index finger movement from a webcam 
and translates it into mouse cursor movements using ydotool.

Dependencies:
- Python 3
- OpenCV: pip install opencv-python
- MediaPipe: pip install mediapipe
- ydotool: A command-line automation tool for Wayland.
  Installation on Arch Linux: `sudo pacman -S ydotool`

Setup:
1. Install the dependencies listed above.
2. Ensure `ydotoold` is running: `systemctl --user start ydotoold.service`
3. Run the script: `python hand_tracker.py`
4. To stop, press 'q' in the OpenCV window.
"""

import cv2
import mediapipe as mp
import subprocess
import numpy as np
import time
import re
import collections

# --- Configuration ---
# Screen resolution
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1080

# Smoothing factor for mouse movement (0.0 - 1.0). Higher is smoother.
SMOOTHING = 0.8  # Increased for much smoother movement

# Movement scaling factor
MOVEMENT_SCALE_X = 1.2
MOVEMENT_SCALE_Y = 1.2

# Smoothing window size (number of frames to average)
SMOOTHING_WINDOW_SIZE = 15

# Maximum allowed movement between frames (in pixels)
# This prevents large jumps in cursor position
MAX_MOVEMENT_PER_FRAME = 20

# Debug mode - print coordinates to console
DEBUG = True

# Frame rate control - limit processing to improve smoothness
PROCESS_EVERY_N_FRAMES = 2

def get_screen_resolution():
    """Gets screen resolution using hyprctl."""
    try:
        output = subprocess.check_output(['hyprctl', 'monitors']).decode('utf-8')
        match = re.search(r'(\d+)x(\d+)@', output)
        if match:
            return int(match.group(1)), int(match.group(2))
    except (FileNotFoundError, subprocess.CalledProcessError) as e:
        print(f"Could not get screen resolution via hyprctl: {e}")
        print(f"Falling back to default: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    return SCREEN_WIDTH, SCREEN_HEIGHT

def move_mouse(x, y):
    """Move the mouse using ydotool with absolute positioning."""
    try:
        cmd = ['ydotool', 'mousemove', '-a', str(int(x)), str(int(y))]
        if DEBUG:
            print(f"Running: {' '.join(cmd)}")
        subprocess.run(cmd, check=False)
        return True
    except Exception as e:
        print(f"Error moving mouse: {e}")
        return False

class LowPassFilter:
    """A simple low-pass filter to smooth movement."""
    def __init__(self, window_size=10):
        self.x_points = collections.deque(maxlen=window_size)
        self.y_points = collections.deque(maxlen=window_size)
        self.window_size = window_size
        
    def update(self, x, y):
        """Add a new point and return the filtered value."""
        self.x_points.append(x)
        self.y_points.append(y)
        
        if len(self.x_points) < 3:  # Need at least 3 points for good filtering
            return x, y
            
        # Calculate weighted average with more weight to recent values
        x_weights = np.linspace(0.5, 1.0, len(self.x_points))
        y_weights = np.linspace(0.5, 1.0, len(self.y_points))
        
        x_filtered = np.average(self.x_points, weights=x_weights)
        y_filtered = np.average(self.y_points, weights=y_weights)
        
        return x_filtered, y_filtered
        
    def reset(self):
        """Reset the filter."""
        self.x_points.clear()
        self.y_points.clear()

def main():
    screen_w, screen_h = get_screen_resolution()
    print(f"Using screen resolution: {screen_w}x{screen_h}")

    # MediaPipe Hands setup
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils

    # Webcam setup
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam.")
        return

    # Set lower resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Camera resolution: {cam_w}x{cam_h}")

    # Mouse movement variables
    plocx, plocy = screen_w // 2, screen_h // 2  # Start in the middle of the screen
    clocx, clocy = screen_w // 2, screen_h // 2
    
    # Initialize mouse position
    move_mouse(clocx, clocy)
    
    # Initialize low-pass filter
    low_pass_filter = LowPassFilter(window_size=SMOOTHING_WINDOW_SIZE)
    
    # FPS variables
    prev_frame_time = 0
    new_frame_time = 0
    
    # Frame processing counter
    process_frame_counter = 0
    
    # Last valid hand position
    last_valid_hand_pos = None
    
    # Flag to track if we've lost the hand
    hand_lost_counter = 0

    try:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Process only every Nth frame for better stability
            process_this_frame = process_frame_counter % PROCESS_EVERY_N_FRAMES == 0
            process_frame_counter += 1
            
            if not process_this_frame:
                # Skip processing this frame
                continue
                
            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time) if prev_frame_time > 0 else 0
            prev_frame_time = new_frame_time
            
            # Convert the BGR image to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Process the image and find hands
            results = hands.process(rgb_image)
            
            # Display FPS
            cv2.putText(image, f"FPS: {fps:.1f}", (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Check if we detected a hand
            hand_detected = results.multi_hand_landmarks is not None
            
            if not hand_detected:
                hand_lost_counter += 1
                if hand_lost_counter > 10:  # If hand lost for too long, reset filter
                    low_pass_filter.reset()
                    
                # Display status
                cv2.putText(image, "Hand not detected", (10, 70), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                hand_lost_counter = 0
                
                # Draw the hand annotations on the image
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Get coordinates of index finger tip (landmark 8)
                    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                    ix, iy = int(index_tip.x * cam_w), int(index_tip.y * cam_h)
                    
                    # Store as last valid position
                    last_valid_hand_pos = (ix, iy)
                    
                    # Convert to screen coordinates with scaling
                    screen_x = np.interp(ix, (0, cam_w), (0, screen_w * MOVEMENT_SCALE_X))
                    screen_y = np.interp(iy, (0, cam_h), (0, screen_h * MOVEMENT_SCALE_Y))
                    
                    # Apply low-pass filter
                    filtered_x, filtered_y = low_pass_filter.update(screen_x, screen_y)
                    
                    # Apply additional smoothing
                    clocx = plocx + (filtered_x - plocx) / SMOOTHING
                    clocy = plocy + (filtered_y - plocy) / SMOOTHING
                    
                    # Limit maximum movement per frame to prevent jumps
                    dx = clocx - plocx
                    dy = clocy - plocy
                    distance = np.sqrt(dx*dx + dy*dy)
                    
                    if distance > MAX_MOVEMENT_PER_FRAME:
                        # Scale down the movement
                        scale = MAX_MOVEMENT_PER_FRAME / distance
                        dx *= scale
                        dy *= scale
                        clocx = plocx + dx
                        clocy = plocy + dy
                    
                    # Ensure coordinates are within screen bounds
                    clocx = max(0, min(clocx, screen_w))
                    clocy = max(0, min(clocy, screen_h))
                    
                    # Move mouse with absolute positioning
                    move_mouse(clocx, clocy)
                    
                    if DEBUG:
                        print(f"Camera: ({ix}, {iy}), Screen: ({clocx:.0f}, {clocy:.0f})")
                    
                    plocx, plocy = clocx, clocy
                    
                    # Draw a circle on index finger tip for feedback
                    cv2.circle(image, (ix, iy), 10, (255, 0, 0), cv2.FILLED)

            # Display the resulting frame
            cv2.imshow('Hand Tracking Mouse Control', image)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    finally:
        # Release resources
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("Script terminated.")

if __name__ == '__main__':
    main()
