#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import time
import sys
import math
import argparse
import threading
from typing import Tuple, List

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Constants
SCREEN_WIDTH = 1920  # Adjust to your screen resolution
SCREEN_HEIGHT = 1080  # Adjust to your screen resolution
SMOOTHING_FACTOR = 0.8  # Adjust for smoother transitions (higher = smoother but more lag)
FOCUS_COOLDOWN = 1.5  # Seconds between focus changes
SCROLL_COOLDOWN = 0.3  # Seconds between scrolls
VIDEO_WIDTH = 320  # Lower resolution for better performance
VIDEO_HEIGHT = 240  # Lower resolution for better performance
FRAME_RATE = 10  # Process fewer frames per second to reduce CPU usage
PROCESS_EVERY_N_FRAMES = 3  # Only process every Nth frame to reduce CPU usage
GESTURE_STABILIZATION_FRAMES = 3  # Number of frames to stabilize a gesture
SCROLL_AMOUNT = 1  # Constant scroll amount (adjust as needed)
PINCH_THRESHOLD = 0.02  # Threshold for pinch detection
CLICK_COOLDOWN = 0.5  # Seconds between clicks

# Head position thresholds
LEFT_THRESHOLD = -0.6  # Values below this are considered "looking left"
RIGHT_THRESHOLD = 0.5  # Values above this are considered "looking right"
# Values between LEFT_THRESHOLD and RIGHT_THRESHOLD are considered "center"

# Camera indices to try (USB cameras typically start at 0 or 1)
CAMERA_INDICES = [1, 0, 2]  # Try USB camera (1) first, then built-in (0), then another USB port (2)

class HeadTracker:
    def __init__(self, center_position="center", debug_mode=True):
        self.last_focus_time = 0
        self.prev_head_rotation = 0
        self.monitors = self._get_monitors()
        
        # Performance optimization: Use static image mode for less CPU usage
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False  # Faster processing for video
        )
        
        # Hand tracking setup
        self.hands = mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )

        # Gesture control state
        self.last_scroll_time = 0
        self.last_click_time = 0
        self.gesture_buffer = []
        
        self.cap = None
        self.running = False
        self.debug_mode = debug_mode
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        
        # Center position configuration
        self.center_position = center_position
        print(f"Center position set to: {self.center_position}")
        
        # Set current monitor based on center position
        if self.center_position == "right" and len(self.monitors) > 1:
            self.current_monitor = 1  # Right monitor
        else:  # "left" or "center"
            self.current_monitor = 0  # Left monitor
            
        # Initialize focus to the current monitor
        self._switch_to_monitor(self.current_monitor)
        print(f"Initially focused on monitor {self.current_monitor}")
        
        # For tracking head position zone
        self.current_head_zone = "center"
        
    def _get_monitors(self):
        """Get list of monitors from Hyprland"""
        try:
            result = subprocess.run(
                ["hyprctl", "monitors", "-j"],
                capture_output=True, text=True, check=True
            )
            monitors = []
            # Parse monitor information from hyprctl
            # This is a simplified version, we'll just store monitor IDs
            for i, _ in enumerate(result.stdout.strip().split("id")):
                if i > 0:  # Skip the first split which is empty
                    monitors.append(i-1)
            
            if not monitors:
                print("No monitors detected, using default")
                monitors = [0]
            
            print(f"Detected {len(monitors)} monitors: {monitors}")
            return monitors
        except Exception as e:
            print(f"Error getting monitors: {e}")
            return [0]  # Default to one monitor
    
    def _try_open_camera(self):
        """Try to open camera, attempting multiple indices"""
        for idx in CAMERA_INDICES:
            print(f"Trying camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Successfully opened camera at index {idx}")
                return cap
        
        print("Could not open any camera")
        return None
        
    def start(self):
        """Start the head tracking process"""
        self.cap = self._try_open_camera()
        if not self.cap:
            print("Error: Could not open any camera.")
            return
            
        # Set lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        
        self.running = True
        print("Head tracking started. Press 'q' to quit.")
        
        try:
            self._tracking_loop()
        finally:
            self.stop()
    
    def stop(self):
        """Stop the head tracking process"""
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Head tracking stopped.")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.running and self.cap.isOpened():
            # Calculate FPS
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_frame_time = current_time
            else:
                self.frame_count += 1
            
            # Read frame
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame from camera.")
                break
            
            # Flip the image horizontally for a selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Skip frames to reduce CPU usage
            if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                if self.debug_mode:
                    self._update_debug_display(frame)
                    cv2.imshow('MediaPipe Head Tracking', frame)
                
                # Check for quit command
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue
            
            # Process frame
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = self.face_mesh.process(image_rgb)
            hand_results = self.hands.process(image_rgb)
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                if self.debug_mode:
                    # Draw face mesh
                    self._draw_face_mesh(frame, face_landmarks)
                
                # Get head rotation
                head_rotation = self._estimate_head_rotation(face_landmarks)
                
                # Apply smoothing
                smoothed_rotation = self._smooth_head_rotation(head_rotation)
                
                # Determine head position zone (left, center, right)
                self._update_head_zone(smoothed_rotation)
                
                # Focus monitor based on head zone
                self._focus_monitor_based_on_head_zone()
                
                if self.debug_mode:
                    # Add debug info to image
                    self._update_debug_display(frame, smoothed_rotation)
            
            if hand_results.multi_hand_landmarks:
                for hand_landmarks in hand_results.multi_hand_landmarks:
                    if self.debug_mode:
                        self._draw_hand_landmarks(frame, hand_landmarks)
                    self._process_hand_gestures(hand_landmarks)

            # Display the image if in debug mode
            if self.debug_mode:
                cv2.imshow('MediaPipe Head Tracking', frame)
            
            # Check for quit command
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    def _update_debug_display(self, frame, smoothed_rotation=None):
        """Update the debug information displayed on the frame"""
        # Add FPS counter
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if smoothed_rotation is not None:
            cv2.putText(frame, f"Head Rotation: {smoothed_rotation:.2f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Head Zone: {self.current_head_zone}", (10, 90), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Current Monitor: {self.current_monitor}", (10, 120), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Center: {self.center_position}", (10, 150), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    def _draw_face_mesh(self, image, face_landmarks):
        """Draw the face mesh on the image"""
        # Only draw essential landmarks to save CPU
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,  # Only draw eyes
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )

    def _draw_hand_landmarks(self, image, hand_landmarks):
        """Draw hand landmarks on the image"""
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
        )

    def _process_hand_gestures(self, hand_landmarks):
        """Detect and process hand gestures for mouse control."""
        current_time = time.time()

        # First check for pinch gesture (higher priority)
        if self._detect_pinch_gesture(hand_landmarks) and (current_time - self.last_click_time) > CLICK_COOLDOWN:
            self._execute_mouse_action("left_click")
            self.last_click_time = current_time
            return  # Skip scroll detection if we just clicked

        # Detect scroll gesture from the current frame
        scroll_gesture = self._detect_scroll_gesture(hand_landmarks)
        
        # Add to gesture buffer for stabilization
        self.gesture_buffer.append(scroll_gesture)
        if len(self.gesture_buffer) > GESTURE_STABILIZATION_FRAMES:
            self.gesture_buffer.pop(0)
            
        # Determine stable gesture
        stable_gesture = "none"
        if len(self.gesture_buffer) == GESTURE_STABILIZATION_FRAMES and all(g == self.gesture_buffer[0] for g in self.gesture_buffer):
            stable_gesture = self.gesture_buffer[0]

        # Cooldown check
        can_scroll = (current_time - self.last_scroll_time) > SCROLL_COOLDOWN

        # Execute action if gesture is stable and not "none"
        if can_scroll and stable_gesture != "none":
            self._execute_mouse_action(stable_gesture)
            self.last_scroll_time = current_time

    def _detect_pinch_gesture(self, hand_landmarks) -> bool:
        """Detect pinch gesture between index finger and thumb."""
        landmarks = hand_landmarks.landmark
        
        # Get thumb and index finger tip positions
        thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        
        # Calculate distance between thumb and index finger tips
        distance = self._get_landmark_dist(thumb_tip, index_tip)
        
        # Return True if the distance is less than the threshold
        return distance < PINCH_THRESHOLD

    def _get_landmark_dist(self, p1, p2) -> float:
        """Calculate Euclidean distance between two landmarks."""
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)

    def _detect_scroll_gesture(self, hand_landmarks) -> str:
        """
        Detects scroll gesture: Index and middle fingers pointing up or down,
        while other fingers are folded.
        """
        # Get landmark y-coordinates
        def is_finger_extended(tip_landmark, pip_landmark):
            return tip_landmark.y < pip_landmark.y

        def is_finger_folded(tip_landmark, pip_landmark):
            return tip_landmark.y > pip_landmark.y

        landmarks = hand_landmarks.landmark

        index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]

        # Check if index and middle fingers are close together
        fingers_are_joint = self._get_landmark_dist(index_tip, middle_tip) < PINCH_THRESHOLD

        if not fingers_are_joint:
            return "none"

        # Check extension/flexion of each finger
        index_extended = is_finger_extended(index_tip, landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP])
        middle_extended = is_finger_extended(middle_tip, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])
        ring_folded = is_finger_folded(landmarks[mp_hands.HandLandmark.RING_FINGER_TIP], landmarks[mp_hands.HandLandmark.RING_FINGER_PIP])
        pinky_folded = is_finger_folded(landmarks[mp_hands.HandLandmark.PINKY_TIP], landmarks[mp_hands.HandLandmark.PINKY_PIP])

        index_down = is_finger_folded(index_tip, landmarks[mp_hands.HandLandmark.INDEX_FINGER_PIP])
        middle_down = is_finger_folded(middle_tip, landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_PIP])

        # Scroll up gesture
        if index_extended and middle_extended and ring_folded and pinky_folded:
            return "scroll_up"
        
        # Scroll down gesture
        if index_down and middle_down and ring_folded and pinky_folded:
            return "scroll_down"
            
        return "none"

    def _execute_mouse_action(self, action: str):
        """Executes a mouse action using ydotool."""
        command = []

        if action == "scroll_up":
            command = ["ydotool", "mousemove", "-w", "-x 0", f"-y {SCROLL_AMOUNT}"]
        elif action == "scroll_down":
            command = ["ydotool", "mousemove", "-w", "-x 0", f"-y -{SCROLL_AMOUNT}"]
        elif action == "left_click":
            command = ["ydotool", "click", "c0"]

        if command:
            try:
                subprocess.run(command, check=True, capture_output=True, text=True)
                print(f"Executed action: {action}")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print(f"Error executing ydotool for {action}: {e.stderr if hasattr(e, 'stderr') else e}")
    
    def _estimate_head_rotation(self, face_landmarks) -> float:
        """Estimate head rotation from face landmarks (left-right)"""
        # Get key points for head pose
        left_temple = face_landmarks.landmark[234]  # Left temple
        right_temple = face_landmarks.landmark[454]  # Right temple
        nose_tip = face_landmarks.landmark[4]  # Nose tip
        
        # Calculate horizontal position of nose relative to temples
        temple_center_x = (left_temple.x + right_temple.x) / 2
        temple_width = abs(right_temple.x - left_temple.x)
        
        # Normalized position of nose relative to face width (-1 to 1)
        # Negative values mean looking left, positive values mean looking right
        if temple_width > 0:
            rotation = (nose_tip.x - temple_center_x) / (temple_width / 2)
            return max(-1.0, min(1.0, rotation))  # Clamp between -1 and 1
        return 0.0
    
    def _smooth_head_rotation(self, current_rotation) -> float:
        """Apply smoothing to head rotation to reduce jitter"""
        smoothed_rotation = self.prev_head_rotation * SMOOTHING_FACTOR + current_rotation * (1 - SMOOTHING_FACTOR)
        self.prev_head_rotation = smoothed_rotation
        return smoothed_rotation
    
    def _update_head_zone(self, head_rotation):
        """Update the current head position zone (left, center, right)"""
        if head_rotation < LEFT_THRESHOLD:
            self.current_head_zone = "left"
        elif head_rotation > RIGHT_THRESHOLD:
            self.current_head_zone = "right"
        else:
            self.current_head_zone = "center"
    
    def _focus_monitor_based_on_head_zone(self):
        """Focus monitor based on head zone and center position setting"""
        current_time = time.time()
        
        # Apply cooldown to prevent too frequent focus changes
        if current_time - self.last_focus_time < FOCUS_COOLDOWN:
            return
        
        # Only proceed if we have multiple monitors
        if len(self.monitors) <= 1:
            return
            
        # Determine target monitor based on head zone and center position
        target_monitor = None
        
        if self.center_position == "left":
            # When center is "left", both center and left head zones focus left monitor
            if self.current_head_zone in ["left", "center"]:
                target_monitor = 0  # Left monitor
            else:  # right head zone
                target_monitor = 1  # Right monitor
                
        elif self.center_position == "right":
            # When center is "right", both center and right head zones focus right monitor
            if self.current_head_zone in ["right", "center"]:
                target_monitor = 1  # Right monitor
            else:  # left head zone
                target_monitor = 0  # Left monitor
                
        else:  # center position is "center"
            # When center is "center", match head zone to monitor
            if self.current_head_zone == "left":
                target_monitor = 0  # Left monitor
            elif self.current_head_zone == "right":
                target_monitor = 1  # Right monitor
            # For center head zone, keep current monitor (no change)
        
        # Only switch if we have a target and it's different from current
        if target_monitor is not None and target_monitor != self.current_monitor:
            self._switch_to_monitor(target_monitor)
            self.current_monitor = target_monitor
            self.last_focus_time = current_time
    
    def _switch_to_monitor(self, monitor_id):
        """Switch focus to the specified monitor"""
        try:
            # Use Hyprland to focus the monitor
            subprocess.run(
                ["hyprctl", "dispatch", "focusmonitor", str(monitor_id)],
                capture_output=True, check=True
            )
            print(f"Switched to monitor {monitor_id}")
        except subprocess.CalledProcessError as e:
            print(f"Error focusing monitor: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")

def parse_args():
    parser = argparse.ArgumentParser(description="Head tracking for Hyprland monitor focus")
    parser.add_argument("--center", choices=["left", "center", "right"], default="center",
                        help="Define where the center position is (default: center)")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug display to save CPU")
    return parser.parse_args()

def main():
    args = parse_args()
    tracker = HeadTracker(center_position=args.center, debug_mode=not args.no_debug)
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        tracker.stop()

if __name__ == "__main__":
    main()
