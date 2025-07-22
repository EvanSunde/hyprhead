#!/usr/bin/env python3

import cv2
import mediapipe as mp
import numpy as np
import subprocess
import time
import sys
import math
import argparse
from typing import Tuple, List

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Constants
SCREEN_WIDTH = 1920  # Adjust to your screen resolution
SCREEN_HEIGHT = 1080  # Adjust to your screen resolution
SMOOTHING_FACTOR = 0.8  # Adjust for smoother transitions (higher = smoother but more lag)
FOCUS_COOLDOWN = 1.5  # Seconds between focus changes
HEAD_ROTATION_THRESHOLD = 0.15  # Threshold for head rotation to trigger monitor change
VIDEO_WIDTH = 320  # Lower resolution for better performance
VIDEO_HEIGHT = 240  # Lower resolution for better performance

# Eye landmarks indices (based on MediaPipe Face Mesh)
LEFT_EYE_INDICES = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_IRIS_INDICES = [474, 475, 476, 477]
RIGHT_IRIS_INDICES = [469, 470, 471, 472]

# Head pose landmarks
HEAD_POSE_LANDMARKS = [33, 263, 1, 61, 291, 199]

class HeadTracker:
    def __init__(self, center_position="center"):
        self.last_focus_time = 0
        self.prev_gaze_point = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.prev_head_rotation = 0
        self.current_monitor = 0
        self.monitors = self._get_monitors()
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.cap = None
        self.running = False
        self.debug_mode = True
        
        # Center position configuration
        self.center_position = center_position
        print(f"Center position set to: {self.center_position}")
        
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
        
    def start(self):
        """Start the head tracking process"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera.")
            return
            
        # Set lower resolution for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        
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
            success, image = self.cap.read()
            if not success:
                print("Failed to capture frame from camera.")
                break
            
            # Flip the image horizontally for a selfie-view display
            image = cv2.flip(image, 1)
            
            # Convert to RGB and process with MediaPipe
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0]
                
                if self.debug_mode:
                    # Draw face mesh
                    self._draw_face_mesh(image, face_landmarks)
                
                # Get head rotation
                head_rotation = self._estimate_head_rotation(face_landmarks)
                
                # Apply smoothing
                smoothed_rotation = self._smooth_head_rotation(head_rotation)
                
                # Focus monitor based on head rotation
                self._focus_monitor_based_on_head(smoothed_rotation)
                
                if self.debug_mode:
                    # Add debug info to image
                    cv2.putText(image, f"Head Rotation: {smoothed_rotation:.2f}", (10, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Current Monitor: {self.current_monitor}", (10, 60), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(image, f"Center: {self.center_position}", (10, 90), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the image if in debug mode
            if self.debug_mode:
                cv2.imshow('MediaPipe Head Tracking', image)
            
            # Check for quit command
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    def _draw_face_mesh(self, image, face_landmarks):
        """Draw the face mesh on the image"""
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
        )
        
        # Draw eyes and irises with different color
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    
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
    
    def _focus_monitor_based_on_head(self, head_rotation):
        """Focus monitor based on head rotation"""
        current_time = time.time()
        
        # Apply cooldown to prevent too frequent focus changes
        if current_time - self.last_focus_time < FOCUS_COOLDOWN:
            return
        
        # Determine which monitor to focus based on head rotation and center position
        if len(self.monitors) > 1:
            # Adjust thresholds based on center position
            if self.center_position == "left":
                # When center is left, we need to turn head more to the left to trigger left monitor
                if head_rotation < -HEAD_ROTATION_THRESHOLD * 2:
                    target_monitor = 0  # Left monitor
                elif head_rotation > -HEAD_ROTATION_THRESHOLD / 2:
                    target_monitor = min(len(self.monitors) - 1, 1)  # Right monitor
                else:
                    return  # No change needed
            elif self.center_position == "right":
                # When center is right, we need to turn head more to the right to trigger right monitor
                if head_rotation > HEAD_ROTATION_THRESHOLD * 2:
                    target_monitor = min(len(self.monitors) - 1, 1)  # Right monitor
                elif head_rotation < HEAD_ROTATION_THRESHOLD / 2:
                    target_monitor = 0  # Left monitor
                else:
                    return  # No change needed
            else:  # center (default)
                if head_rotation < -HEAD_ROTATION_THRESHOLD:
                    target_monitor = 0  # Left monitor
                elif head_rotation > HEAD_ROTATION_THRESHOLD:
                    target_monitor = min(len(self.monitors) - 1, 1)  # Right monitor
                else:
                    return  # No change needed
            
            # Only change if different from current
            if target_monitor != self.current_monitor:
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
    return parser.parse_args()

def main():
    args = parse_args()
    tracker = HeadTracker(center_position=args.center)
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        tracker.stop()

if __name__ == "__main__":
    main()
