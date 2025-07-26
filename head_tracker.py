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
import socket
import os
from typing import Tuple, List
from collections import deque

# MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Constants
SCREEN_WIDTH = 1920 
SCREEN_HEIGHT = 1080
SMOOTHING_FACTOR = 0.8
FOCUS_COOLDOWN = 1.5
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
FRAME_RATE = 30
PROCESS_EVERY_N_FRAMES = 3 # Increased to reduce CPU load

# Hand tracking constants
PINCH_THRESHOLD = 0.03
CLICK_COOLDOWN = 0.5
SCROLL_COOLDOWN = 0.1
SCROLL_REGION_THRESHOLD = 0.5
SCROLL_AMOUNT_UPPER = 2
SCROLL_AMOUNT_LOWER = 5
PINCH_DRAG_THRESHOLD = 0.03
PINCH_HISTORY_SIZE = 5

# Head position thresholds
LEFT_THRESHOLD = -0.6
RIGHT_THRESHOLD = 0.5

# Camera indices to try
CAMERA_INDICES = [0, 1, 2]

# ydotool daemon socket path
YDOTOD_SOCKET_PATH = "/run/user/1000/.ydotool_socket"

class HeadTracker:
    def __init__(self, center_position="center", debug_mode=True, enable_hand_gestures=True):
        self.last_focus_time = 0
        self.prev_head_rotation = 0
        self.monitors = self._get_monitors()
        
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            static_image_mode=False
        )
        
        self.hands = None
        self.enable_hand_gestures = enable_hand_gestures
        self.last_click_time = 0
        self.last_scroll_time = 0
        self.pinch_active = False
        self.pinch_start_y = 0
        self.continuous_scroll_active = False
        self.continuous_scroll_speed = 0
        self.continuous_scroll_direction = 0
        self.last_hand_landmarks = None
        self.pinch_history = deque(maxlen=PINCH_HISTORY_SIZE)
        
        self.debug_frame = None
        self.last_debug_update_time = 0
        self.debug_update_interval = 0.1
        
        self.cap = None
        self.running = False
        self.debug_mode = debug_mode
        self.frame_count = 0
        self.last_frame_time = time.time()
        self.fps = 0
        
        self.center_position = center_position
        print(f"Center position set to: {self.center_position}")
        
        if self.center_position == "right" and len(self.monitors) > 1:
            self.current_monitor = 1
        else:
            self.current_monitor = 0
            
        self._switch_to_monitor(self.current_monitor)
        print(f"Initially focused on monitor {self.current_monitor}")
        
        self.current_head_zone = "center"

    def _get_monitors(self):
        try:
            result = subprocess.run(
                ["hyprctl", "monitors", "-j"],
                capture_output=True, text=True, check=True
            )
            monitors = []
            for i, _ in enumerate(result.stdout.strip().split("id")):
                if i > 0:
                    monitors.append(i-1)
            
            if not monitors:
                print("No monitors detected, using default")
                return [0]
            
            print(f"Detected {len(monitors)} monitors: {monitors}")
            return monitors
        except Exception as e:
            print(f"Error getting monitors: {e}")
            return [0]

    def _try_open_camera(self):
        for idx in CAMERA_INDICES:
            print(f"Trying camera index {idx}...")
            cap = cv2.VideoCapture(idx)
            if cap.isOpened():
                print(f"Successfully opened camera at index {idx}")
                return cap
        print("Could not open any camera")
        return None
        
    def start(self):
        self.cap = self._try_open_camera()
        if not self.cap:
            print("Error: Could not open any camera.")
            return
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
        
        success, frame = self.cap.read()
        if success:
            self.debug_frame = frame.copy()
        
        if self.enable_hand_gestures:
            try:
                print("Initializing hand tracking...")
                self.hands = mp_hands.Hands(
                    max_num_hands=1,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5,
                    static_image_mode=False
                )
                print("Hand tracking initialized successfully")
            except Exception as e:
                print(f"Error initializing hand tracking: {e}")
                self.enable_hand_gestures = False
        
        self.running = True
        print("Head tracking started. Press 'q' to quit.")
        
        try:
            self._tracking_loop()
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("Head tracking stopped.")
    
    def _tracking_loop(self):
        while self.running and self.cap.isOpened():
            current_time = time.time()
            if current_time - self.last_frame_time >= 1.0:
                self.fps = self.frame_count
                self.frame_count = 0
                self.last_frame_time = current_time
            
            self.frame_count += 1
            
            success, frame = self.cap.read()
            if not success:
                print("Failed to capture frame from camera.")
                break
            
            frame = cv2.flip(frame, 1)
            
            if self.continuous_scroll_active and current_time - self.last_scroll_time > SCROLL_COOLDOWN:
                self._execute_scroll(self.continuous_scroll_direction, self.continuous_scroll_speed)
                self.last_scroll_time = current_time
            
            if self.frame_count % PROCESS_EVERY_N_FRAMES != 0:
                if self.debug_mode and self.debug_frame is not None:
                    cv2.imshow('MediaPipe Head Tracking', self.debug_frame)
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
                continue
            
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            face_results = self.face_mesh.process(image_rgb)
            
            if self.enable_hand_gestures and self.hands:
                try:
                    hand_results = self.hands.process(image_rgb)
                    if hand_results.multi_hand_landmarks:
                        self._process_hand_gestures(hand_results.multi_hand_landmarks, frame)
                except Exception as e:
                    print(f"Error processing hand gestures: {e}")
                    self.enable_hand_gestures = False
                    print("Hand tracking disabled due to error")
            
            if face_results.multi_face_landmarks:
                face_landmarks = face_results.multi_face_landmarks[0]
                
                if self.debug_mode:
                    self._draw_face_mesh(frame, face_landmarks)
                
                head_rotation = self._estimate_head_rotation(face_landmarks)
                smoothed_rotation = self._smooth_head_rotation(head_rotation)
                self._update_head_zone(smoothed_rotation)
                self._focus_monitor_based_on_head_zone()
                
                if self.debug_mode:
                    self._update_debug_display(frame, smoothed_rotation)

            if self.debug_mode and (current_time - self.last_debug_update_time) >= self.debug_update_interval:
                self.debug_frame = frame.copy()
                self.last_debug_update_time = current_time
                cv2.imshow('MediaPipe Head Tracking', self.debug_frame)
            
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
    
    def _update_debug_display(self, frame, smoothed_rotation=None):
        cv2.putText(frame, f"FPS: {self.fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if smoothed_rotation is not None:
            cv2.putText(frame, f"Head Rotation: {smoothed_rotation:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Head Zone: {self.current_head_zone}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Current Monitor: {self.current_monitor}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Center: {self.center_position}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if self.enable_hand_gestures:
            if self.pinch_active:
                cv2.putText(frame, "Pinch Active", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if self.continuous_scroll_active:
                cv2.putText(frame, f"Scrolling: {'Down' if self.continuous_scroll_direction > 0 else 'Up'} ({self.continuous_scroll_speed})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    def _draw_face_mesh(self, image, face_landmarks):
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
        )
    
    def _draw_hand_landmarks(self, image, hand_landmarks):
        if self.debug_mode:
            try:
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=hand_landmarks,
                    connections=mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style()
                )
            except Exception as e:
                print(f"Error drawing hand landmarks: {e}")
    
    def _is_pinch_stable(self, is_current_pinch):
        self.pinch_history.append(is_current_pinch)
        if len(self.pinch_history) < PINCH_HISTORY_SIZE:
            return is_current_pinch
        true_count = sum(1 for state in self.pinch_history if state)
        return true_count > (PINCH_HISTORY_SIZE // 2)
    
    def _process_hand_gestures(self, multi_hand_landmarks, frame):
        current_time = time.time()
        
        if len(multi_hand_landmarks) > 0:
            try:
                hand_landmarks = multi_hand_landmarks[0]
                self._draw_hand_landmarks(frame, hand_landmarks)
                
                raw_pinch = self._detect_pinch_gesture(hand_landmarks)
                is_pinching = self._is_pinch_stable(raw_pinch)
                
                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                
                if is_pinching:
                    if not self.pinch_active:
                        self.pinch_active = True
                        self.pinch_start_y = index_tip.y
                        
                        if self._detect_three_finger_pinch(hand_landmarks):
                            if current_time - self.last_click_time > CLICK_COOLDOWN:
                                self._execute_mouse_action("right_click")
                                self.last_click_time = current_time
                    else:
                        y_movement = index_tip.y - self.pinch_start_y
                        
                        if abs(y_movement) > PINCH_DRAG_THRESHOLD:
                            direction = 1 if y_movement > 0 else -1
                            scroll_amount = SCROLL_AMOUNT_LOWER if index_tip.y > SCROLL_REGION_THRESHOLD else SCROLL_AMOUNT_UPPER
                            
                            self.continuous_scroll_active = True
                            self.continuous_scroll_direction = direction
                            self.continuous_scroll_speed = scroll_amount
                            
                            if current_time - self.last_scroll_time > SCROLL_COOLDOWN:
                                self._execute_scroll(direction, scroll_amount)
                                self.last_scroll_time = current_time
                else:
                    if self.pinch_active:
                        # If a brief pinch occurred without scrolling, treat as a left click
                        if not self.continuous_scroll_active and (current_time - self.last_click_time > CLICK_COOLDOWN):
                            self._execute_mouse_action("left_click")
                            self.last_click_time = current_time
                        
                        self.pinch_active = False
                        self.continuous_scroll_active = False
                
                self.last_hand_landmarks = hand_landmarks
            except Exception as e:
                print(f"Error processing hand gesture: {e}")
    
    def _detect_pinch_gesture(self, hand_landmarks) -> bool:
        try:
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            distance = self._get_landmark_dist(thumb_tip, index_tip)
            return distance < PINCH_THRESHOLD
        except Exception as e:
            print(f"Error detecting pinch gesture: {e}")
            return False
    
    def _detect_three_finger_pinch(self, hand_landmarks) -> bool:
        try:
            landmarks = hand_landmarks.landmark
            thumb_tip = landmarks[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = landmarks[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = landmarks[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            thumb_index_dist = self._get_landmark_dist(thumb_tip, index_tip)
            thumb_middle_dist = self._get_landmark_dist(thumb_tip, middle_tip)
            
            return (thumb_index_dist < PINCH_THRESHOLD and thumb_middle_dist < PINCH_THRESHOLD * 1.2)
        except Exception as e:
            print(f"Error detecting three-finger pinch: {e}")
            return False
    
    def _get_landmark_dist(self, p1, p2) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2 + (p1.z - p2.z)**2)
    
    def _send_ydotoold_cmd(self, cmd: str):
        """Sends a command to the running ydotoold daemon."""
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(YDOTOD_SOCKET_PATH)
                s.sendall(cmd.encode())
        except (ConnectionRefusedError, FileNotFoundError):
            # This is a common, non-fatal error if the daemon isn't running
            pass
        except Exception as e:
            print(f"Error sending command to ydotoold: {e}")

    def _execute_mouse_action(self, action: str):
        """Execute a mouse action using ydotoold."""
        command = ""
        if action == "left_click":
            command = "click 0xC0\n" # Keycode for left click
        elif action == "right_click":
            command = "click 0xC2\n" # Keycode for right click
        
        if command:
            self._send_ydotoold_cmd(command)
            print(f"Executed daemon action: {action}")
    
    def _execute_scroll(self, direction, amount):
        """Execute a scroll action using ydotoold."""
        scroll_y = amount * direction
        command = f"mousemove -w 0 {scroll_y}\n"
        self._send_ydotoold_cmd(command)
        print(f"Daemon scrolling: direction={direction}, amount={amount}")
    
    def _estimate_head_rotation(self, face_landmarks) -> float:
        left_temple = face_landmarks.landmark[234]
        right_temple = face_landmarks.landmark[454]
        nose_tip = face_landmarks.landmark[4]
        
        temple_center_x = (left_temple.x + right_temple.x) / 2
        temple_width = abs(right_temple.x - left_temple.x)
        
        if temple_width > 0:
            rotation = (nose_tip.x - temple_center_x) / (temple_width / 2)
            return max(-1.0, min(1.0, rotation))
        return 0.0
    
    def _smooth_head_rotation(self, current_rotation) -> float:
        smoothed_rotation = self.prev_head_rotation * SMOOTHING_FACTOR + current_rotation * (1 - SMOOTHING_FACTOR)
        self.prev_head_rotation = smoothed_rotation
        return smoothed_rotation
    
    def _update_head_zone(self, head_rotation):
        if head_rotation < LEFT_THRESHOLD:
            self.current_head_zone = "left"
        elif head_rotation > RIGHT_THRESHOLD:
            self.current_head_zone = "right"
        else:
            self.current_head_zone = "center"
    
    def _focus_monitor_based_on_head_zone(self):
        current_time = time.time()
        
        if current_time - self.last_focus_time < FOCUS_COOLDOWN:
            return
        
        if len(self.monitors) <= 1:
            return
            
        target_monitor = self.current_monitor
        if self.center_position == "left":
            target_monitor = 1 if self.current_head_zone == "right" else 0
        elif self.center_position == "right":
            target_monitor = 0 if self.current_head_zone == "left" else 1
        else: # center
            if self.current_head_zone == "left":
                target_monitor = 0
            elif self.current_head_zone == "right":
                target_monitor = 1
        
        if target_monitor != self.current_monitor:
            self._switch_to_monitor(target_monitor)
            self.current_monitor = target_monitor
            self.last_focus_time = current_time
    
    def _switch_to_monitor(self, monitor_id):
        try:
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
    parser = argparse.ArgumentParser(description="Head and hand tracking for Hyprland")
    parser.add_argument("--center", choices=["left", "center", "right"], default="center",
                        help="Define the 'home' monitor position (default: center)")
    parser.add_argument("--no-debug", action="store_true",
                        help="Disable debug display to save resources")
    parser.add_argument("--no-hands", action="store_true",
                        help="Disable hand gesture tracking")
    return parser.parse_args()

def main():
    args = parse_args()
    
    if not os.path.exists(YDOTOD_SOCKET_PATH):
        print("Warning: ydotoold socket not found. Hand gestures will not work.")
        print("Please run 'ydotoold' in a separate terminal.")

    tracker = HeadTracker(
        center_position=args.center, 
        debug_mode=not args.no_debug,
        enable_hand_gestures=not args.no_hands
    )
    try:
        tracker.start()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        tracker.stop()

if __name__ == "__main__":
    main()