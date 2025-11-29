#!/usr/bin/env python3

import cv2
import mediapipe as mp
import socket
import os
import argparse
import time
import sys
import glob

# --- ULTRA LOW POWER SETTINGS ---
# Lower resolution = Faster processing
CAMERA_WIDTH = 320
CAMERA_HEIGHT = 240
# How long to sleep between checks (seconds).
# 0.2 = 5 checks per second. Increase to 0.5 for even less CPU.
POLL_INTERVAL = 0.2

# Sensitivity
LOOK_LEFT_THRESHOLD = -0.4
LOOK_RIGHT_THRESHOLD = 0.4

# --- IPC CLIENT (No Subprocess) ---
class HyprlandClient:
    def __init__(self):
        self.socket_path = None
        self._find_socket()

    def _find_socket(self):
        # Locate Hyprland socket
        signature = os.getenv("HYPRLAND_INSTANCE_SIGNATURE")
        candidates = glob.glob("/tmp/hypr/*/.socket.sock")
        if signature:
            candidates.insert(0, f"/tmp/hypr/{signature}/.socket.sock")
            xdg = os.getenv("XDG_RUNTIME_DIR")
            if xdg: candidates.append(f"{xdg}/hypr/{signature}/.socket.sock")

        for path in candidates:
            if os.path.exists(path):
                self.socket_path = path
                return
        print("❌ Error: Hyprland socket not found.", file=sys.stderr)

    def focus_monitor(self, monitor_id):
        if not self.socket_path: return
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.settimeout(0.1) # Don't block
                s.connect(self.socket_path)
                s.sendall(f"dispatch focusmonitor {monitor_id}".encode('utf-8'))
                s.recv(1024) # Clear buffer
        except Exception:
            pass # Ignore errors to keep loop tight

# --- MAIN LOGIC ---
def run_tracker(left_id, center_id, right_id, debug_mode):
    # 1. Setup Camera (Auto-detect)
    cap = None
    for idx in [2, 1, 0]:
        temp = cv2.VideoCapture(idx)
        if temp.isOpened():
            ret, _ = temp.read()
            if ret:
                cap = temp
                # Optimize Camera for CPU
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, 5) # Request low FPS from hardware
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Don't buffer old frames
                print(f"✅ Using Camera {idx}")
                break
            temp.release()

    if not cap:
        print("❌ No camera found.")
        return

    # 2. Setup Lightweight Face Detection (BlazeFace)
    # This is MUCH faster than FaceMesh (6 points vs 468 points)
    mp_face_detection = mp.solutions.face_detection
    detector = mp_face_detection.FaceDetection(
        model_selection=0, # 0 = Short range (faces < 2m), very fast
        min_detection_confidence=0.5
    )

    hypr = HyprlandClient()
    current_zone = "center"

    print("🚀 Ultra-Low Power Tracker Started.")

    try:
        while True:
            # Grab a frame
            success, image = cap.read()
            if not success:
                time.sleep(1)
                continue

            # Process Frame
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = detector.process(image)

            if results.detections:
                # Get the first face
                detection = results.detections[0]

                # Extract Keypoints (Relative 0.0 - 1.0)
                # Keypoint 4: Right Ear, 5: Left Ear, 2: Nose Tip
                # (Note: MediaPipe Keypoint indices for FaceDetection)
                keypoints = detection.location_data.relative_keypoints
                right_ear = keypoints[4]
                left_ear = keypoints[5]
                nose = keypoints[2]

                # Math: Calculate Rotation
                # Note: Camera is mirrored? Usually webcam feed is not mirrored in raw data
                # but visually we flip it. Let's calculate raw.
                # If nose is closer to left ear (in x), looking that way.

                ear_center_x = (left_ear.x + right_ear.x) / 2
                face_width = abs(left_ear.x - right_ear.x)

                if face_width > 0:
                    # Invert logic because image is not flipped in this script to save CPU
                    # Standard Webcam: Left Ear is on Right side of pixels
                    rot = (nose.x - ear_center_x) / (face_width * 0.5)

                    # Determine Zone
                    # You might need to swap these signs depending on your specific camera mirroring
                    # Current logic: Nose moves right -> Rot increases
                    target_monitor = None
                    new_zone = current_zone

                    if rot < LOOK_LEFT_THRESHOLD: # Looking Right (in raw image) -> Left in Real Life
                        new_zone = "right" # Swap if inverted
                        target_monitor = right_id
                    elif rot > LOOK_RIGHT_THRESHOLD:
                        new_zone = "left"  # Swap if inverted
                        target_monitor = left_id
                    else:
                        new_zone = "center"
                        target_monitor = center_id

                    # Fix Direction if needed (User reported mirroring issues usually)
                    # Let's assume standard selfie mirror for calculations
                    # If this is backward, swap "--left" and "--right" args when running.

                    if new_zone != current_zone:
                        if target_monitor is not None:
                            hypr.focus_monitor(target_monitor)
                            # print(f"Switch -> {new_zone}") # Comment out for silent run
                        current_zone = new_zone

            # --- THE MAGIC SLEEP ---
            # This is what gives you 1% CPU.
            # We don't need to check 30 times a second. 5 is enough.
            time.sleep(POLL_INTERVAL)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        detector.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=int, help="ID for Left Monitor")
    parser.add_argument("--center", type=int, default=0, help="ID for Center Monitor")
    parser.add_argument("--right", type=int, help="ID for Right Monitor")
    args = parser.parse_args()

    # NOTE: I removed the debug window code entirely to save resources.
    # If directions are swapped, just swap your --left and --right arguments.
    run_tracker(args.left, args.center, args.right, False)
