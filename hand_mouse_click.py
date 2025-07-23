import cv2
import mediapipe as mp
import math
import subprocess
import time

# Prerequisites for this script to work:
# 1. Install ydotool:
#    On Debian/Ubuntu: sudo apt-get install ydotool
#    On Arch Linux: sudo pacman -S ydotool
#
# 2. Start the ydotoold daemon. This is best done as a systemd service,
#    but for a quick test, you can run this in a separate terminal:
#    sudo ydotoold --socket-path="$HOME/.ydotool.socket" --socket-own="$(id -u):$(id -g)"
#
# 3. Set the YDOTOOL_SOCKET environment variable for the client (this script).
#    You can do this before running the script:
#    export YDOTOOL_SOCKET="$HOME/.ydotool.socket"
#    python3 hand_mouse_click.py

# --- Configuration ---
CAMERA_INDEX = 0  # 0 for the first USB camera, 1 for the second, etc.
PINCH_THRESHOLD = 0.06  # Normalized distance for pinch detection. Adjust as needed.
POLL_RATE_MS = 10  # ms to wait between checks. Lower for more responsiveness.
FRAME_SKIP = 1  # Process every Nth frame. Increase to reduce CPU usage.
FRAME_WIDTH = 640  # Smaller resolution = less CPU
FRAME_HEIGHT = 480
SHOW_PREVIEW = True  # Set to False to disable the preview window (saves CPU)

# --- State ---
left_click_held = False
right_click_held = False
frame_counter = 0

# --- MediaPipe setup ---
mp_hands = mp.solutions.hands
# Reduce max_num_hands to 1 for efficiency
# Increase min_detection_confidence and min_tracking_confidence to reduce false positives
hands = mp_hands.Hands(
    static_image_mode=False,  # Set to False for video (more efficient)
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=0  # Use the lightest model (0, 1, or 2)
)
mp_drawing = mp.solutions.drawing_utils
# Simplified drawing specs for better performance
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

def get_normalized_distance(landmarks, p1_id, p2_id):
    """Calculates Euclidean distance between two landmarks in normalized coordinates."""
    p1 = landmarks.landmark[p1_id]
    p2 = landmarks.landmark[p2_id]
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def is_finger_extended(landmarks, tip_id, pip_id, mcp_id):
    """
    Checks if a finger is extended by comparing y-coordinates of tip, pip, and mcp joints.
    MediaPipe's y-axis origin is at the top, so a smaller y is higher.
    """
    tip = landmarks.landmark[tip_id]
    pip = landmarks.landmark[pip_id]
    mcp = landmarks.landmark[mcp_id]
    return tip.y < pip.y and pip.y < mcp.y

def ydotool_mouse_down(button):
    """Presses a mouse button down. 0 for left, 1 for right."""
    # Using click with the button code + 0x40 (mouse down flag)
    button_code = button + 0x40  # 0x40 for mouse down
    subprocess.run(['ydotool', 'click', hex(button_code)], check=False)

def ydotool_mouse_up(button):
    """Releases a mouse button. 0 for left, 1 for right."""
    # Using click with the button code + 0x80 (mouse up flag)
    button_code = button + 0x80  # 0x80 for mouse up
    subprocess.run(['ydotool', 'click', hex(button_code)], check=False)

def manage_click_state(is_pinch, is_extended, was_held, button_code, button_name):
    """Manages the state of a mouse click (down/up)."""
    is_held = was_held
    if is_pinch and is_extended:
        if not was_held:
            print(f"{button_name} click down")
            ydotool_mouse_down(button_code)
            is_held = True
    elif was_held:
        print(f"{button_name} click up")
        ydotool_mouse_up(button_code)
        is_held = False
    return is_held


def main():
    global left_click_held, right_click_held, frame_counter

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"Error: Could not open camera {CAMERA_INDEX}.")
        print("Please make sure a camera is connected and the index is correct.")
        return

    # Set camera resolution to reduce processing load
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    # Disable autofocus to save CPU
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)

    print("Starting hand gesture mouse control. Press 'q' in the OpenCV window to quit.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue
                
            frame_counter += 1
            if frame_counter % FRAME_SKIP != 0:
                # Skip this frame to reduce CPU load
                if SHOW_PREVIEW:
                    cv2.imshow('Hand Mouse Control', frame)
                if cv2.waitKey(POLL_RATE_MS) & 0xFF == ord('q'):
                    break
                continue

            # Flip horizontally for more intuitive interaction
            frame = cv2.flip(frame, 1)
            
            # Convert only when we're processing the frame
            # Process in RGB (required by MediaPipe)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process the frame with MediaPipe
            results = hands.process(frame_rgb)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]  # We only use one hand
                
                if SHOW_PREVIEW:
                    # Draw landmarks with simplified specs for better performance
                    mp_drawing.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=drawing_spec,
                        connection_drawing_spec=drawing_spec
                    )
                
                # Only check the fingers we need
                index_finger_extended = is_finger_extended(
                    hand_landmarks,
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_PIP,
                    mp_hands.HandLandmark.INDEX_FINGER_MCP
                )
                middle_finger_extended = is_finger_extended(
                    hand_landmarks,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_MCP
                )

                thumb_index_dist = get_normalized_distance(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP)
                thumb_middle_dist = get_normalized_distance(hand_landmarks, mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.MIDDLE_FINGER_TIP)

                is_left_pinch = thumb_index_dist < PINCH_THRESHOLD
                is_right_pinch = thumb_middle_dist < PINCH_THRESHOLD

                # Left click uses button 0, right click uses button 1
                left_click_held = manage_click_state(is_left_pinch, index_finger_extended, left_click_held, 0x00, "Left")
                right_click_held = manage_click_state(is_right_pinch, middle_finger_extended, right_click_held, 0x01, "Right")

            else:
                # No hands detected, release any active clicks
                if left_click_held:
                    print("Left click up (hand out of frame)")
                    ydotool_mouse_up(0x00)
                    left_click_held = False
                if right_click_held:
                    print("Right click up (hand out of frame)")
                    ydotool_mouse_up(0x01)
                    right_click_held = False

            if SHOW_PREVIEW:
                cv2.imshow('Hand Mouse Control', frame)

            if cv2.waitKey(POLL_RATE_MS) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if left_click_held:
            ydotool_mouse_up(0x00)
        if right_click_held:
            ydotool_mouse_up(0x01)
        
        cap.release()
        cv2.destroyAllWindows()
        hands.close()
        print("Script terminated.")

if __name__ == "__main__":
    main()
