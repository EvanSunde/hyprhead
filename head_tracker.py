import cv2
import mediapipe as mp
import time
import subprocess

# For webcam input:
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Mediapipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Monitor state
focused_monitor = None  # can be 'left', 'right', or None
last_switch_time = 0
debounce_time = 2  # 2 seconds

def focus_monitor(direction):
    global focused_monitor, last_switch_time
    current_time = time.time()
    if direction != focused_monitor and (current_time - last_switch_time) > debounce_time:
        print(f"Focusing {direction} monitor")
        if direction == "left":
            subprocess.run(["hyprctl", "dispatch", "focusmonitor", "l"])
        elif direction == "right":
            subprocess.run(["hyprctl", "dispatch", "focusmonitor", "r"])
        focused_monitor = direction
        last_switch_time = current_time

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    # Draw the face mesh annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get landmarks for eyes and nose
            nose_tip = face_landmarks.landmark[1]
            left_eye_inner = face_landmarks.landmark[145]
            right_eye_inner = face_landmarks.landmark[374]

            # Simple logic for head direction
            # Compare x-coordinates of nose tip with inner eye landmarks
            nose_x = nose_tip.x
            left_eye_x = left_eye_inner.x
            right_eye_x = right_eye_inner.x

            # Determine head orientation
            if nose_x < left_eye_x - 0.03: # Looking left
                focus_monitor("left")
            elif nose_x > right_eye_x + 0.03: # Looking right
                focus_monitor("right")

            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=drawing_spec,
                connection_drawing_spec=drawing_spec)

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Face Mesh', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
face_mesh.close() 
