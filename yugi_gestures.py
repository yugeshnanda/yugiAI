import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Download model if missing ─────────────────────────────────────
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    )
    print("Downloading hand landmarker model (~8 MB)…")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model ready!")

# ── MediaPipe setup ───────────────────────────────────────────────
HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=mp_vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_tracking_confidence=0.5,
)

# ── Gesture classifier ────────────────────────────────────────────
def classify_gesture(lm):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]

    fingers = [1 if lm[tip_ids[0]].x < lm[pip_ids[0]].x else 0]
    for i in range(1, 5):
        fingers.append(1 if lm[tip_ids[i]].y < lm[pip_ids[i]].y else 0)

    total = sum(fingers)

    if total == 0:
        return "Fist", (50, 50, 200)
    if total == 5:
        return "Open hand", (50, 200, 50)
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "Peace", (200, 200, 50)
    if fingers[0] and not fingers[1] and not fingers[2]:
        return "Thumbs up", (200, 100, 50)
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "Pointing", (100, 50, 200)
    if total == 1 and fingers[4]:
        return "Pinky", (50, 200, 200)

    return f"Hand ({total} fingers)", (150, 150, 150)

# ── Draw landmarks ────────────────────────────────────────────────
def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 160, 80), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 220, 120), -1)

# ── Draw HUD overlay ──────────────────────────────────────────────
def draw_hud(frame, gesture_label, color, hand_count):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 60), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "YUGI", (16, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, gesture_label, (130, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.putText(frame, f"Hands: {hand_count}", (w - 140, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 36), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Week 1 | Gesture Detection | Press Q to quit",
                (12, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (160, 160, 160), 1)

# ── Main loop ─────────────────────────────────────────────────────
def main():
    cap = None
    for i in range(3):
        test = cv2.VideoCapture(i)
        if test.isOpened():
            cap = test
            print(f"Camera found at index {i}")
            break
        test.release()

    if cap is None:
        print("No camera found!")
        exit()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Yugi - Gesture Detection | Press Q to quit")

    current_gesture = "No hand detected"
    current_color   = (150, 150, 150)
    hand_count      = 0
    start_time      = time.time()

    with mp_vision.HandLandmarker.create_from_options(options) as detector:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame  = cv2.flip(frame, 1)
            rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ts_ms  = int((time.time() - start_time) * 1000)
            result = detector.detect_for_video(
                mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms
            )

            hand_count = 0
            if result.hand_landmarks:
                hand_count = len(result.hand_landmarks)
                for lm in result.hand_landmarks:
                    draw_hand(frame, lm)
                    current_gesture, current_color = classify_gesture(lm)
            else:
                current_gesture = "No hand detected"
                current_color   = (100, 100, 100)

            draw_hud(frame, current_gesture, current_color, hand_count)
            cv2.imshow("Yugi - Week 1", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print("Yugi shutting down.")

if __name__ == "__main__":
    main()
