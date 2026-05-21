import cv2
import numpy as np
import mediapipe as mp
import urllib.request
import os
import time

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Download models if missing ────────────────────────────────────
HAND_MODEL = "hand_landmarker.task"
FACE_MODEL = "face_landmarker.task"

MODELS = {
    HAND_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    FACE_MODEL: (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    ),
}

for path, url in MODELS.items():
    if not os.path.exists(path):
        print(f"Downloading {path}…")
        urllib.request.urlretrieve(url, path)
        print(f"{path} ready!")

# ── MediaPipe setup ───────────────────────────────────────────────
HAND_CONNECTIONS = mp_vision.HandLandmarksConnections.HAND_CONNECTIONS

hand_detector = mp_vision.HandLandmarker.create_from_options(
    mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )
)

face_detector = mp_vision.FaceLandmarker.create_from_options(
    mp_vision.FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.6,
    )
)

# ── Gesture classifier ────────────────────────────────────────────
def classify_gesture(lm):
    tip_ids = [4, 8, 12, 16, 20]
    pip_ids = [3, 6, 10, 14, 18]
    fingers = [1 if lm[tip_ids[0]].x < lm[pip_ids[0]].x else 0]
    for i in range(1, 5):
        fingers.append(1 if lm[tip_ids[i]].y < lm[pip_ids[i]].y else 0)
    total = sum(fingers)
    if total == 0:  return "Fist",      (50, 50, 200)
    if total == 5:  return "Open hand", (50, 200, 50)
    if fingers[1] and fingers[2] and not fingers[3] and not fingers[4]:
        return "Peace",     (200, 200, 50)
    if fingers[0] and not fingers[1] and not fingers[2]:
        return "Thumbs up", (200, 100, 50)
    if fingers[1] and not fingers[2] and not fingers[3] and not fingers[4]:
        return "Pointing",  (100, 50, 200)
    if total == 1 and fingers[4]:
        return "Pinky",     (50, 200, 200)
    return f"Hand ({total} fingers)", (150, 150, 150)

# ── Emotion classifier from face landmarks ────────────────────────
# Key landmark indices in MediaPipe's 478-point face model:
#   13 = upper lip center, 14 = lower lip center
#   61 = left mouth corner, 291 = right mouth corner
#   159 = left eye upper lid, 145 = left eye lower lid
#   70  = left eyebrow inner, 107 = left eyebrow outer
DEBUG_METRICS = {}

def classify_emotion(face_lm):
    upper_lip    = face_lm[13]
    lower_lip    = face_lm[14]
    left_corner  = face_lm[61]
    right_corner = face_lm[291]
    eye_upper    = face_lm[159]
    eye_lower    = face_lm[145]
    brow_inner   = face_lm[70]
    brow_outer   = face_lm[107]

    mouth_open     = abs(lower_lip.y - upper_lip.y)
    mouth_center_y = (upper_lip.y + lower_lip.y) / 2
    corner_avg_y   = (left_corner.y + right_corner.y) / 2
    mouth_curve    = mouth_center_y - corner_avg_y

    eye_height   = abs(eye_lower.y - eye_upper.y)
    brow_avg_y   = (brow_inner.y + brow_outer.y) / 2
    brow_raise   = eye_upper.y - brow_avg_y

    DEBUG_METRICS["curve"]  = round(mouth_curve, 4)
    DEBUG_METRICS["open"]   = round(mouth_open, 4)
    DEBUG_METRICS["brow"]   = round(brow_raise, 4)
    DEBUG_METRICS["eye_h"]  = round(eye_height, 4)

    if brow_raise > 0.065:
        return "Surprise", (220, 180, 50)
    if mouth_curve > 0.012:
        return "Happy",    (50, 220, 50)
    return "Neutral",      (180, 180, 180)

# ── Draw hand landmarks ───────────────────────────────────────────
def draw_hand(frame, lm):
    h, w = frame.shape[:2]
    pts  = [(int(l.x * w), int(l.y * h)) for l in lm]
    for conn in HAND_CONNECTIONS:
        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 160, 80), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 220, 120), -1)

# ── Draw HUD ──────────────────────────────────────────────────────
def draw_hud(frame, gesture, g_color, emotion, e_color, hand_count):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 70), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "YUGI", (16, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.putText(frame, gesture, (130, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, g_color, 2)
    cv2.putText(frame, f"Emotion: {emotion}", (130, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, e_color, 2)
    cv2.putText(frame, f"Hands: {hand_count}", (w - 150, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 36), (w, h), (15, 15, 15), -1)
    cv2.addWeighted(overlay2, 0.65, frame, 0.35, 0, frame)
    cv2.putText(frame, "Week 2 | Gesture + Emotion | Press Q to quit",
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
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Yugi Week 2 - Gesture + Emotion Detection | Press Q to quit")

    current_gesture = "No hand detected"
    g_color         = (150, 150, 150)
    current_emotion = "No face"
    e_color         = (150, 150, 150)
    hand_count      = 0
    frame_count     = 0
    start_time      = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame  = cv2.flip(frame, 1)
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ts_ms  = int((time.time() - start_time) * 1000)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # ── Gesture detection (every frame) ──────────────────────
        hand_result = hand_detector.detect_for_video(mp_img, ts_ms)
        hand_count  = 0
        if hand_result.hand_landmarks:
            hand_count = len(hand_result.hand_landmarks)
            for lm in hand_result.hand_landmarks:
                draw_hand(frame, lm)
                current_gesture, g_color = classify_gesture(lm)
        else:
            current_gesture = "No hand detected"
            g_color         = (100, 100, 100)

        # ── Emotion detection (every 5 frames for performance) ───
        if frame_count % 5 == 0:
            face_result = face_detector.detect_for_video(mp_img, ts_ms)
            if face_result.face_landmarks:
                current_emotion, e_color = classify_emotion(
                    face_result.face_landmarks[0]
                )
            else:
                current_emotion = "No face"
                e_color         = (100, 100, 100)

        frame_count += 1

        draw_hud(frame, current_gesture, g_color, current_emotion, e_color, hand_count)

        if DEBUG_METRICS:
            debug_str = "  ".join(f"{k}:{v}" for k, v in DEBUG_METRICS.items())
            cv2.putText(frame, debug_str, (12, frame.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        cv2.imshow("Yugi - Week 2", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    hand_detector.close()
    face_detector.close()
    print("Yugi shutting down.")

if __name__ == "__main__":
    main()
