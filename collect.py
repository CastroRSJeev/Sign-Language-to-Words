import cv2
import numpy as np
import csv
import os
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode, HandLandmarksConnections

LABELS   = list("ABCDEFGHIKLMNOPQRSTUVWXY")  # No J, no Z
SAMPLES  = 200   # samples per letter
OUT_CSV  = "data/landmarks.csv"
os.makedirs("data", exist_ok=True)

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
)

# Find already collected labels so we can resume
existing = set()
if os.path.exists(OUT_CSV):
    with open(OUT_CSV) as f:
        for row in csv.reader(f):
            if row:
                existing.add(row[0])
    print(f"Resuming — already collected: {sorted(existing)}")

file_mode = "a" if os.path.exists(OUT_CSV) else "w"

with open(OUT_CSV, file_mode, newline="") as f:
    writer = csv.writer(f)

    with HandLandmarker.create_from_options(options) as detector:
        cap = cv2.VideoCapture(0)

        for letter in LABELS:
            if letter in existing:
                print(f"Skipping {letter} (already collected)")
                continue

            collected = 0
            print(f"\n>>> Get ready for letter: {letter}  —  press SPACE to start, Q to quit")

            # Wait for space
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                cv2.putText(frame, f"Next: {letter} — press SPACE to start", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
                cv2.imshow("Collect", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):
                    break
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Quit.")
                    exit()

            # Collect samples
            while collected < SAMPLES:
                ret, frame = cap.read()
                if not ret:
                    break
                frame    = cv2.flip(frame, 1)
                h, w     = frame.shape[:2]
                rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
                result   = detector.detect(mp_image)

                detected = False
                if result.hand_landmarks:
                    lm  = result.hand_landmarks[0]
                    pts = [(int(l.x * w), int(l.y * h)) for l in lm]
                    for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                        cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 255), 2)
                    for pt in pts:
                        cv2.circle(frame, pt, 4, (0, 255, 0), -1)

                    coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
                    writer.writerow([letter] + coords.tolist())
                    f.flush()
                    collected += 1
                    detected   = True

                color = (0, 255, 0) if detected else (0, 0, 255)
                cv2.putText(frame, f"{letter}: {collected}/{SAMPLES}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
                cv2.imshow("Collect", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Quit.")
                    exit()

            print(f"  {letter}: {collected} samples saved")

        cap.release()
        cv2.destroyAllWindows()

print(f"\nDone! Data saved to {OUT_CSV}")
