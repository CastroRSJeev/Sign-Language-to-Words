import cv2
import numpy as np
import torch
import torch.nn as nn
from collections import deque
from mediapipe import Image, ImageFormat
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode, HandLandmarksConnections

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes       = np.load("model/label_map.npy", allow_pickle=True)
num_classes   = len(classes)
idx_to_letter = {i: c for i, c in enumerate(classes)}

class LandmarkMLP(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(63, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
    def forward(self, x):
        return self.net(x)

model = LandmarkMLP(num_classes).to(device)
model.load_state_dict(torch.load("model/sign_model.pth", map_location=device))
model.eval()

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="model/hand_landmarker.task"),
    running_mode=RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.6,
    min_hand_presence_confidence=0.5,
)

BUFFER_SIZE       = 15
CONFIRM_THRESHOLD = 12
buffer         = deque(maxlen=BUFFER_SIZE)
word           = ""
last_confirmed = ""

cap = cv2.VideoCapture(0)
print("Controls: SPACE=add space | BACKSPACE=delete | C=clear | Q=quit")

with HandLandmarker.create_from_options(options) as detector:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame    = cv2.flip(frame, 1)
        h, w     = frame.shape[:2]
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
        result   = detector.detect(mp_image)

        letter, confidence = "", 0.0

        if result.hand_landmarks:
            lm  = result.hand_landmarks[0]
            pts = [(int(l.x * w), int(l.y * h)) for l in lm]

            for conn in HandLandmarksConnections.HAND_CONNECTIONS:
                cv2.line(frame, pts[conn.start], pts[conn.end], (0, 200, 255), 2)
            for pt in pts:
                cv2.circle(frame, pt, 4, (0, 255, 0), -1)

            coords = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
            inp    = torch.tensor(coords).unsqueeze(0).to(device)

            with torch.no_grad():
                probs = torch.softmax(model(inp), dim=1)
            conf, idx  = probs.max(1)
            confidence = conf.item()
            letter     = idx_to_letter[idx.item()]

            buffer.append(letter)
            if len(buffer) == BUFFER_SIZE:
                most_common = max(set(buffer), key=buffer.count)
                if buffer.count(most_common) >= CONFIRM_THRESHOLD and most_common != last_confirmed:
                    word          += most_common
                    last_confirmed = most_common
        else:
            buffer.clear()
            last_confirmed = ""

        # UI
        cv2.rectangle(frame, (0, h - 80), (w, h), (0, 0, 0), -1)
        cv2.putText(frame, f"Word: {word}", (10, h - 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(frame, "SPC=space  BS=del  C=clear  Q=quit", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        if letter:
            cv2.putText(frame, f"{letter}  {confidence*100:.0f}%", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 200, 255), 3)

        cv2.imshow("Sign Language", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            word += " "
            last_confirmed = ""
        elif key == 8:
            word = word[:-1]
        elif key == ord('c'):
            word, last_confirmed = "", ""

cap.release()
cv2.destroyAllWindows()
