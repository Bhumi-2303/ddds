import cv2
import dlib
import numpy as np
import pandas as pd
import datetime
import pygame
import time
from scipy.spatial import distance as dist
from tensorflow.keras.models import load_model
import os

# ================================
# 🔊 Initialize pygame for buzzer
# ================================
pygame.mixer.init()
pygame.mixer.music.load(os.path.join(os.path.dirname(__file__), "buzzer.mp3"))

# ================================
# 🧠 Load pre-trained model
# ================================
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/mobilenetv2_base.h5")
model = load_model(MODEL_PATH)

# ================================
# 🧍 Face detection + landmarks
# ================================
PREDICTOR_PATH = os.path.join(os.path.dirname(__file__), "../models/shape_predictor_68_face_landmarks.dat")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# ================================
# 🗂️ Directory setup
# ================================
SELF_LEARNING_DIR = os.path.join(os.path.dirname(__file__), "../personalized_data")
os.makedirs(SELF_LEARNING_DIR, exist_ok=True)

HISTORY_FILE = os.path.join(os.path.dirname(__file__), "../driver_history.csv")
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=["Timestamp", "Status"]).to_csv(HISTORY_FILE, index=False)

# ================================
# 📏 Landmark index definitions
# ================================
LEFT_EYE = list(range(36, 42))
RIGHT_EYE = list(range(42, 48))
MOUTH = list(range(48, 68))

# ================================
# 📐 Helper functions
# ================================
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    return (A + B) / (2.0 * C)

# ================================
# ⚙️ Thresholds and initialization
# ================================
EYE_THRESH = 0.25
MOUTH_THRESH = 0.70
EYE_FRAMES = 15

frame_counter = 0
status = "Normal"
cap = cv2.VideoCapture(0)
buzzer_on = False
buzzer_start_time = 0

# ================================
# 🚘 Main loop
# ================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in LEFT_EYE])
        right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in RIGHT_EYE])
        mouth = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in MOUTH])

        leftEAR = eye_aspect_ratio(left_eye)
        rightEAR = eye_aspect_ratio(right_eye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # =============================
        # 🚨 Detection conditions
        # =============================
        if ear < EYE_THRESH:
            frame_counter += 1
            if frame_counter >= EYE_FRAMES:
                status = "Drowsy"
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                if not buzzer_on:
                    pygame.mixer.music.play()
                    buzzer_on = True
                    buzzer_start_time = time.time()

        elif mar > MOUTH_THRESH:
            status = "Yawning"
            cv2.putText(frame, "YAWNING DETECTED!", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 3)
            if not buzzer_on:
                pygame.mixer.music.play()
                buzzer_on = True
                buzzer_start_time = time.time()

        else:
            frame_counter = 0
            status = "Normal"

        # Draw feature outlines
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [mouth], True, (255, 255, 0), 1)

    # Stop buzzer after 2 seconds
    if buzzer_on and (time.time() - buzzer_start_time > 2):
        pygame.mixer.music.stop()
        buzzer_on = False

    # Display current status
    cv2.putText(frame, f"Status: {status}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    # =============================
    # 🧾 Save driver history
    # =============================
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df = pd.DataFrame([[current_time, status]], columns=["Timestamp", "Status"])
    df.to_csv(HISTORY_FILE, mode='a', header=False, index=False)

    # =============================
    # 🧠 Save self-learning data
    # =============================
    if status != "Normal":
        filename = os.path.join(SELF_LEARNING_DIR, f"{status}_{current_time.replace(':', '-')}.jpg")
        cv2.imwrite(filename, frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ================================
# 🧹 Cleanup
# ================================
cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
