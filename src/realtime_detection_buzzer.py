import cv2
import dlib
import pygame
import numpy as np
from scipy.spatial import distance as dist
from datetime import datetime
import pandas as pd
import os
from tensorflow.keras.models import load_model

# ==== Initialize Pygame for Buzzer ====
pygame.mixer.init()
pygame.mixer.music.load("buzzer.mp3")

# ==== Paths ====
MODEL_PATH = "../models/mobilenetv2_base.h5"
PREDICTOR_PATH = "../models/shape_predictor_68_face_landmarks.dat"
LOG_PATH = "../logs/driver_history.csv"

# ==== Load Models ====
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
model = load_model(MODEL_PATH)

# ==== EAR & MAR Thresholds ====
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 15
MOUTH_AR_THRESH = 0.6

COUNTER = 0
ALARM_ON = False

# ==== Utility Functions ====
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

def log_event(event, details=""):
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    df = pd.DataFrame([[time, event, details]], columns=["Time", "Event", "Details"])
    if not os.path.exists(LOG_PATH):
        df.to_csv(LOG_PATH, index=False)
    else:
        df_existing = pd.read_csv(LOG_PATH)
        df_combined = pd.concat([df_existing, df]).tail(20)  # Keep last 20 events
        df_combined.to_csv(LOG_PATH, index=False)

def play_buzzer():
    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.play()

# ==== Main Detection Loop ====
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("⚠️ Camera not detected. Please check connection.")
    exit()

print("🔵 Driver Drowsiness Detection System Started...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    for face in faces:
        shape = predictor(gray, face)
        shape_np = np.zeros((68, 2), dtype=int)
        for i in range(68):
            shape_np[i] = (shape.part(i).x, shape.part(i).y)

        leftEye = shape_np[42:48]
        rightEye = shape_np[36:42]
        mouth = shape_np[48:68]

        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        mar = mouth_aspect_ratio(mouth)

        # ==== Check for Drowsiness ====
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True
                    play_buzzer()
                    log_event("Drowsy", f"Eyes closed for {COUNTER/10:.1f}s")
        else:
            COUNTER = 0
            ALARM_ON = False

        # ==== Check for Yawn ====
        if mar > MOUTH_AR_THRESH:
            if 'yawn_count' not in locals():
                yawn_count = 0
                last_yawn_time = datetime.now()
            
            time_diff = (datetime.now() - last_yawn_time).seconds
            if time_diff > 30:  # reset counter every 30 seconds
                yawn_count = 0
            
            yawn_count += 1
            last_yawn_time = datetime.now()
            log_event("Yawn", f"Yawn #{yawn_count} | MAR={mar:.2f}")

            if yawn_count >= 6:
                play_buzzer()
                log_event("Alert", f"6 yawns detected within short time!")
                yawn_count = 0  # reset counter after buzzer

    # ==== Display Frame ====
    cv2.putText(frame, "Status: Drowsy" if ALARM_ON else "Status: Normal",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (0, 0, 255) if ALARM_ON else (0, 255, 0), 2)

    cv2.imshow("Driver Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
