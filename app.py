import cv2
import numpy as np
import dlib
import streamlit as st
from scipy.spatial import distance as dist
import threading
import time
import pygame
import os

if not os.path.exists("models/shape_predictor_68_face_landmarks.dat"):
    st.error("⚠️ Model files missing! Please ensure you have `models/shape_predictor_68_face_landmarks.dat` and `mobilenetv2_base.h5` uploaded.")
    st.stop()

# Initialize pygame for sound alert
pygame.mixer.init()
pygame.mixer.music.load("buzzer.mp3")  # <-- make sure this file exists in your project folder

# Constants
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 20
MOUTH_AR_THRESH = 0.6

# Initialize counters
COUNTER = 0
ALARM_ON = False

# Dlib face detector + landmarks
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(os.path.join("models", "shape_predictor_68_face_landmarks.dat"))  # pretrained model file

# Helper functions
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])  # 51, 59
    B = dist.euclidean(mouth[4], mouth[8])   # 53, 57
    C = dist.euclidean(mouth[0], mouth[6])   # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

def sound_alarm():
    pygame.mixer.music.play()
    time.sleep(2)
    pygame.mixer.music.stop()

# Streamlit UI setup
st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("### Stay safe with real-time eye & mouth monitoring.")

run = st.checkbox("Start Detection")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)
    st.success("Webcam started successfully.")
    while run:
        ret, frame = cap.read()
        if not ret:
            st.warning("Camera not detected.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = np.array([[p.x, p.y] for p in shape.parts()])

            leftEye = shape[36:42]
            rightEye = shape[42:48]
            mouth = shape[48:68]

            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            mar = mouth_aspect_ratio(mouth)

            # Eye aspect ratio visualization
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(mouth)], -1, (255, 0, 0), 1)

            

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    if not ALARM_ON:
                        ALARM_ON = True
                        threading.Thread(target=sound_alarm).start()

                    cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
            else:
                COUNTER = 0
                ALARM_ON = False

            if mar > MOUTH_AR_THRESH:
                cv2.putText(frame, "YAWNING DETECTED!", (50, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)

            cv2.putText(frame, f"EAR: {ear:.2f}", (30, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame)

    cap.release()
    st.info("Webcam stopped successfully.")
else:
    st.info("Click 'Start Detection' to activate webcam.")
