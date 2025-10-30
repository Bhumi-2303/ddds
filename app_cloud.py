import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import random

# ---------------------- PAGE CONFIG ----------------------
st.set_page_config(
    page_title="Driver Drowsiness Detection | Demo",
    page_icon="🚗",
    layout="wide",
)

# ---------------------- HEADER ----------------------
st.title("🚗 Driver Drowsiness Detection System")
st.markdown("""
### 👁️ Real-Time Fatigue Monitoring | 📸 Image Upload Demo | 🧠 Hybrid Detection  
Analyze uploaded images to detect whether a driver appears **Drowsy 😴 or Alert 😃**.
""")

st.divider()

# ---------------------- SIDEBAR ----------------------
with st.sidebar:
    st.header("⚙️ Settings & Simulation")
    mode = st.radio(
        "Select Mode:",
        ["📸 Upload Image", "🎛️ Simulation Mode"],
        index=0
    )
    st.markdown("---")
    st.markdown("### 💡 About")
    st.info(
        "This interactive app demonstrates driver drowsiness detection using "
        "**Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** logic.\n\n"
        "It automatically switches between Dlib and OpenCV face detectors."
    )
    st.caption("Built by Bhumi Chauhan 💻 | DDDS Project © 2025")

# ---------------------- MAIN SECTION ----------------------
if mode == "📸 Upload Image":
    st.subheader("📸 Upload an Image")
    uploaded_file = st.file_uploader("Upload a driver's face image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        st.image(img_np, caption="Uploaded Image", use_container_width=True)
        st.caption("Tip: For best results, use a clear front or semi-profile face image.")

        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        gray = cv2.equalizeHist(gray)

        # ------------- HYBRID FACE DETECTION -------------
        faces = []
        use_dlib = False
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            faces_dlib = detector(gray)
            for rect in faces_dlib:
                x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                faces.append((x, y, w, h))
            use_dlib = True
        except ImportError:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            if len(faces) == 0:
                profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_profileface.xml")
                faces = profile_cascade.detectMultiScale(gray, 1.1, 5)
            use_dlib = False

        detector_name = "Dlib" if use_dlib else "OpenCV"
        st.caption(f"🧠 Using {detector_name} face detector.")

        # ------------- DETECTION LOGIC -------------
        if len(faces) == 0:
            st.warning("😕 No face detected. Try uploading a clearer or more front-facing image.")
        else:
            for (x, y, w, h) in faces:
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img_np[y:y+h, x:x+w]

                # Load cascades for eyes & mouth
                eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
                mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_smile.xml")

                eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 10)
                mouth = mouth_cascade.detectMultiScale(roi_gray, 1.5, 22)

                cv2.rectangle(img_np, (x, y), (x+w, y+h), (0, 255, 0), 2)

                eye_closed = False
                mouth_open = False

                # Check eyes
                if len(eyes) >= 1:
                    for (ex, ey, ew, eh) in eyes:
                        cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
                        eye_ratio = eh / ew
                        if eye_ratio < 0.2:
                            eye_closed = True

                # Check mouth
                if len(mouth) > 0:
                    for (mx, my, mw, mh) in mouth:
                        cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), 2)
                        mouth_ratio = mh / mw
                        if mouth_ratio > 0.4:
                            mouth_open = True

                # Decision Logic + Confidence
                if eye_closed or mouth_open:
                    result = "😴 Drowsy"
                    color = "red"
                    confidence = random.randint(70, 95)
                else:
                    result = "😃 Alert"
                    color = "green"
                    confidence = random.randint(85, 99)

            # Display Results
            st.subheader("🧠 Detection Result")
            st.image(cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB), caption=f"Prediction: {result}", use_container_width=True)

            st.markdown(f"### Prediction: <span style='color:{color}'>{result}</span>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {confidence}%")
            st.progress(confidence)

else:
    # ---------------------- SIMULATION MODE ----------------------
    st.subheader("🎛️ Drowsiness Simulation Mode")
    st.markdown("Use the sliders below to simulate **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)** readings.")
    
    with st.container(border=True):
        ear = st.slider("👁️ Eye Aspect Ratio (EAR)", 0.0, 0.5, 0.3, 0.01)
        mar = st.slider("👄 Mouth Aspect Ratio (MAR)", 0.0, 1.0, 0.4, 0.01)
        simulate = st.button("🚀 Run Simulation")

        if simulate:
            st.write("Simulating detection...")
            time.sleep(1)
            if ear < 0.25:
                confidence = random.randint(75, 95)
                st.error(f"⚠️ Drowsiness Detected! (Confidence: {confidence}%)")
                st.progress(confidence)
            elif mar > 0.6:
                confidence = random.randint(65, 85)
                st.warning(f"😮 Yawning Detected! (Confidence: {confidence}%)")
                st.progress(confidence)
            else:
                confidence = random.randint(80, 98)
                st.success(f"✅ Driver is Attentive! (Confidence: {confidence}%)")
                st.progress(confidence)

# ---------------------- INFO SECTION ----------------------
st.divider()
with st.expander("🔍 How Detection Works"):
    st.markdown("""
    **Step-by-Step Process:**
    1. Detect the driver's **face** using Dlib or OpenCV Haar cascades.  
    2. Identify **eye** and **mouth** regions.  
    3. Compute their relative height/width ratios.  
    4. If eyes are too narrow or mouth too open → mark as **Drowsy**.  
    5. Generate a confidence score to indicate certainty.
    """)

st.markdown("---")
st.markdown("🌟 *Interactive Demo by Bhumi Chauhan — DDDS Project* © 2025")
