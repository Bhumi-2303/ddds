import streamlit as st
import os

# --- Page setup ---
st.set_page_config(page_title="Driver Drowsiness Detection (Demo)", layout="wide")
st.title("🚗 Driver Drowsiness Detection System – Demo Mode")

st.markdown("""
Welcome to the **Driver Drowsiness Detection System** demo.  
This version is optimized for **Streamlit Cloud**, where webcam access is restricted.  
You can explore how the model works and how to run it locally for real-time detection.
""")

# --- Overview section ---
st.header("📖 Overview")
st.markdown("""
This system monitors a driver’s eye and mouth activity to detect signs of drowsiness or fatigue.  
If prolonged eye closure or yawning is detected, an **alarm sound** is triggered to alert the driver.
""")

# --- Model details ---
st.header("🧠 Model & Files Used")
st.code("""
models/
├── shape_predictor_68_face_landmarks.dat   # Dlib landmark model
├── mobilenetv2_base.h5                     # CNN model for feature extraction
""")

st.success("These models are used locally to analyze facial features and detect signs of drowsiness.")

# --- Working principle ---
st.header("⚙️ How It Works")
st.markdown("""
1. **Face Detection:** The system locates the driver’s face.  
2. **Landmark Extraction:** Key facial points are identified using Dlib.  
3. **EAR (Eye Aspect Ratio):** Detects eye closure duration.  
4. **MAR (Mouth Aspect Ratio):** Detects yawning.  
5. **Alert System:** If thresholds are crossed, a buzzer (`buzzer.mp3`) plays to alert the driver.
""")

# --- Single demo image ---
st.header("🖼️ Example Output")
st.image("demo.jpg", caption="Detected Drowsiness Example", use_container_width=True)

st.info("⚠️ Webcam access is disabled on Streamlit Cloud. Please run locally to experience full detection.")

# --- Local run instructions ---
st.markdown("---")
st.header("💻 Run Locally")
st.code("""
# Clone and run locally
git clone https://github.com/bhumi-2303/ddds.git
cd ddds
pip install -r requirements.txt
streamlit run app_local_dlib.py
""")

# --- Footer ---
st.markdown("---")
st.markdown("© 2025 Bhumi Chauhan – Driver Drowsiness Detection System")
