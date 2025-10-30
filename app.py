import streamlit as st
import os

st.set_page_config(page_title="Driver Drowsiness Detection", layout="wide")
st.title("🚗 Driver Drowsiness Detection System (Demo Mode)")
st.markdown("""
This app detects driver drowsiness using **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**.
Due to Streamlit Cloud restrictions, webcam access is **disabled online**,  
but you can [run this app locally](#run-locally) to experience full detection in real time.
""")

st.subheader("📊 Model Files")
st.code("""
models/
├── shape_predictor_68_face_landmarks.dat
├── mobilenetv2_base.h5
""")

st.subheader("🧠 How it Works")
st.markdown("""
1. Detects face landmarks using Dlib  
2. Calculates **Eye Aspect Ratio (EAR)** and **Mouth Aspect Ratio (MAR)**  
3. Triggers alert sound when EAR < threshold or MAR > threshold
""")

st.image("demo.jpg", caption="Facial Landmarks Example", use_container_width=True)

st.info("⚠️ Webcam access is disabled on Streamlit Cloud. Please run locally to test the full functionality.")

st.markdown("---")
st.markdown("### 💻 Run Locally")
st.code("""
git clone https://github.com/bhumi-2303/repo.git
cd repo
pip install -r requirements.txt
streamlit run streamlit_app.py
""")
