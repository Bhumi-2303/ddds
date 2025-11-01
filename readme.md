# 🚗 Driver Drowsiness Detection System (DDDS)

A real-time **Driver Drowsiness Detection System** built with **Deep Learning**, **Computer Vision**, and **Streamlit** for interactive monitoring and alerting. The system detects signs of drowsiness using **eye aspect ratio (EAR)** and **mouth aspect ratio (MAR)**, triggering a buzzer alert when fatigue is detected.

---

## 🧠 Features

* 🔍 Real-time face and eye detection using **dlib** and **OpenCV**
* 💤 EAR & MAR calculation to detect drowsiness and yawning
* 🔔 Buzzer alert via **pygame** when driver is drowsy
* 📊 Live monitoring dashboard with:

  * Drowsiness counter
  * Real-time status (Awake / Drowsy)
  * EAR, MAR, and detection summary
* 🌐 Streamlit-based interface for deployment
* 🧩 Supports personalized model training

---

## 🏗️ Project Structure

```
DDDS/
│
├── app.py                      # Main Streamlit application
├── models/
│   ├── mobilenetv2_base.h5     # CNN model for feature extraction
│   └── shape_predictor_68_face_landmarks.dat  # Dlib facial landmark model
├── src/
│   ├── realtime_detection.py
│   ├── realtime_detection_buzzer.py
│   ├── train_model.py
│   ├── personalized_training.py
│   └── buzzer.mp3
├── dataset/
│   ├── train/
│   ├── val/
│   └── test/
├── logs/
│   └── driver_history.csv      # Record of driver activity
├── readme.md                   # Documentation (this file)
├── requirements.txt            # Python dependencies
├── .gitignore
└── download_dataset.sh         # Optional dataset download script
```

---

## ⚙️ Installation

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<Bhumi-2303>/ddds.git
cd ddds
```

### 2️⃣ Create Virtual Environment

```bash
python3 -m venv ddd_env
source ddd_env/bin/activate  # for Linux / macOS
# OR
ddd_env\Scripts\activate     # for Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Application

```bash
streamlit run app.py
```

---

## 🧩 Dataset

Dataset used: [Drowsiness Dataset – Kaggle](https://www.kaggle.com/datasets/hoangtung719/drowsiness-dataset)

* The dataset includes **open-eye, closed-eye, and yawning** images.
* The model was trained on a **MobileNetV2** base with fine-tuned layers.

---

## 🔊 Model Files

Make sure these model files exist in the `models/` directory:

* `mobilenetv2_base.h5`
* `shape_predictor_68_face_landmarks.dat`

If they exceed 100MB (for Streamlit Cloud):
Add auto-download logic in `app.py` using Google Drive / Hugging Face.

---

## ☁️ Deployment (Streamlit Cloud)

1. Push your project to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io).
3. Create a new app → Select repo → Set `app.py` as entry file.
4. Deploy 🚀

If large model files are missing, Streamlit will show a warning message.

---

## 🧠 How It Works

1. Captures frames from webcam.
2. Detects facial landmarks (eyes & mouth).
3. Calculates EAR & MAR values.
4. Triggers buzzer if eyes are closed or yawning persists.
5. Displays real-time drowsiness status on Streamlit dashboard.

---

## 📸 Screenshots (Optional)

* Live dashboard with EAR/MAR visualization
* Status indicators (Awake / Drowsy)
* Alert buzzer in action

---

## 🧰 Tech Stack

* **Python 3.10**
* **OpenCV**
* **dlib**
* **TensorFlow / Keras**
* **Streamlit**
* **pygame**
* **NumPy / imutils**

---

## 🧑‍💻 Contributors

**Bhavini Chauhan**
3rd-year IT Engineering Student
Project: Driver Drowsiness Detection System
GTU — RNGPIT

---

## 📄 License

This project is for educational purposes under the GNU GPL v3 license.

---

### ⭐ Show your support

If you found this project helpful, consider giving it a star ⭐ on GitHub!
