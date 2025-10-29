# 🚗 Driver Drowsiness Detection System (DDDS)

An AI-powered deep learning system that detects **driver fatigue and drowsiness** using **MobileNetV2** and **Computer Vision** techniques.  
Developed as part of the **Design Engineering Project (3150001)** under **Gujarat Technological University (GTU)** at **R.N.G. Patel Institute of Technology (RNGPIT)**.

---

## 🧠 Overview

Driver drowsiness is a major cause of road accidents.  
This project monitors a driver's facial expressions in real time and identifies signs of fatigue such as **eye closure** or **yawning**.

If the model detects drowsiness, it can be configured to alert the driver immediately — helping to prevent accidents and save lives.

---

## 🏗️ System Workflow

1. **Dataset Preparation**  
   - Dataset divided into `train`, `val`, and `test` sets.
   - Each contains 4 subcategories:  
     `open_eyes`, `closed_eyes`, `yawn`, `no_yawn`.

2. **Model Training (Base Model)**  
   - MobileNetV2 (transfer learning) trained on Kaggle dataset.

3. **Personalization**  
   - The base model is fine-tuned using a few driver-specific images for better accuracy.

4. **Real-Time Detection**  
   - The trained model is used with OpenCV to detect drowsiness from a webcam feed in real time.

---

## 🗂️ Project Structure

DDDS/
│
├── dataset/
│ ├── train/
│ ├── val/
│ └── test/
│
├── personalized_data/
│ ├── open_eyes/
│ ├── closed_eyes/
│ ├── yawn/
│ └── no_yawn/
│
├── models/
│ ├── mobilenetv2_base.h5
│ └── personalized_model.h5
│
├── src/
│ ├── train_model.py
│ ├── personalized_training.py
│ ├── realtime_detection.py
│ └── utils.py
│
├── requirements.txt
├── download_dataset.sh
├── .gitignore
└── README.md


---

## ⚙️ Requirements

| Tool | Recommended Version |
|------|----------------------|
| **Python** | 3.10.x |
| **TensorFlow** | ≥ 2.17.0 |
| **Keras** | ≥ 3.4.0 |
| **OpenCV** | ≥ 4.9.0 |
| **NumPy** | ≥ 1.26 |
| **scikit-learn** | ≥ 1.5 |

Install all dependencies with:
```bash
pip install -r requirements.txt

📦 Dataset
📘 Dataset Source

This project uses the Drowsiness Detection Dataset from Kaggle:

https://www.kaggle.com/datasets/hoangtung719/drowsiness-dataset

It contains four categories:

open_eyes

closed_eyes

yawn

no_yawn

Each image is labeled and divided into train, val, and test folders.

⚡ Manual Setup (if you already have the dataset)

Simply place your extracted dataset folder inside the project like this:

DDDS/dataset/
    train/
    val/
    test/

🧩 Automatic Download (for new users)

If the dataset is not present, you can automatically download it from Kaggle using the provided script:

bash download_dataset.sh

The script will:

Download the dataset from Kaggle

Unzip it

Organize it inside the dataset/ folder

🧩 Model Training

To train the MobileNetV2 base model on your dataset:

cd src
python train_model.py

After training, the model will be saved as:

models/mobilenetv2_base.h5

👤 Personalized Model (Driver-Specific Fine-Tuning)

You can fine-tune the model for a specific driver using personal images.

Create a new folder:

DDDS/personalized_data/


Add your own photos in these 4 categories:

open_eyes/

closed_eyes/

yawn/

no_yawn/

Run:

python personalized_training.py

After training, a new model will be saved as:

models/personalized_model.h5

🎥 Real-Time Detection

Once the model is ready, you can test it live using your webcam:

python realtime_detection.py

A window will appear showing:

The driver’s live face

The detected state (e.g., "Closed Eyes", "Yawn")

Press Q to quit.

⚡ Features

Real-time detection using OpenCV

MobileNetV2 backbone for lightweight performance

Transfer learning for quick training

Personalized fine-tuning for specific drivers

Compatible with Raspberry Pi (TensorFlow Lite)

🔮 Future Scope

Eye blink and yawn frequency tracking

Automatic alarm/buzzer alerts (GPIO on Raspberry Pi)

Cloud analytics for fleet monitoring

Mobile app interface for safety stats
