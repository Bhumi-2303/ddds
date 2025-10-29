#!/bin/bash

# ============================================
# 🚗 Driver Drowsiness Detection Dataset Setup
# ============================================

echo "🚀 Starting dataset download..."

# Check if Kaggle CLI is installed
if ! command -v kaggle &> /dev/null
then
    echo "❌ Kaggle CLI not found!"
    echo "Please install it first using:"
    echo "    pip install kaggle"
    echo "Then place your Kaggle API key at:"
    echo "    ~/.kaggle/kaggle.json"
    exit 1
fi

# Create the dataset directory
mkdir -p dataset

# Download the dataset from Kaggle
echo "📦 Downloading dataset from Kaggle..."
kaggle datasets download -d hoangtung719/drowsiness-dataset -p dataset --unzip

# Unzip if not automatically unzipped
if [ -f "dataset/drowsiness-dataset.zip" ]; then
    echo "🗜️ Unzipping dataset..."
    unzip dataset/drowsiness-dataset.zip -d dataset
    rm dataset/drowsiness-dataset.zip
fi

echo "✅ Dataset setup completed!"
echo "Your dataset is now available in: ./dataset/"
