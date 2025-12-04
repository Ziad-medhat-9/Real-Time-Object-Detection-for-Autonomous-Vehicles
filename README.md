# 🚦 Automated Traffic Sign Recognition System

A Deep Learning--Powered Traffic Sign Classification & Detection Project

Demo link : https://autonomous-vehicle-deploy.streamlit.app 
## 📌 Overview

The **Automated Traffic Sign Recognition System** is an end-to-end deep
learning solution designed to **classify** and **detect** traffic signs
from images. It is built for applications such as:

-   Autonomous driving\
-   Advanced Driver Assistance Systems (ADAS)\
-   Smart transportation systems\
-   Road monitoring and safety analytics

The project uses **Python**, **TensorFlow/Keras**, and includes a
modern, user-friendly **Streamlit interface** for real-time predictions.
Optional cloud deployment is supported through **Microsoft Azure**.

## 🧠 Features

### ✔️ Traffic Sign Classification

Predict the category of a traffic sign using CNN architectures or
pretrained models such as **VGG16**, **ResNet**, or **MobileNet**.

### ✔️ Traffic Sign Detection

Detect and locate multiple traffic signs in an image using object
detection models such as **YOLO**, **SSD**, or **Faster R-CNN**.

### ✔️ Streamlit Web Interface

A clean and interactive interface that allows users to:

-   Upload traffic sign images\
-   View classification predictions\
-   Visualize detection bounding boxes\
-   Display confidence scores and model outputs

### ✔️ Preprocessing Pipeline

-   Image resizing and normalization\
-   Data augmentation (rotation, flip, brightness jittering)\
-   Train/Validation/Test splitting\
-   Automatic class balancing

### ✔️ Model Evaluation Tools

Includes visual and numerical metrics such as:

-   Accuracy\
-   Precision, Recall, F1-Score\
-   Confusion Matrix\
-   IoU and mAP for object detection

## 🚀 Streamlit App

### ▶️ Run the App

``` bash
streamlit run streamlit_app/app.py
```

### 🖥️ App Capabilities

-   Upload an image and receive prediction results immediately\
-   View bounding boxes for detection models\
-   Access model confidence scores\
-   Smooth and responsive UI

## 🧪 Model Development

### 🔹 Classification Models

-   Custom CNN\
-   Transfer learning using VGG16, ResNet50, MobileNet, etc.\
-   Fine-tuned architectures for optimal performance

### 🔹 Detection Models

-   YOLO (v5/v8)\
-   SSD\
-   Faster R-CNN\
-   Anchor box optimization and fine-tuning for traffic sign shapes

## 🧰 Technologies Used

  Category               Tools
  ---------------------- ------------------------------------------------
  Frameworks             TensorFlow, Keras, PyTorch (optional)
  UI                     Streamlit
  Cloud Deployment       Azure ML, Azure Container Instances (optional)
  Image Processing       OpenCV, Pillow
  Programming Language   Python

## 📊 Evaluation

### 🔸 Classification Metrics

-   Accuracy\
-   Precision\
-   Recall\
-   F1-score\
-   Confusion matrix visualization

### 🔸 Detection Metrics

-   Mean Average Precision (mAP)\
-   Intersection-over-Union (IoU)\
-   Detection confidence thresholds

## ☁️ Optional Cloud Deployment

The system can be deployed using **Microsoft Azure**, including:

-   Azure Machine Learning for model hosting\
-   Azure Cognitive Services for easy scaling\
-   Docker-based container deployments\
-   REST API endpoints for real-time inference

## 🛠️ Installation

### 1. Clone the Repository

``` bash
git clone https://github.com/your-username/traffic-sign-recognition.git
cd traffic-sign-recognition
```

### 2. Install Dependencies

``` bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App

``` bash
streamlit run streamlit_app/app.py
```

## 📄 License

MIT License (or your chosen license)

## 🤝 Contributing

Contributions are welcome!\
Feel free to open issues, submit pull requests, or suggest enhancements.

## ⭐ Support

If you find this project helpful, consider giving it a **star ⭐ on
GitHub**!
