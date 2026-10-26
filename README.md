# PneumoVision-AI

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/PyTorch%2FTensorFlow-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![Interface](https://img.shields.io/badge/UI-Gradio-FF5722.svg)](https://gradio.app/)
[![Deployment](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow)](https://huggingface.co/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**PneumoVision-AI** is an intelligent, deep learning-powered medical image classification application designed to detect pneumonia from chest X-ray images in real time. Built with computer vision models and an intuitive web interface, PneumoVision-AI offers instant diagnostic assistance, categorizing scans as either **Normal** or **Pneumonic** along with actionable visual feedback.

---

## 🖼️ Application Preview

| **Pneumonia Detected** | **Normal Chest Scan** |
| :---: | :---: |
| <img src="https://github.com/user-attachments/assets/e8b6960a-d9b8-4264-b4a4-434a2bdb407a" width="400" alt="Pneumonia Detection Output" /> | <img src="https://github.com/user-attachments/assets/9783e507-9d95-493f-85ec-5ee65dee5b58" width="400" alt="Normal Scan Output" /> |

---

## ✨ Key Features

* **Instant X-Ray Classification**: Classifies uploaded chest X-ray images (`.png`, `.jpg`, `.jpeg`, `.webp`) in seconds.
* **Deep Learning Engine**: Leverages convolutional neural networks (CNNs) optimized for medical imaging feature extraction.
* **Visual Diagnostic Feedback**: Generates dynamic visual output indicating test results (**Normal** vs. **Pneumonic**).
* **User-Friendly Web Interface**: Clean, responsive UI built with Gradio for effortless image upload and inference visualization.
* **Multi-Platform Access**: Integrated quick links to Hugging Face Spaces live demo, video demonstrations, and developer social links.

---

## 🛠️ Tech Stack

* **Core Language:** Python 3.9+
* **Deep Learning Framework:** PyTorch / TensorFlow & Keras
* **Image Processing:** OpenCV, PIL (Pillow), NumPy
* **Frontend / UI:** Gradio
* **Hosting / Deployment:** Hugging Face Spaces

---

## 📂 Project Structure

```text
PneumoVision-AI/
├── assets/                  # Documentation images and screenshots
├── models/                  # Trained neural network weights (.pth / .h5)
│   └── pneumonia_detector.pth
├── src/                     # Core application source code
│   ├── preprocess.py        # Image normalization and pre-processing
│   ├── predict.py           # Model inference engine
│   └── utils.py             # Result card generation and plotting
├── app.py                   # Main Gradio web application script
├── requirements.txt         # Project dependencies
├── LICENSE                  # MIT License
└── README.md                # Project documentation
