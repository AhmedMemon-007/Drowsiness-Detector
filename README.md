# Driver Drowsiness Detection

This project is a **real-time driver drowsiness detection system** using computer vision and eye aspect ratio (EAR) analysis.  
It uses **OpenCV**, **MediaPipe**, and **NumPy** to track facial landmarks, calculate PERCLOS (Percentage of Eye Closure), and trigger alerts when the driver is likely drowsy.

---

## 🚀 Features
- Real-time face and eye tracking using **MediaPipe FaceMesh**
- Computes **Eye Aspect Ratio (EAR)** and **PERCLOS**
- Plays a **beep alert** if drowsiness is detected
- Displays live EAR and PERCLOS values on screen
- Works with your webcam

---

## 📦 Requirements
Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
