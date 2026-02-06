# NeuroMove — Camera-Based Rehab Assessment System

A real-time rehabilitation assessment system that uses **computer vision and AI** to evaluate physiotherapy exercises using **only a camera** — no wearables, no external sensors.

Built for the **Rehab Tech Challenge Hackathon**.

---

## 🚀 What This Project Does

- Tracks human movement in real time using a standard webcam
- Detects and counts rehabilitation exercise repetitions
- Quantifies movement quality using clinical metrics
- Provides personalized coaching-style feedback
- Uses AI to assess whether repetitions are correct or incorrect
- Monitors environmental conditions like lighting and camera distance
- Works with a standalone HTML frontend (GitHub Pages–ready)

---

## 🏋️ Supported Exercises

- Arm Raise (shoulder flexion)
- Sit-to-Stand
- Knee Extension
- Head Rotation (neck mobility)

The system automatically adapts to **left or right side movements** based on landmark confidence.

---

## 🧠 Core Technologies Used

- **Computer Vision**  
  Single-camera human pose estimation for joint tracking

- **Motion Analysis**  
  Range of Motion (ROM), smoothness, and consistency metrics

- **AI / Machine Learning**  
  Lightweight, explainable classifier (Logistic Regression) trained on movement features

- **Environmental Awareness**
  - Relative depth estimation (body scale proxy)
  - Lighting quality detection
  - Detection range warnings

- **Backend**
  - Python + Flask
  - OpenCV for video processing

- **Frontend**
  - HTML / CSS / JavaScript
  - Can be hosted on GitHub Pages

---

## 🧩 System Architecture (High Level)

Camera Input
↓
Pose Estimation
↓
Motion & Quality Analysis
↓
AI + Rule Engine
↓
User Interface

Each module is independent and can be extended without changing the core pipeline.

---

## 📊 Metrics Computed Per Rep

- Range of Motion (ROM)
- Smoothness (control of movement)
- Consistency across repetitions
- Overall score (0–100)
- AI-based correctness assessment

---

## 🌗 Robustness Features

- **Lighting Detection**
  - Warns if the scene is too dark or too bright

- **Relative Depth Estimation**
  - Detects if the user is too close or too far from the camera
  - Uses body proportions (shoulder width proxy)

- **Graceful Degradation**
  - Alerts users instead of silently failing under poor conditions

---

## 🧪 AI Training Workflow

The AI model can be retrained during runtime:

1. Perform good and bad repetitions
2. Label feature data in `backend/ai/train.py`
3. Run:
   ```bash
   python backend/ai/train.py


## Author 

Yatharth Garg

Rehab Tech Hackathon Project
