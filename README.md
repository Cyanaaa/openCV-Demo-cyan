# Human Pose Classification with MediaPipe and OpenCV

This project demonstrates how to classify human poses (Standing, Sitting, Lying) using **MediaPipe Pose** and **scikit-learn**, with real-time inference via **OpenCV** and **a webcam or video input**.


## 📌 Features

- ✅ Extract skeleton keypoints using MediaPipe Pose
- ✅ Calculate body joint angles (hip-based)
- ✅ Train a classifier (Random Forest) to distinguish 3 poses:
  - `Stand`
  - `Sit`
  - `Lie`
- ✅ Real-time webcam detection
- ✅ Full visualization of the skeleton and predictions
- ✅ Support for automated frame extraction from labeled videos

---

## 📁 Project Structure

```
project/
├── videos/                    # Raw MP4 videos, one action per file (e.g. sit_1.mp4)
├── dataset_clean/            # Extracted and filtered frames
├── pose_classifier.pkl       # Trained model
├── step1_extract_and_filter.py   # Extract frames + filter invalid skeletons
├── step2_train_model.py          # Train classifier
├── step3_classify_camera.py      # Realtime webcam classification (with skeleton)
```

---

## ⚙️ Setup

1. Create virtual environment:
   ```bash
   conda create -n opencv_pose python=3.10
   conda activate opencv_pose
   ```

2. Install dependencies:
   ```bash
   pip install mediapipe opencv-python scikit-learn tqdm joblib
   ```

---

## 🏃 Usage

### 1. Extract Labeled Frames from Videos

Put your labeled videos into the `videos/` folder. Each filename should include the class: `stand`, `sit`, or `lie`.

```bash
python step1_extract_and_filter.py
```

This will:
- Read each video
- Extract frames every N frames
- Discard frames where no full skeleton is detected
- Save cleaned frames to `dataset_clean/`

---

### 2. Train the Pose Classifier

```bash
python step2_train_model.py
```

- Calculates angles (shoulder-hip-knee) for each image
- Trains a RandomForest classifier
- Saves model as `pose_classifier.pkl`

---

### 3. Run Real-Time Webcam Inference

```bash
python step3_classify_camera.py
```

This will:
- Start your webcam
- Run real-time pose detection
- Display skeleton and predicted label (`Stand`, `Sit`, `Lie`)

---

## 🎓 Educational Goals

This tutorial is designed for students to:
- Understand human pose estimation using skeleton keypoints
- Learn how to extract features (joint angles) for classification
- Build interpretable models for real-world applications (elderly care, fall detection)
- Practice working with OpenCV and MediaPipe in Python

---

## 📷 Example Videos

> You may record 3 short videos demonstrating standing, sitting, and lying to use as training input.

---

## 🔒 License

MIT License

---

## 🙋‍♂️ Author

Developed for teaching purposes by [Your Name], 2025  
Feel free to fork, modify, and use in your own projects!
