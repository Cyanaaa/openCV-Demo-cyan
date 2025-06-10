import os
import cv2
import numpy as np
import mediapipe as mp
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tqdm import tqdm
import joblib

data_dir = "dataset_clean"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

label_map = {"stand": 0, "sit": 1, "lie": 2}
X, y = [], []

def extract_pose_angles(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.pose_landmarks:
        return None

    landmarks = results.pose_landmarks.landmark
    keypoints = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])

    def angle(a, b, c):
        ab = a - b
        cb = c - b
        cosine = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
        return np.arccos(np.clip(cosine, -1.0, 1.0))

    try:
        left_angle = angle(keypoints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                           keypoints[mp_pose.PoseLandmark.LEFT_HIP.value],
                           keypoints[mp_pose.PoseLandmark.LEFT_KNEE.value])
        right_angle = angle(keypoints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                            keypoints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                            keypoints[mp_pose.PoseLandmark.RIGHT_KNEE.value])
        return [left_angle, right_angle]
    except:
        return None

files = os.listdir(data_dir)
print("提取姿态角中...")

for fname in tqdm(files):
    img = cv2.imread(os.path.join(data_dir, fname))
    angles = extract_pose_angles(img)
    if angles is None:
        continue
    for label in label_map:
        if label in fname:
            X.append(angles)
            y.append(label_map[label])
            break

X = np.array(X)
y = np.array(y)
print(f"有效样本数: {len(X)}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("分类报告:")
print(classification_report(y_test, y_pred))

joblib.dump(clf, "pose_classifier.pkl")
print("✅ 模型保存为 pose_classifier.pkl")
