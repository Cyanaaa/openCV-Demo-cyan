import cv2
import mediapipe as mp
import numpy as np
import joblib

video_path = "test_video.mp4"  # 替换成你的视频路径
clf = joblib.load("pose_classifier.pkl")
labels = ["Stand", "Sit", "Lie"]

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False)

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

cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    angles = extract_pose_angles(frame)
    if angles:
        pred = clf.predict([angles])[0]
        label = labels[pred]
        cv2.putText(frame, f"{label}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 2)

    cv2.imshow("Action Recognition", cv2.resize(frame, (640, 360)))
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
