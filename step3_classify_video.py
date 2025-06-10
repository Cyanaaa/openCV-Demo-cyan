import cv2
import mediapipe as mp
import numpy as np
import joblib

# 加载模型和标签
clf = joblib.load("pose_classifier.pkl")
labels = ["Stand", "Sit", "Lie"]

# 初始化摄像头
cap = cv2.VideoCapture(0)

# 初始化 MediaPipe Pose 和绘图模块
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False)

def extract_pose_angles(image, draw=False):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None

    if draw:
        # 在图像上绘制骨架
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),
        )

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

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # 提取角度 & 绘制骨架
    angles = extract_pose_angles(frame, draw=True)

    # 预测姿态
    if angles:
        pred = clf.predict([angles])[0]
        label = labels[pred]
        cv2.putText(frame, f"Pose: {label}", (30, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # 显示图像
    cv2.imshow("Pose Detection (with Skeleton)", cv2.resize(frame, (640, 360)))

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
