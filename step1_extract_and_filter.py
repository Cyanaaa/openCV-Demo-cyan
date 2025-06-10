import os
import cv2
import mediapipe as mp
from tqdm import tqdm

input_dir = "videos"
output_dir = "dataset_clean"
os.makedirs(output_dir, exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True)

label_map = {"stand": 0, "sit": 1, "lie": 2}
frame_interval = 1  # 每隔多少帧提取一次

def has_valid_pose(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return results.pose_landmarks is not None

for fname in os.listdir(input_dir):
    label = None
    for key in label_map:
        if key in fname.lower():
            label = key
            break
    if label is None:
        print(f"跳过未识别类别：{fname}")
        continue

    cap = cv2.VideoCapture(os.path.join(input_dir, fname))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved_count = 0
    idx = 0

    print(f"正在处理: {fname}（类别: {label}）")

    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval != 0:
            continue
        if not has_valid_pose(frame):
            continue
        out_name = f"{label}_{fname[:-4]}_{idx}.jpg"
        cv2.imwrite(os.path.join(output_dir, out_name), frame)
        idx += 1
        saved_count += 1

    cap.release()
    print(f"提取完成: {saved_count} 张帧")

print("✅ 所有视频处理完毕！")
