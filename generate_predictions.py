#ga dipake lagi yaa, gosah di run

# generate_predictions.py
import os
import cv2
from ultralytics import YOLO

model = YOLO("yolov8m.pt")  # atau yolov8s.pt
video_path = "data/aboda/video9.avi"
output_dir = "results/predictions/video9"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)[0]

    output_path = os.path.join(output_dir, f"frame_{frame_idx:04d}.txt")
    with open(output_path, "w") as f:
        for box in results.boxes:
            cls = int(box.cls[0])
            xc, yc, w, h = map(float, box.xywhn[0])  # Normalized
            f.write(f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    frame_idx += 1

cap.release()
print(f"Done writing predictions to {output_dir}")