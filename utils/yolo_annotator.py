import os
import cv2
from ultralytics import YOLO

# Pengaturan model dan kelas target
model = YOLO("yolov8s.pt") 
target_classes = [0, 24, 26, 28]  # person + backpack, suitcase, handbag

# Lokasi folder video
video_dir = "data/aboda"
frame_root = "data/aboda_frames"
label_root = "data/aboda_annotations"

os.makedirs(frame_root, exist_ok=True)
os.makedirs(label_root, exist_ok=True)

# Proses setiap video
for video_name in os.listdir(video_dir):
    if not video_name.endswith(".avi"):
        continue

    video_path = os.path.join(video_dir, video_name)
    base_name = os.path.splitext(video_name)[0]

    img_out_dir = os.path.join(frame_root, base_name)
    label_out_dir = os.path.join(label_root, base_name)

    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(label_out_dir, exist_ok=True)

    # -------- Ekstrak Frame --------
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(img_out_dir, f"frame_{frame_idx:04d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_idx += 1
    cap.release()

    print(f"Extracted {frame_idx} frames from {video_name}")

    # -------- Anotasi dengan YOLO --------
    for filename in sorted(os.listdir(img_out_dir)):
        if not filename.endswith(".jpg"):
            continue
        img_path = os.path.join(img_out_dir, filename)
        results = model(img_path)[0]

        detections = []
        for box in results.boxes:
            cls = int(box.cls[0])
            if cls not in target_classes:
                continue

            xywh = box.xywh[0].cpu().numpy()  # [x_center, y_center, width, height]
            w, h = results.orig_shape[1], results.orig_shape[0]
            norm_box = [0] + [round(float(val / dim), 6) for val, dim in zip(xywh, [w, h, w, h])]
            detections.append(norm_box)

        out_name = filename.replace(".jpg", ".txt")
        out_path = os.path.join(label_out_dir, out_name)
        with open(out_path, "w") as f:
            for det in detections:
                f.write(" ".join(map(str, det)) + "\n")

    print(f"Annotated {base_name} → {label_out_dir}")
