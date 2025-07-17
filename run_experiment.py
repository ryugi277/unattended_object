import os
from utils.extract_frames import extract_frames
from utils.yolo_annotator import annotate_folder_with_yolo

video_dir = r"data\aboda"
output_base = r"data\aboda_frames"
skip_frames = 10

# Buat folder output jika belum ada
if not os.path.exists(output_base):
    os.makedirs(output_base)

# Ambil semua file .avi
for filename in os.listdir(video_dir):
    if filename.endswith(".avi"):
        video_path = os.path.join(video_dir, filename)
        video_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_base, video_name)

        print(f"[INFO] Ekstraksi {filename} ...")
        extract_frames(video_path, output_dir, skip_frames=skip_frames)

annotation_base = r"data\aboda_annotations"
model_name = "yolov8s.pt"

for video_name in os.listdir(output_base):
    frame_dir = os.path.join(output_base, video_name)
    annot_dir = os.path.join(annotation_base, video_name)

    print(f"[INFO] Anotasi YOLOv8 untuk {video_name} ...")
    annotate_folder_with_yolo(
        img_folder=frame_dir,
        output_txt_folder=annot_dir,
        model_name=model_name,
        target_classes=[0, 24, 26, 28]
    )