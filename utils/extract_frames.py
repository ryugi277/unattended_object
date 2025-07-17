import cv2
import os

def extract_frames(video_path, output_dir, skip_frames=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % skip_frames == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            filepath = os.path.join(output_dir, filename)
            cv2.imwrite(filepath, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"Extracted {saved_count} frames from {os.path.basename(video_path)} to {output_dir}")
