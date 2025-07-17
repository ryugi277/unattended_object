import cv2
from ultralytics import YOLO
import os

def detect_unattended(video_path, output_path):
    model = YOLO("yolov8m.pt")  # ganti dengan yolov8s.pt jika ingin model yang lebih ringan
    cap = cv2.VideoCapture(video_path)

    # Konfigurasi output video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    unattended_counters = {}
    threshold = 10  # jumlah frame sebelum dinyatakan unattended

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        persons, bags = [], []

        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[cls]

            # Visualisasi: warna tas kuning, lainnya biru
            color = (0, 255, 255) if cls in [24, 26, 28] else (255, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            if cls == 0:
                persons.append([x1, y1, x2, y2])
            elif cls in [24, 26, 28]:
                bags.append([x1, y1, x2, y2])

        # Cek setiap tas, apakah jauh dari manusia
        for bag in bags:
            is_near_person = any(iou(bag, p) > 0.2 for p in persons)
            bag_id = tuple(map(int, bag))

            if not is_near_person:
                unattended_counters[bag_id] = unattended_counters.get(bag_id, 0) + 1
            else:
                unattended_counters[bag_id] = 0

            if unattended_counters[bag_id] > threshold:
                x1, y1, x2, y2 = bag
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "UNATTENDED", (x1, y1 - 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()


def iou(box1, box2):
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0