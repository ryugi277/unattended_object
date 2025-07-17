import os
from glob import glob
from collections import defaultdict
from sklearn.metrics import precision_score, recall_score, f1_score

# Path folder
GT_DIR = "data/aboda_annotations/video9"
PRED_DIR = "results/predictions/video9"

# Ambil semua frame ground truth yang tersedia
gt_files = sorted(glob(os.path.join(GT_DIR, "frame_*.txt")))
frame_names = [os.path.basename(f) for f in gt_files]

# Simpan hasil untuk semua frame
y_true_all = []
y_pred_all = []

for fname in frame_names:
    gt_path = os.path.join(GT_DIR, fname)
    pred_path = os.path.join(PRED_DIR, fname)

    # Ambil GT boxes
    with open(gt_path, "r") as f:
        gt_lines = f.readlines()
    gt_labels = [line.strip().split()[0] for line in gt_lines]

    # Ambil prediksi (jika file ada, jika tidak = kosong)
    if os.path.exists(pred_path):
        with open(pred_path, "r") as f:
            pred_lines = f.readlines()
        pred_labels = [line.strip().split()[0] for line in pred_lines]
    else:
        pred_labels = []

    # Konversi ke array biner: '0' = tas (positif), lainnya abaikan
    y_true = [1 if label == '0' else 0 for label in gt_labels]
    y_pred = [1 if label == '0' else 0 for label in pred_labels]

    # Pad biar panjangnya sama
    max_len = max(len(y_true), len(y_pred))
    y_true += [0] * (max_len - len(y_true))
    y_pred += [0] * (max_len - len(y_pred))

    y_true_all.extend(y_true)
    y_pred_all.extend(y_pred)

# Hitung metrik
precision = precision_score(y_true_all, y_pred_all, zero_division=0)
recall = recall_score(y_true_all, y_pred_all, zero_division=0)
f1 = f1_score(y_true_all, y_pred_all, zero_division=0)

print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")