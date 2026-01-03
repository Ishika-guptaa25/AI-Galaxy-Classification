import os
import shutil
import pandas as pd

RAW_DIR = "../data/raw"
TRAIN_DIR = "../data/train"

IMG_DIR = os.path.join(RAW_DIR, "Train_images")
CSV_PATH = os.path.join(RAW_DIR, "train_label.csv")

LABEL_MAP = {
    "Class1": "spiral",
    "Class2": "elliptical",
    "Class3": "irregular",
    "Other": "lenticular"

}

os.makedirs(TRAIN_DIR, exist_ok=True)
for cls in CLASSES.values():
    os.makedirs(os.path.join(TRAIN_DIR, cls), exist_ok=True)

df = pd.read_csv(CSV_PATH, header=None, names=["image_id", "label"])

for _, row in df.iterrows():
    img_name = f"{row['image_id']}.jpg"
    class_name = LABEL_MAP[row['label']]

    src = os.path.join(IMG_DIR, img_name)
    dst = os.path.join(TRAIN_DIR, label, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("✅ Dataset prepared successfully!")

