import os
import shutil
import pandas as pd

BASE_DIR = "aptos2019-blindness-detection"
CSV_PATH = os.path.join(BASE_DIR, "train.csv")
IMAGE_DIR = os.path.join(BASE_DIR, "train_images")
OUTPUT_DIR = "data/train"


# Safety checks
if not os.path.exists(CSV_PATH):
    print("train.csv not found ❌")
    exit()

df = pd.read_csv(CSV_PATH)

print("CSV Loaded ✅")
print("Total Images:", len(df))

for _, row in df.iterrows():

    img_name = row["id_code"] + ".png"
    label = str(row["diagnosis"])

    src = os.path.join(IMAGE_DIR, img_name)
    dst = os.path.join(OUTPUT_DIR, label, img_name)

    if os.path.exists(src):
        shutil.copy(src, dst)

print("Dataset split completed ✅")
