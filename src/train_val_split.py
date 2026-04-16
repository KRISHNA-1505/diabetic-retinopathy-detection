import os
import shutil
import random

SOURCE = "data/train"
VAL = "data/val"

os.makedirs(VAL, exist_ok=True)

for cls in os.listdir(SOURCE):

    os.makedirs(os.path.join(VAL, cls), exist_ok=True)

    images = os.listdir(os.path.join(SOURCE, cls))
    random.shuffle(images)

    split = int(0.2 * len(images))  # 80/20 split

    val_imgs = images[:split]

    for img in val_imgs:
        src = os.path.join(SOURCE, cls, img)
        dst = os.path.join(VAL, cls, img)
        shutil.move(src, dst)

print("Train/Val split completed ✅")
