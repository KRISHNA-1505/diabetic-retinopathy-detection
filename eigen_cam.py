import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms

from src.model import get_swin_model

from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = get_swin_model(num_classes=5)
model.load_state_dict(
    torch.load("models/swin_dr.pth", map_location=DEVICE)
)
model.to(DEVICE)
model.eval()

# -----------------------------
# TARGET LAYER (Best for Swin)
# -----------------------------
target_layer = model.patch_embed.proj

# -----------------------------
# IMAGE PATH
# -----------------------------
import os

sample_image = os.listdir("data/train/2")[0]
img_path = os.path.join("data/train/2", sample_image)

print("Using image:", img_path)

# -----------------------------
# PREPROCESS
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

img = Image.open(img_path).convert("RGB")
img_tensor = transform(img).unsqueeze(0).to(DEVICE)

rgb_img = np.array(img.resize((224,224))) / 255.0

# -----------------------------
# EIGEN-CAM
# -----------------------------
cam = EigenCAM(
    model=model,
    target_layers=[target_layer]
)

grayscale_cam = cam(
    input_tensor=img_tensor
)[0]

# -----------------------------
# OVERLAY
# -----------------------------
visualization = show_cam_on_image(
    rgb_img,
    grayscale_cam,
    use_rgb=True
)

cv2.imwrite("eigen_cam_output.jpg", visualization)

print("Eigen-CAM saved as eigen_cam_output.jpg ✅")