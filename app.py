import streamlit as st
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms

from src.model import get_swin_model
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="DR Detection",
    page_icon="🩺",
    layout="centered"
)

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    model = get_swin_model(num_classes=5)
    model.load_state_dict(
        torch.load("models/swin_best_dr.pth", map_location=DEVICE)
    )
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# -----------------------------
# CLASS LABELS
# -----------------------------
class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

# -----------------------------
# TITLE
# -----------------------------
st.title("🩺 Diabetic Retinopathy Detection")

st.markdown("""
Upload a retinal fundus image to predict the severity of diabetic retinopathy 
using a Swin Transformer model.
""")

# -----------------------------
# RESET BUTTON


# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# PROCESS
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(image, caption="Uploaded Image", width="stretch")

    # -------------------------
    # PREDICT BUTTON
    # -------------------------
    if st.button("🔍 Predict"):

        transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
        ])

        img_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # -------------------------
        # PREDICTION (NO ANIMATION)
        # -------------------------
        with st.spinner("Analyzing retinal image..."):

            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)

                pred_class = torch.argmax(probs).item()
                confidence = probs[0][pred_class].item()

        # -------------------------
        # RESULT SECTION
        # -------------------------
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Uploaded Image")
            st.image(image, width="stretch")

        with col2:
            st.subheader("Prediction Result")

            severity_colors = {
                0: "green",
                1: "yellow",
                2: "orange",
                3: "red",
                4: "darkred"
            }

            st.markdown(
                f"<h2 style='color:{severity_colors[pred_class]}'>"
                f"{class_names[pred_class]}</h2>",
                unsafe_allow_html=True
            )

            st.metric("Confidence", f"{confidence*100:.2f}%")
            st.progress(int(confidence * 100))

            if confidence < 0.6:
                st.warning("Low confidence prediction. Please consult a doctor.")

        # -------------------------
        # TOP-3 PREDICTIONS
        # -------------------------
        st.subheader("Top Predictions")

        top3 = torch.topk(probs[0], 3)

        for i in range(3):
            st.write(
                f"{class_names[top3.indices[i]]}: "
                f"{top3.values[i]*100:.2f}%"
            )

        # -------------------------
        # PROBABILITY BAR CHART
        # -------------------------
        probs_np = probs[0].cpu().numpy()

        df = pd.DataFrame({
            "Class": class_names,
            "Probability": probs_np
        }).sort_values(by="Probability", ascending=False)

        st.subheader("Prediction Probabilities")
        st.bar_chart(df.set_index("Class"))

        # -------------------------
        # EIGEN-CAM
        # -------------------------
        target_layer = model.patch_embed.proj

        cam = EigenCAM(
            model=model,
            target_layers=[target_layer]
        )

        rgb_img = np.array(image.resize((224,224))) / 255.0

        grayscale_cam = cam(input_tensor=img_tensor)[0]

        cam_image = show_cam_on_image(
            rgb_img,
            grayscale_cam,
            use_rgb=True
        )

        st.subheader("Model Explanation (Eigen-CAM)")
        st.image(cam_image, width="stretch")

# -----------------------------
# INFO SECTION
# -----------------------------
st.markdown("---")

st.markdown("""
### DR Severity Levels
- **No DR**: No signs of diabetic retinopathy  
- **Mild**: Microaneurysms  
- **Moderate**: More lesions present  
- **Severe**: Extensive hemorrhages  
- **Proliferative**: Advanced stage  
""")

st.markdown("Developed using Swin Transformer | APTOS Dataset")