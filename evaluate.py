import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import get_swin_model

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# DATASET
# -----------------------------
val_data = datasets.ImageFolder(
    "data/val",
    transform=transform
)

val_loader = DataLoader(
    val_data,
    batch_size=4,
    shuffle=False
)

class_names = val_data.classes
num_classes = len(class_names)

print("Classes:", class_names)

# -----------------------------
# LOAD MODEL
# -----------------------------
model = get_swin_model(num_classes=num_classes)

model.load_state_dict(
    torch.load("models/swin_best_dr.pth", map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# -----------------------------
# PREDICTIONS
# -----------------------------
all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for images, labels in val_loader:

        images = images.to(DEVICE)

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs.cpu().numpy())

# Convert to numpy
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(8,6))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix — Swin DR")

plt.savefig("confusion_matrix.png")
plt.show()

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\nClassification Report:\n")

report = classification_report(
    all_labels,
    all_preds,
    target_names=class_names
)

print(report)

# -----------------------------
# ROC CURVE
# -----------------------------
labels_bin = label_binarize(all_labels, classes=list(range(num_classes)))

plt.figure()

for i in range(num_classes):

    fpr, tpr, _ = roc_curve(labels_bin[:, i], all_probs[:, i])
    roc_auc = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{class_names[i]} (AUC = {roc_auc:.2f})"
    )

plt.plot([0,1],[0,1],'k--')

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Diabetic Retinopathy")

plt.legend()

plt.savefig("roc_curve.png")
plt.show()