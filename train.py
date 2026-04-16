import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.model import get_swin_model

import os
import copy
from collections import Counter

# -----------------------------
# DEVICE
# -----------------------------
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# -----------------------------
# PARAMETERS
# -----------------------------
BATCH_SIZE = 4
EPOCHS = 15
LR = 3e-5
PATIENCE = 5

# -----------------------------
# DATA AUGMENTATION
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

# -----------------------------
# DATASETS
# -----------------------------
train_data = datasets.ImageFolder(
    "data/train",
    transform=train_transform
)

val_data = datasets.ImageFolder(
    "data/val",
    transform=val_transform
)

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE
)

print(f"Train Images: {len(train_data)}")
print(f"Val Images: {len(val_data)}")

# -----------------------------
# CLASS WEIGHTS
# -----------------------------
labels = [label for _, label in train_data.samples]
class_counts = Counter(labels)

print("\nClass Counts:", class_counts)

total_samples = sum(class_counts.values())

class_weights = [
    total_samples / class_counts[i]
    for i in range(len(class_counts))
]

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)

print("Class Weights:", class_weights)

# -----------------------------
# FOCAL LOSS + LABEL SMOOTHING
# -----------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, smoothing=0.1):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):

        log_probs = torch.nn.functional.log_softmax(inputs, dim=1)
        probs = torch.exp(log_probs)

        targets_onehot = torch.zeros_like(inputs)
        targets_onehot.scatter_(1, targets.unsqueeze(1), 1)

        # Label smoothing
        targets_onehot = targets_onehot * (1 - self.smoothing) + self.smoothing / inputs.size(1)

        ce_loss = -targets_onehot * log_probs
        focal_weight = (1 - probs) ** self.gamma

        loss = focal_weight * ce_loss

        if self.alpha is not None:
            alpha = self.alpha.unsqueeze(0)
            loss = alpha * loss

        return loss.sum(dim=1).mean()

# -----------------------------
# MODEL
# -----------------------------
model = get_swin_model(num_classes=5)
model = model.to(DEVICE)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
criterion = FocalLoss(alpha=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

# -----------------------------
# LR SCHEDULER
# -----------------------------
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS
)

# -----------------------------
# TRAINING LOOP
# -----------------------------
best_acc = 0
best_model = copy.deepcopy(model.state_dict())
trigger = 0

for epoch in range(EPOCHS):

    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    # ---- TRAIN ----
    model.train()
    running_loss = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    scheduler.step()

    avg_loss = running_loss / len(train_loader)
    print(f"Training Loss: {avg_loss:.4f}")

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    # ---- BEST MODEL SAVE ----
    if accuracy > best_acc:
        best_acc = accuracy
        best_model = copy.deepcopy(model.state_dict())
        trigger = 0
        print("🔥 New Best Model Saved")

    else:
        trigger += 1

    # ---- EARLY STOPPING ----
    if trigger >= PATIENCE:
        print("⛔ Early stopping triggered")
        break

# -----------------------------
# SAVE BEST MODEL
# -----------------------------
os.makedirs("models", exist_ok=True)

torch.save(
    best_model,
    "models/swin_best_dr.pth"
)

print(f"\nBest Accuracy: {best_acc:.2f}%")
print("Best model saved as swin_best_dr.pth ✅")