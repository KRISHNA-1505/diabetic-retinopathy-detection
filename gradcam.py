import numpy as np
import cv2

# -----------------------------
# Create Synthetic Fundus Image
# -----------------------------
size = 224

# Black background
img = np.zeros((size, size, 3), dtype=np.uint8)

# Draw circular retina
cv2.circle(img, (112,112), 100, (20,20,120), -1)

# Add vessels (random lines)
for i in range(25):
    x1 = np.random.randint(40,180)
    y1 = np.random.randint(40,180)
    x2 = np.random.randint(40,180)
    y2 = np.random.randint(40,180)
    cv2.line(img,(x1,y1),(x2,y2),(0,0,255),1)

# Add optic disc
cv2.circle(img,(150,120),20,(200,200,200),-1)

# -----------------------------
# Create Fake GradCAM Heatmap
# -----------------------------
heatmap = np.zeros((size,size))

for i in range(5):
    x = np.random.randint(70,150)
    y = np.random.randint(70,150)
    cv2.circle(heatmap,(x,y),20,np.random.uniform(0.6,1.0),-1)

heatmap = cv2.GaussianBlur(heatmap,(31,31),0)

# Normalize
heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

# Convert heatmap to color
heatmap_color = cv2.applyColorMap(
    np.uint8(255 * heatmap),
    cv2.COLORMAP_JET
)

# -----------------------------
# Overlay Heatmap
# -----------------------------
overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

# Save images
cv2.imwrite("sample_fundus.jpg", img)
cv2.imwrite("sample_gradcam.jpg", overlay)

print("Sample images generated successfully!")