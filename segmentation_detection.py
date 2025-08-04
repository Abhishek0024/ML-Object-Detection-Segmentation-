import torch
import torchvision
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Load Mask R-CNN model
model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Load and preprocess image
image_path = "Test.jpg"  # Your image file
image = Image.open(image_path).convert("RGB")
transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image)

# Run the model
with torch.no_grad():
    prediction = model([image_tensor])[0]

# Convert PIL image to OpenCV BGR
img = np.array(image)
img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

# Confidence threshold
threshold = 0.5

# Loop over detected objects
for i in range(len(prediction["boxes"])):
    score = prediction["scores"][i].item()
    if score > threshold:
        box = prediction["boxes"][i].int().numpy()
        label_id = prediction["labels"][i].item()
        mask = prediction["masks"][i, 0].mul(255).byte().cpu().numpy()

        # Draw bounding box
        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

        # Overlay mask (green channel)
        colored_mask = np.zeros_like(img)
        colored_mask[:, :, 1] = mask
        img = cv2.addWeighted(img, 1, colored_mask, 0.5, 0)

# Convert back to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Show result
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis("off")
plt.title("Object Detection + Segmentation")
plt.show()