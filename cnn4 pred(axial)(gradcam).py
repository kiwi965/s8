import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Parameters
num_classes = 5  # Update based on your model
model_path = "cnn axial 100 epoch 16 batch  tl 0 vl 58 ta 99 va 94.pth"  # Path to the saved model
classes = ["buckle", "displaced", "no fracture", "non_displaced", "segmented"]  # Update with your class names
image_folder = "axial test"  # Path to the folder containing test images
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transformations for the test images
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Load the model
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# Grad-CAM functions
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

        # Hook for gradients
        target_layer.register_backward_hook(self.save_gradient)

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate_heatmap(self, input_tensor, class_index):
        activations = None

        # Hook for activations
        def forward_hook(module, input, output):
            nonlocal activations
            activations = output

        handle = self.target_layer.register_forward_hook(forward_hook)

        # Forward pass
        outputs = self.model(input_tensor)
        handle.remove()

        # Backward pass for the target class
        self.model.zero_grad()
        one_hot_output = torch.zeros_like(outputs)
        one_hot_output[0][class_index] = 1
        outputs.backward(gradient=one_hot_output)

        # Grad-CAM calculation
        gradients = self.gradients.cpu().data.numpy()
        activations = activations.cpu().data.numpy()
        weights = np.mean(gradients, axis=(2, 3))  # Global average pooling
        heatmap = np.sum(weights[:, :, np.newaxis, np.newaxis] * activations, axis=1)[0]
        heatmap = np.maximum(heatmap, 0)  # ReLU
        heatmap /= np.max(heatmap)  # Normalize to [0, 1]

        return heatmap

# Function to overlay heatmap on image
def overlay_heatmap(image, heatmap):
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize(image.size, resample=Image.BILINEAR)
    heatmap = np.asarray(heatmap)
    heatmap = plt.cm.jet(heatmap)[:, :, :3]  # Convert to RGB colormap
    heatmap = (heatmap * 255).astype(np.uint8)

    overlayed = Image.blend(image.convert("RGBA"), Image.fromarray(heatmap, mode="RGBA"), alpha=0.5)
    return overlayed

# Test the model and generate Grad-CAM heatmaps
target_layer = model.layer4[1].conv2  # Target the last convolutional layer
grad_cam = GradCAM(model, target_layer)

image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]
for image_path in image_paths:
    image = Image.open(image_path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    outputs = model(input_tensor)
    _, preds = torch.max(outputs, 1)
    predicted_class = classes[preds.item()]
    print(f"Image: {image_path}, Predicted Class: {predicted_class}")

    # Grad-CAM heatmap
    heatmap = grad_cam.generate_heatmap(input_tensor, preds.item())
    overlayed_image = overlay_heatmap(image, heatmap)

    # Save or display the heatmap
    plt.imshow(overlayed_image)
    plt.title(f"Class: {predicted_class}")
    plt.axis("off")
    plt.show()
