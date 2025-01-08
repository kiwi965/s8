import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

# Parameters
num_classes = 5  # Update based on your model
model_path = "cnn axial 100 epoch 16 batch  tl 0 vl 58 ta 99 va 94.pth"  # Path to the saved model
classes = ["buckle", "displaced", "no fracture", "non_displaced", "segmented"]  # Update with your class names
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

# Function to predict an image
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)

    predicted_class = classes[preds.item()]
    return predicted_class

# Test the model with some images
image_paths = ["buckle 2.jpg", "dis 1.jpg","nd 1.jpg","nf 3.jpg","seg 4.jpg"]  # Update with your test image paths
for image_path in image_paths:
    predicted_class = predict_image(image_path)
    print(f"Image: {image_path}, Predicted Class: {predicted_class}")
