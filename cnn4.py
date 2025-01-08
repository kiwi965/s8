import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
# Parameters
num_classes = 5  # Update based on your dataset
batch_size = 16
epochs = 100
learning_rate = 0.001
data_dir = "trainval(sagittal)"  # Update this to your dataset path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define transformations
transform = {
    "train": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    "valid": transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
}

# Load datasets
data = {
    "train": datasets.ImageFolder(root=f"{data_dir}/train", transform=transform["train"]),
    "valid": datasets.ImageFolder(root=f"{data_dir}/valid", transform=transform["valid"]),
}

# Create dataloaders
dataloaders = {
    "train": DataLoader(data["train"], batch_size=batch_size, shuffle=True),
    "valid": DataLoader(data["valid"], batch_size=batch_size, shuffle=False),
}

# Initialize model
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training and validation
for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")
    print("-" * 30)

    train_loss = 0.0
    train_corrects = 0
    valid_loss = 0.0
    valid_corrects = 0

    for phase in ["train", "valid"]:
        if phase == "train":
            model.train()
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in dataloaders[phase]:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == "train"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(data[phase])
        epoch_acc = running_corrects.double() / len(data[phase])

        if phase == "train":
            train_loss = epoch_loss
            train_corrects = epoch_acc
        else:
            valid_loss = epoch_loss
            valid_corrects = epoch_acc

        print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    print(f"Training Loss: {train_loss:.4f} | Training Accuracy: {train_corrects:.4f}")
    print(f"Validation Loss: {valid_loss:.4f} | Validation Accuracy: {valid_corrects:.4f}")
    print()

print("Training complete")

# Save the model
torch.save(model.state_dict(), "cnn axix ident 100 epoch 16 batch .pth")
print("Model saved as whatever.pth")
