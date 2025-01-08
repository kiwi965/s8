import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # For progress bar


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

num_classes = 5 

# Define data transformations with Grayscale conversion
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Convert RGB to grayscale
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load datasets
train_data = datasets.ImageFolder("trainval(axial)/train", transform=transform)
valid_data = datasets.ImageFolder("trainval(axial)/valid", transform=transform)

# DataLoaders
train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
valid_loader = DataLoader(valid_data, batch_size=8, shuffle=False)

# Define the CNN model
class RibFractureCNN(nn.Module):
    def __init__(self, num_classes):
        super(RibFractureCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Input: 1 channel, Output: 32 filters
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv8 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # Pooling layer
        self.fc1 = nn.Linear(32 * 8 * 8, 128)  # Fully connected layer
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(torch.relu(self.conv2(x)))
        x = torch.relu(self.conv3(x))
        x = self.pool(torch.relu(self.conv4(x)))
        x = torch.relu(self.conv5(x))
        x = self.pool(torch.relu(self.conv6(x)))
        x = torch.relu(self.conv7(x))
        x = self.pool(torch.relu(self.conv8(x)))
        x = x.view(-1, 32 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model, define loss and optimizer
#model = RibFractureCNN()
model = RibFractureCNN(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training function with progress bar'''

def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_true, train_pred = [], []
        
        # Training loop with progress bar
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{epochs} (Train)", unit="batch") as pbar:
            for inputs, labels in train_loader:
                # Move inputs and labels to the same device as the model
                inputs, labels = inputs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_true.extend(labels.cpu().numpy())  # Move labels to CPU before converting to numpy
                train_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())  # Move predictions to CPU
                
                # Update progress bar
                pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy_score(train_true, train_pred)})
                pbar.update(1)

        train_acc = accuracy_score(train_true, train_pred)

        # Validation loop
        model.eval()
        valid_loss = 0.0
        valid_true, valid_pred = [], []
        with torch.no_grad():
            with tqdm(total=len(valid_loader), desc=f"Epoch {epoch + 1}/{epochs} (Valid)", unit="batch") as pbar:
                for inputs, labels in valid_loader:
                    # Move inputs and labels to the same device as the model
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    valid_loss += loss.item()
                    valid_true.extend(labels.cpu().numpy())  # Move labels to CPU before converting to numpy
                    valid_pred.extend(torch.argmax(outputs, dim=1).cpu().numpy())  # Move predictions to CPU
                    
                    # Update progress bar
                    pbar.set_postfix({"Loss": loss.item(), "Accuracy": accuracy_score(valid_true, valid_pred)})
                    pbar.update(1)

        valid_acc = accuracy_score(valid_true, valid_pred)

        # Print metrics summary
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Train Loss: {train_loss / len(train_loader):.4f} | Train Accuracy: {train_acc:.4f}")
        print(f"Valid Loss: {valid_loss / len(valid_loader):.4f} | Valid Accuracy: {valid_acc:.4f}")
        print("-" * 50)

# Train the model
train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10)
# Save the model
torch.save(model.state_dict(), "model_after_minusone_axial.pth")
print("Model saved as rib_fracture_cnn.pth")