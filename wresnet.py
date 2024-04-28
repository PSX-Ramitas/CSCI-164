import torch
import torchvision.models as models
import copy
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# Load pre-trained resnet101 model and modify it for 15 cat breeds
resnetModel = models.resnet101(pretrained=True)
num_features = resnetModel.fc.in_features
resnetModel.fc = nn.Linear(num_features, 15)
# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnetModel = resnetModel.to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnetModel.parameters(), lr=0.001, momentum=0.9)
# Data transformations and loaders
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
train_data = datasets.ImageFolder(root='./cats/Training', transform=transform)
val_data = datasets.ImageFolder(root='./cats/Validation', transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

def train(model, criterion, optimizer, train_loader, device):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        
        # Forward
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    # Calculate average loss and accuracy
    avg_loss = running_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

# Validation loop
def validate(model, criterion, val_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return running_loss / len(val_loader), accuracy

# Assume all imports and model initialization are already done
num_epochs = 10
best_model_wts = copy.deepcopy(resnetModel.state_dict())
best_acc = 0.0

for epoch in range(num_epochs):
    print(f'Epoch {epoch+1}/{num_epochs}')
    print('-' * 10)

    # Train and validate
    train_loss, train_accuracy = train(resnetModel, criterion, optimizer, train_loader, device)
    val_loss, val_accuracy = validate(resnetModel, criterion, val_loader, device)

    print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
    print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

    # Check if this is the best model based on validation accuracy
    if val_accuracy > best_acc:
        best_acc = val_accuracy
        best_model_wts = copy.deepcopy(resnetModel.state_dict())

    print()

# Load best model weights
resnetModel.load_state_dict(best_model_wts)

# Save the best model
torch.save(resnetModel.state_dict(), 'best_resnet101_cat_breeds.pth')
print('Best model saved with accuracy:', best_acc)
