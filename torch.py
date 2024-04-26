import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torch.optim import Adam

model = models.vgg16(pretrained=True)
for param in model.parameters():
    param.requires_grad = False  # Freeze all parameters


num_features = model.classifier[0].in_features
model.classifier = nn.Sequential(
    nn.Linear(num_features, 512),  
    nn.BatchNorm1d(512),           
    nn.LeakyReLU(negative_slope=0.01),  
    nn.Dropout(0.2),               

    nn.Linear(512, 256),
    nn.BatchNorm1d(256),
    nn.LeakyReLU(negative_slope=0.01),
    nn.Dropout(0.1),               

    nn.Linear(256, 15),
    nn.Softmax(dim=1)              
)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=10, translate=(0.2, 0.2), scale=(0.9, 1.1), shear=10),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = ImageFolder('./cats/Training', transform=transform)
valid_dataset = ImageFolder('./cats/Validation', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False, num_workers=4)

  # Unfreeze the last convolutional block
for param in model.features[-6:].parameters():
    param.requires_grad = True

# Setup the optimizer with differential learning rates
optimizer = Adam([
    {'params': model.features.parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
])

# Define the loss function
criterion = nn.CrossEntropyLoss()


def train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
        
        model.eval()
        valid_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_val_loss = valid_loss / len(valid_loader.dataset)
        epoch_val_accuracy = correct / total
        
        print(f"Epoch {epoch+1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {epoch_val_loss:.4f}, Validation Accuracy: {epoch_val_accuracy:.4f}")
        
train_model(model, train_loader, valid_loader, criterion, optimizer, epochs=20)
