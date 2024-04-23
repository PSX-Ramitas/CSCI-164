import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Define constants
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 14  # Assuming you have 15 different cat breeds
NUM_EPOCHS = 20

# Define the relative path to the dataset directory from the location of your Python script
data_dir = 'CatBreeds/Gano-Cat-Breeds-V1_1'

# Check if the directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The directory '{data_dir}' does not exist.")

# Define the path to the training directory (assuming it's directly under the dataset directory)
train_data_dir = data_dir

# Check if the directory exists
if not os.path.exists(train_data_dir):
    raise FileNotFoundError(f"The directory '{train_data_dir}' does not exist.")

# Define custom dataset
class CatDataset(Dataset):
    def __init__(self, filenames, labels, transform=None):
        self.filenames = filenames
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image = Image.open(self.filenames[idx]).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Data preprocessing and augmentation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming RGB images
])

# Initialize lists to store file paths and labels
train_filenames = []
train_labels = []

# Loop through each subdirectory (each breed)
for label in os.listdir(train_data_dir):
    label_dir = os.path.join(train_data_dir, label)
    # Loop through each image file in the subdirectory
    for filename in os.listdir(label_dir):
        train_filenames.append(os.path.join(label_dir, filename))
        train_labels.append(label)  # Append the label name (categorical) without conversion

# Split data into train and validation sets
train_filenames, val_filenames, train_labels, val_labels = train_test_split(
    train_filenames, train_labels, test_size=0.2, random_state=42)

# Initialize a dictionary to map labels to indices
label_to_index = {label: idx for idx, label in enumerate(set(train_labels)) if label != 'train'}
print("Label to index mapping:", label_to_index)


# Exclude 'train' label from the training labels
train_labels_filtered = [label for label in train_labels if label != 'train']

# Convert filtered training labels to numerical indices using the dictionary
train_labels_numeric = [label_to_index[label] for label in train_labels_filtered]

# Exclude 'train' label from the validation labels
val_labels_filtered = [label for label in val_labels if label != 'train']

# Convert filtered validation labels to numerical indices using the dictionary
val_labels_numeric = [label_to_index[label] for label in val_labels_filtered]

# Define datasets and data loaders
train_dataset = CatDataset(train_filenames, train_labels_numeric, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CatDataset(val_filenames, val_labels_numeric, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train the model
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)  # Assuming labels are already converted to tensors
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % 100 == 99:
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 100))
            running_loss = 0.0

print('Finished Training')

# Evaluate the model
correct = 0
total = 0
with torch.no_grad():
    for data in val_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the validation images: %d %%' % (
    100 * correct / total))
