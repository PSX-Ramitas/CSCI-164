import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchsummary import summary
from sklearn.model_selection import train_test_split
from PIL import Image
import cv2  # OpenCV
import matplotlib.pyplot as plt
import time

if torch.cuda.is_available():
    print("CUDA (GPU) is available.")
else:
    print("CUDA (GPU) is not available. Using CPU instead.")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Define constants
IMAGE_SIZE = 64
BATCH_SIZE = 32
NUM_CLASSES = 16  # Assuming you have 15 different cat breeds
NUM_EPOCHS = 1

# Define the relative path to the dataset directory from the location of your Python script
#data_dir = 'CatBreeds\\Gano-Cat-Breeds-V1_1'
data_dir = 'C:\\Users\\Tomas\\Documents\\CatBreeds\\Gano-Cat-Breeds-V1_1'
# Check if the directory exists
if not os.path.exists(data_dir):
    raise FileNotFoundError(f"The directory '{data_dir}' does not exist.")

# Load pre-trained Haar cascade classifier for cat face detection
cat_face_cascade = cv2.CascadeClassifier('haarcascade_frontalcatface_extended.xml')

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

        # Detect cat faces and extract the first detected face
       # gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
       # gray_tensor = torch.from_numpy(gray).to(device)

        cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(cat_faces) > 0:
            x, y, w, h = cat_faces[0]
            face_image = image.crop((x, y, x+w, y+h))  # Crop the detected face region
            if self.transform:
                face_image = self.transform(face_image)
            return face_image, label
        else:
            # If no cat face detected, return original image with label
            if self.transform:
                image = self.transform(image)
            return image, label

# Define data augmentation transformations for training
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(20),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming RGB images
])

# Data preprocessing transformation for validation
preprocess_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Assuming RGB images
])

# Initialize lists to store file paths and labels
all_filenames = []
all_labels = []

# Loop through each subdirectory (each breed)
for label in os.listdir(data_dir):
    label_dir = os.path.join(data_dir, label)
    # Loop through each image file in the subdirectory
    for filename in os.listdir(label_dir):
        all_filenames.append(os.path.join(label_dir, filename))
        all_labels.append(label)  # Append the label name (categorical) without conversion

# Convert labels to numerical indices
label_to_index = {label: idx for idx, label in enumerate(set(all_labels))}

# Convert all labels to numerical indices using the dictionary
all_labels_numeric = [label_to_index[label] for label in all_labels]

# Split data into train and validation sets
train_filenames, val_filenames, train_labels, val_labels = train_test_split(
    all_filenames, all_labels_numeric, test_size=0.2, random_state=42)

# Define datasets and data loaders
train_dataset = CatDataset(train_filenames, train_labels, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

val_dataset = CatDataset(val_filenames, val_labels, transform=preprocess_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
print("Train Dataset size:", len(train_dataset))
print("Validation Dataset size:", len(val_dataset), '\n')

# Define CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(128 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, NUM_CLASSES)
    def forward(self, x):
        x= x.to(device)
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, 128 * 8 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        return x

#model = CNN()   --no gpu
model = CNN().to(device)
summary(model, (3, 64, 64))

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define function to calculate accuracy
def calculate_accuracy(loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)  # Move images to device
            labels = labels.to(device)  # Move labels to device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct / total

# Define function to calculate false positives and false negatives
def calculate_false_predictions(loader):
    model.eval()
    false_positives = 0
    false_negatives = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)  # Move images to device
            labels = labels.to(device)  # Move labels to device
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            false_positives += ((predicted == 1) & (labels == 0)).sum().item()  # Count false positives
            false_negatives += ((predicted == 0) & (labels == 1)).sum().item()  # Count false negatives
    return false_positives, false_negatives

print('\n', "Preparing to Train Model...", '\n')

# Record start time
start_time = time.time()

# Train the model
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_false_positives = []
train_false_negatives = []
val_false_positives = []
val_false_negatives = []

for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
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
    
    # Calculate accuracy and false predictions on training and validation sets
    train_accuracy = calculate_accuracy(train_loader)
    val_accuracy = calculate_accuracy(val_loader)
    train_losses.append(running_loss / len(train_loader))
    val_losses.append(loss.item())
    train_accuracies.append(train_accuracy)
    val_accuracies.append(val_accuracy)
    
    train_false_positives_epoch, train_false_negatives_epoch = calculate_false_predictions(train_loader)
    val_false_positives_epoch, val_false_negatives_epoch = calculate_false_predictions(val_loader)
    train_false_positives.append(train_false_positives_epoch)
    train_false_negatives.append(train_false_negatives_epoch)
    val_false_positives.append(val_false_positives_epoch)
    val_false_negatives.append(val_false_negatives_epoch)

    # Print training loss and validation accuracy for the epoch
    print('Epoch %d: Training Loss: %.3f, Training Accuracy: %.2f%%, Validation Accuracy: %.2f%%' %
          (epoch + 1, train_losses[-1], train_accuracies[-1] * 100, val_accuracies[-1] * 100))
    print('False Positives (Train): %d, False Negatives (Train): %d' % (train_false_positives_epoch, train_false_negatives_epoch))
    print('False Positives (Validation): %d, False Negatives (Validation): %d' % (val_false_positives_epoch, val_false_negatives_epoch))

# Save the trained model
torch.save(model.state_dict(), 'catModelsVJ.pth')

print("Training complete, beginning prediction...", '\n')

# Load the trained model
model = CNN().to(device) 
model.load_state_dict(torch.load('catModelsVJ.pth'))
model.eval()

# Define function to predict breed of a test image and show the detected face
def predict_breed_with_face(image_path):
    # Load and preprocess the test image
    img = Image.open(image_path).convert('RGB')
    img_copy = img.copy()  # Create a copy of the original image
    
    # Convert image to grayscale for face detection
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)
    
    # Detect cat faces
    cat_faces = cat_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # If faces are detected, draw rectangles around them and display the images
    if len(cat_faces) > 0:
        for (x, y, w, h) in cat_faces:
            cv2.rectangle(np.array(img_copy), (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Save the test image with detected faces
        cv2.imwrite('output_image.jpg', img_copy)
        print("Saved cropped image as output_image.jpg")

        # Preprocess the image for the model
        img_tensor = preprocess_transform(img_copy).unsqueeze(0)  # Add batch dimension
        
        # Forward pass through the model
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
        # Map predicted index to breed label
        index_to_label = {v: k for k, v in label_to_index.items()}
        predicted_breed = index_to_label[predicted.item()]
    else:
        print("No cat faces detected.")
        # Preprocess the image for the model
        img_tensor = preprocess_transform(img).unsqueeze(0)  # Add batch dimension
        
        # Forward pass through the model
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
    
        # Map predicted index to breed label
        index_to_label = {v: k for k, v in label_to_index.items()}
        predicted_breed = index_to_label[predicted.item()]
    return predicted_breed

# Test image path
test_image_path = 'test.jpg'

# Predict breed of the test image and show the detected face
prediction = predict_breed_with_face(test_image_path)
print(f"Predicted breed: {prediction}")

# Record end time
end_time = time.time()

# Calculate runtime
runtime = end_time - start_time
# Calculate runtime in minutes and seconds
minutes = int(runtime // 60)
seconds = int(runtime % 60)

print(f"Total runtime: {minutes} minutes {seconds} seconds")

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, label='Training Loss')
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_accuracies, label='Training Accuracy')
plt.plot(range(1, NUM_EPOCHS + 1), val_accuracies, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot false positives and false negatives
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_false_positives, label='False Positives (Train)')
plt.plot(range(1, NUM_EPOCHS + 1), train_false_negatives, label='False Negatives (Train)')
plt.plot(range(1, NUM_EPOCHS + 1), val_false_positives, label='False Positives (Validation)')
plt.plot(range(1, NUM_EPOCHS + 1), val_false_negatives, label='False Negatives (Validation)')
plt.xlabel('Epoch')
plt.ylabel('Count')
plt.title('False Positives and False Negatives')
plt.legend()
plt.show()