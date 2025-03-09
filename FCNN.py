import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader, TensorDataset

# Load dataset (replace 'iron_ore_data.csv' with actual dataset file)
df = pd.read_csv('iron_ore_data.csv')

# Display basic info
print("Dataset Overview:")
print(df.info())
print(df.describe())

# Visualizing class distribution
sns.countplot(x='Ore_Type', data=df)
plt.title("Ore Type Distribution")
plt.show()

# Encode categorical target variable if necessary
if df['Ore_Type'].dtype == 'object':
    label_encoder = LabelEncoder()
    df['Ore_Type'] = label_encoder.fit_transform(df['Ore_Type'])

# Define features and target
X = df.drop(columns=['Ore_Type']).values  # Feature columns (e.g., Fe content, density, SiO2, etc.)
y = df['Ore_Type'].values  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define FCNN model
class FCNN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(FCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, num_classes)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Initialize model
input_size = X_train.shape[1]
num_classes = len(np.unique(y))
model_fcnn = FCNN(input_size, num_classes)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_fcnn.parameters(), lr=0.001)

# Train the model
epochs = 20
for epoch in range(epochs):
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model_fcnn(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}')

# Evaluate model
model_fcnn.eval()
y_pred_list = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        y_pred = model_fcnn(X_batch)
        y_pred_list.extend(torch.argmax(y_pred, axis=1).tolist())

accuracy = accuracy_score(y_test, y_pred_list)
print(f'FCNN Accuracy: {accuracy:.2f}')
print('FCNN Classification Report:\n', classification_report(y_test, y_pred_list))

# Load YOLOv5 model for image-based classification
model_yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

def detect_ore(image_path):
    image = cv2.imread(image_path)
    results = model_yolo(image)
    results.show()  # Display results
    return results.pandas().xyxy[0]  # Return detected objects

# Example usage
image_path = 'iron_ore_sample.jpg'  # Replace with actual image
ore_detections = detect_ore(image_path)
print("YOLO Detections:")
print(ore_detections)
