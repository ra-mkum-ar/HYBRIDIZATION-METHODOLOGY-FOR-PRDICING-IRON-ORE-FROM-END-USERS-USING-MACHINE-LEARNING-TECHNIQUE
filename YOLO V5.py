import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

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
X = df.drop(columns=['Ore_Type'])  # Feature columns (e.g., Fe content, density, SiO2, etc.)
y = df['Ore_Type']  # Target variable

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train a RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
print('Classification Report:\n', classification_report(y_test, y_pred))

# Confusion matrix visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Feature importance visualization
feature_importances = model.feature_importances_
feature_names = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=feature_importances, y=feature_names)
plt.title("Feature Importance")
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.show()

# Example: Predicting a new sample
sample_data = np.array([[65.4, 2.1, 4.5, 3.2, 1.1]])  # Example values for features
sample_data = scaler.transform(sample_data)
prediction = model.predict(sample_data)
predicted_ore_type = label_encoder.inverse_transform(prediction)
print(f'Predicted Ore Type: {predicted_ore_type[0]}')

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
