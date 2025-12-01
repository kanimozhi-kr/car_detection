# Databricks notebook source
# DBTITLE 1,Install Libraries in Serverless Mode
# MAGIC %pip install torch torchvision transformers opencv-python streamlit pillow scikit-learn

# COMMAND ----------

# DBTITLE 1,Restart the Python kernel
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Verify Installation
# MAGIC %pip show torch torchvision streamlit

# COMMAND ----------

# DBTITLE 1,Kaggle Dataset
# MAGIC
# MAGIC %pip install kaggle

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------



!mkdir -p /tmp/.kaggle
with open('/tmp/.kaggle/kaggle.json', 'w') as f:
    f.write('{"username":"kanimozhikr","key":"b55c134d870dc6150c8f21e49f30fb35"}')




# COMMAND ----------


!kaggle datasets download -d anujms/car-damage-detection -p /tmp/
!unzip /tmp/car-damage-detection.zip -d /tmp/car_damage/


# COMMAND ----------

!ls /tmp/car_damage

# COMMAND ----------

# DBTITLE 1,data preprocessing

from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Path to dataset
data_dir = "/tmp/car_damage"

# Define transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load dataset
dataset = datasets.ImageFolder(root=data_dir, transform=transform)

# Split dataset
train_size = int(0.7 * len(dataset))
val_size = int(0.2 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}, Test samples: {len(test_dataset)}")


# COMMAND ----------

# DBTITLE 1,Training

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load pretrained ResNet50 with updated syntax
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze all layers except the last fully connected layer
for param in model.parameters():
    param.requires_grad = False

# Replace the final layer for binary classification (real vs fake)
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

# Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {running_loss/len(train_loader):.4f}")

# set
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# ✅ Save the trained model
torch.save(model.state_dict(), "/tmp/fraud_detector.pth")
print("Model saved at /tmp/fraud_detector.pth")


# COMMAND ----------

# MAGIC
# MAGIC %sh
# MAGIC cp /tmp/fraud_detector.pth ./fraud_detector.pth
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %pip install streamlit pyngrok torch torchvision pillow

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC
# MAGIC %%writefile app.py
# MAGIC import streamlit as st
# MAGIC import torch
# MAGIC import torch.nn as nn
# MAGIC import torchvision.transforms as transforms
# MAGIC from PIL import Image
# MAGIC
# MAGIC # your model architecture
# MAGIC class FraudDetector(nn.Module):
# MAGIC     def __init__(self):
# MAGIC         super(FraudDetector, self).__init__()
# MAGIC         self.fc = nn.Linear(512, 2)  # Adjust based on your model
# MAGIC
# MAGIC     def forward(self, x):
# MAGIC         return self.fc(x)
# MAGIC
# MAGIC # Load model from current directory
# MAGIC model_path = "./fraud_detector.pth"
# MAGIC model = FraudDetector()
# MAGIC model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
# MAGIC model.eval()
# MAGIC
# MAGIC # Streamlit UI
# MAGIC st.title("Car Insurance Fraud Detection")
# MAGIC uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
# MAGIC
# MAGIC if uploaded_file:
# MAGIC     image = Image.open(uploaded_file).convert("RGB")
# MAGIC     st.image(image, caption="Uploaded Image", use_column_width=True)
# MAGIC
# MAGIC     transform = transforms.Compose([
# MAGIC         transforms.Resize((224, 224)),
# MAGIC         transforms.ToTensor()
# MAGIC     ])
# MAGIC     img_tensor = transform(image).unsqueeze(0)
# MAGIC
# MAGIC     with torch.no_grad():
# MAGIC         output = model(img_tensor)
# MAGIC         probs = torch.softmax(output, dim=1)
# MAGIC         _, predicted = torch.max(output, 1)
# MAGIC         label = "Fraudulent" if predicted.item() == 1 else "Genuine"
# MAGIC
# MAGIC     st.success(f"Prediction: {label}")
# MAGIC