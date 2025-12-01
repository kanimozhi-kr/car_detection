
import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Define your model architecture (adjust as per your training)
class FraudDetector(nn.Module):
    def __init__(self):
        super(FraudDetector, self).__init__()
        self.fc = nn.Linear(512, 2)  # Example architecture

    def forward(self, x):
        return self.fc(x)

# Load model
model_path = "fraud_detector.pth"  # Ensure this file is in the repo
model = FraudDetector()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Streamlit UI
st.title("Car Insurance Fraud Detection")
st.write("Upload an image to check if it's fraudulent or genuine.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image (adjust based on your training pipeline)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        _, predicted = torch.max(output, 1)
        label = "Fraudulent" if predicted.item() == 1 else "Genuine"

    st.success(f"Prediction: {label}")
    st.write(f"Confidence Scores: {probs.squeeze().tolist()}")
