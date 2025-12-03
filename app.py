%pip install streamlit pyngrok torch torchvision pillow
dbutils.library.restartPython()

import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# Use the same architecture as the saved model
model = models.resnet50()
model.fc = torch.nn.Linear(2048, 2)  # Adjust output classes if needed

model_path = "./fraud_detector.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

st.title("Car Insurance Fraud Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

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
