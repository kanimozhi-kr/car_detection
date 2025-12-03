import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import requests

# Use the same architecture as the saved model
model = models.resnet50()
model.fc = torch.nn.Linear(2048, 2)  # Adjust output classes if needed

model_path = "./fraud_detector.pth"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

st.title("Car Insurance Fraud Detection")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

def is_ai_generated(image):
    # Example: Use a public API to check for AI-generated images (replace with your own logic or API)
    api_url = "https://api.deepware.ai/detect"
    try:
        response = requests.post(api_url, files={"image": image})
        result = response.json()
        return result.get("ai_generated", False)
    except Exception:
        return False

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Check if image is AI-generated
    ai_generated = is_ai_generated(uploaded_file)
    if ai_generated:
        st.warning("This image appears to be AI-generated or AI-edited. Marking as Fraudulent.")
        st.success("Prediction: Fraudulent")
    else:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Add normalization
        ])
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)
            predicted = torch.argmax(probs, dim=1)  # Use probabilities for prediction
            # If image is AI-generated, always mark as Fraudulent regardless of model prediction
            if ai_generated:
                label = "Fraudulent"
            else:
                label = "Fraudulent" if predicted.item() == 1 else "Genuine"
            st.write(f"Confidence (Fraudulent): {probs[0,1].item():.2f}")
            st.write(f"Confidence (Genuine): {probs[0,0].item():.2f}")

        st.success(f"Prediction: {label}")