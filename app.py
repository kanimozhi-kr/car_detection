import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

from transformers import CLIPProcessor, CLIPModel

# Class names
class_names = ["real_car","not_car", "ai_generated_car", "ai_edited_car"]
real_car_vs_not_car_labels = ["real_car", "not_real_car"]

# Model definition: ResNet50
class CarTypeDetector(nn.Module):
    def __init__(self, num_classes=4):
        super(CarTypeDetector, self).__init__()
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
    def forward(self, x):
        return self.model(x)

# Load ResNet50 model with strict=False to ignore key mismatches
resnet_model_path = "./car_type_detector_resnet50.pth"
resnet_model = CarTypeDetector(num_classes=len(class_names))
state_dict = torch.load(resnet_model_path, map_location=torch.device('cpu'))
resnet_model.load_state_dict(state_dict, strict=False)
resnet_model.eval()

# Load OpenAI CLIP model
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

# Image transform for ResNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# OpenAI CLIP hierarchical car detection
def openai_car_hierarchical_predict(image):
    # First, check car vs not_car
    car_vs_not_car_labels = ["car", "not_car"]
    inputs = clip_processor(
        text=car_vs_not_car_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1).squeeze().tolist()
        pred_idx = logits.argmax(dim=1).item()
        pred_label = car_vs_not_car_labels[pred_idx]
    if pred_label == "not_car":
        return "not_car", dict(zip(car_vs_not_car_labels, probs))
    # If car, check ai_generated vs ai_edited vs real
    car_type_labels = ["ai_generated_car", "ai_edited_car", "real_car"]
    inputs2 = clip_processor(
        text=car_type_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs2 = clip_model(**inputs2)
        logits2 = outputs2.logits_per_image
        probs2 = logits2.softmax(dim=1).squeeze().tolist()
        pred_idx2 = logits2.argmax(dim=1).item()
        pred_label2 = car_type_labels[pred_idx2]
    return pred_label2, dict(zip(car_type_labels, probs2))

def clip_real_car_predict(image):
    inputs = clip_processor(
        text=real_car_vs_not_car_labels,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        outputs = clip_model(**inputs)
        logits = outputs.logits_per_image
        probs = logits.softmax(dim=1).squeeze().tolist()
        pred_idx = logits.argmax(dim=1).item()
        pred_label = real_car_vs_not_car_labels[pred_idx]
    return pred_label, dict(zip(real_car_vs_not_car_labels, probs))

# Streamlit UI
st.title("Car Type Detection: Not Car / AI Generated / AI Edited / Real Car")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ResNet50 prediction
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = resnet_model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probs, dim=1).item()
        label = class_names[pred_idx]
        st.success(f"ResNet50 Prediction: {label}")
        st.write(f"ResNet50 Probabilities: {dict(zip(class_names, probs.squeeze().tolist()))}")

    # CLIP prediction
    clip_inputs = clip_processor(
        text=class_names,
        images=image,
        return_tensors="pt",
        padding=True
    )
    with torch.no_grad():
        clip_outputs = clip_model(**clip_inputs)
        logits_per_image = clip_outputs.logits_per_image
        clip_probs = logits_per_image.softmax(dim=1).squeeze().tolist()
        clip_pred_idx = logits_per_image.argmax(dim=1).item()
        clip_label = class_names[clip_pred_idx]
        st.success(f"CLIP Prediction: {clip_label}")
        st.write(f"CLIP Probabilities: {dict(zip(class_names, clip_probs))}")

    # CLIP real car vs not real car prediction
    real_car_label, real_car_probs = clip_real_car_predict(image)
    st.success(f"CLIP Real Car Prediction: {real_car_label}")
    st.write(f"CLIP Real Car Probabilities: {real_car_probs}")

    # OpenAI hierarchical car detection
    hier_label, hier_probs = openai_car_hierarchical_predict(image)
    st.success(f"OpenAI Hierarchical Prediction: {hier_label}")
    st.write(f"OpenAI Hierarchical Probabilities: {hier_probs}")