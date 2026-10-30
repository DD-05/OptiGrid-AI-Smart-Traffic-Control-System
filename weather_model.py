import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class names (original training order)
CLASS_NAMES = ['clear', 'fog', 'heavy_rain']

# Load model
def load_model(model_path="weather_mobilenet.pth"):
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, len(CLASS_NAMES))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# Load model globally
model = load_model()

# Transform (same as training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 🔥 Updated Prediction Function
def predict_weather(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image)
        probs = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(probs, 1)

    predicted_class = CLASS_NAMES[predicted.item()]

    # ✅ Merge fog + heavy_rain into one
    if predicted_class in ["fog", "heavy_rain"]:
        return "bad_weather"
    else:
        return "clear"
