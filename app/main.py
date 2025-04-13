from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import torch
from torchvision import transforms
from transformers import SwinForImageClassification, SwinConfig, AutoImageProcessor
import os
import io
import requests

app = FastAPI()

# === CONFIG ===
MODEL_PATH = "app/best_swin_model.pt"
MODEL_DRIVE_ID = "1oW-WkbZ6RuO-na8EgdxF1KoAYQ9j3PqI"  # Replace with actual ID
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASS_NAMES = ["healthy", "patient"]
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"

# === Image Processor ===
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
normalize = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# === Download model from Google Drive ===
def download_model_from_drive(file_id: str, destination: str):
    if os.path.exists(destination):
        print(f"Model already exists at {destination}")
        return
    print("Downloading model from Google Drive...")
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Failed to download model: {response.status_code}")
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    with open(destination, "wb") as f:
        f.write(response.content)
    print("Model download complete.")

# === Load model ===
download_model_from_drive(MODEL_DRIVE_ID, MODEL_PATH)

config = SwinConfig.from_pretrained(MODEL_NAME)
config.num_labels = NUM_CLASSES
model = SwinForImageClassification.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === Predict Endpoint ===
@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(pixel_values=image_tensor)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()

        return JSONResponse(content={
            'class_name': CLASS_NAMES[pred_class],
            'confidence': round(probs[0][pred_class].item(), 4)
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
