from flask import Flask, request, jsonify
import torch
from PIL import Image
from torchvision import transforms
from transformers import SwinForImageClassification, SwinConfig, AutoImageProcessor
import io
import os
import gdown

app = Flask(__name__)

# === CONFIG ===
MODEL_NAME = "microsoft/swin-tiny-patch4-window7-224"
MODEL_PATH = "best_swin_model.pt"
MODEL_FILE_ID = "1oW-WkbZ6RuO-na8EgdxF1KoAYQ9j3PqI"
NUM_CLASSES = 2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
CLASS_NAMES = ["healthy", "patient"]

# === Download model if not present ===
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={MODEL_FILE_ID}", MODEL_PATH, quiet=False)

# === Image processor ===
image_processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
normalize = transforms.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    normalize,
])

# === Load model ===
download_model()
config = SwinConfig.from_pretrained(MODEL_NAME)
config.num_labels = NUM_CLASSES
model = SwinForImageClassification.from_pretrained(MODEL_NAME, config=config, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# === API endpoint ===
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    image_file = request.files['image']
    image = Image.open(image_file.stream).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(pixel_values=image_tensor)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    return jsonify({
        'class_name': CLASS_NAMES[pred_class],
        'confidence': round(probs[0][pred_class].item(), 4)
    })

# === Run Flask server ===
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
