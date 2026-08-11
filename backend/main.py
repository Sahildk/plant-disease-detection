import io
import json
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
import torchvision.models as models

# Initialize FastAPI App
app = FastAPI(
    title="Plant Disease Detection API",
    description="A simple API that predicts plant disease from leaf images using a fine-tuned MobileNetV3 model.",
    version="1.0.0"
)

# Enable CORS for frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load class names
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(BASE_DIR, "model", "labels.json")
model_path = os.path.join(BASE_DIR, "model", "plant_disease_model.pth")

try:
    with open(labels_path, "r") as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} classes.")
except FileNotFoundError:
    raise RuntimeError(f"Class labels not found at {labels_path}. Please run training first.")

# Define and load PyTorch model
try:
    # Instantiate the base model architecture
    model = models.mobilenet_v3_large()
    
    # Replace the classification head to match training structure
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, len(class_names))
    )
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
    print("Model loaded successfully.")
except Exception as e:
    raise RuntimeError(f"Failed to load model weights: {e}")

# Preprocessing transforms (matches training validation setup)
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": True,
        "classes_count": len(class_names)
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
    
    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply preprocessing transforms
        input_tensor = preprocess(image).unsqueeze(0)
        
        # Run inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            confidence, class_idx = torch.max(probabilities, dim=0)
            
        class_name = class_names[class_idx.item()]
        
        # Format predicted class name for user display (e.g. Tomato___Early_blight -> Tomato - Early blight)
        display_name = class_name.replace("___", " - ").replace("_", " ")
        
        # Determine health status
        status = "Healthy" if "healthy" in class_name.lower() else "Disease Detected"
        
        return {
            "disease": display_name,
            "raw_class": class_name,
            "confidence": float(confidence.item()),
            "status": status
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
