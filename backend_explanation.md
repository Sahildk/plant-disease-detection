# 📘 Backend Code Explanation — `backend/main.py`

This document provides a simple, section-by-section breakdown of how the **FastAPI** backend server in [`backend/main.py`](file:///g:/Github/plant-disease-detection/backend/main.py) processes requests and runs deep learning inference.

---

## 1. Imports and Dependencies (Lines 1–10)

```python
import io
import json
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms
import torchvision.models as models
```

*   **`io`**: Handles in-memory byte streams so we can open uploaded images directly without saving temporary files to disk.
*   **`json`**: Loads the 38 class label names stored in `labels.json`.
*   **`torch` & `torch.nn`**: Core PyTorch library used to load neural network weights and run matrix calculations.
*   **`PIL.Image`**: Python Imaging Library used to decode uploaded image formats (JPEG, PNG, WEBP) into RGB format.
*   **`FastAPI` & Middleware**: Web framework used to build our high-performance HTTP REST API.
*   **`torchvision`**: PyTorch library providing pre-trained computer vision architectures (`mobilenet_v3_large`) and image transformation functions.

---

## 2. Initializing FastAPI & Configuring CORS (Lines 11–25)

```python
app = FastAPI(
    title="Plant Disease Detection API",
    description="A simple API that predicts plant disease from leaf images...",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

*   **`FastAPI(...)`**: Instantiates the web application and generates automatic interactive documentation at `http://127.0.0.1:8000/docs`.
*   **`CORSMiddleware`**: Cross-Origin Resource Sharing middleware. It allows browser frontends (like React running on `http://localhost:5173`) to send HTTP requests to this backend without being blocked by browser security restrictions.

---

## 3. Dynamic File Paths & Class Label Loading (Lines 27–38)

```python
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
labels_path = os.path.join(BASE_DIR, "model", "labels.json")
model_path = os.path.join(BASE_DIR, "model", "plant_disease_model.pth")

try:
    with open(labels_path, "r") as f:
        class_names = json.load(f)
    print(f"Loaded {len(class_names)} classes.")
except FileNotFoundError:
    raise RuntimeError(...)
```

*   **`BASE_DIR`**: Calculates the absolute directory path of `main.py`. This ensures the model weights and labels are found regardless of which folder you run the command from.
*   **`json.load(f)`**: Loads the list of 38 plant disease class names (`["Apple___Apple_scab", "Tomato___Early_blight", ...]`).

---

## 4. Model Architecture Setup & Weight Loading (Lines 40–59)

```python
try:
    model = models.mobilenet_v3_large()
    
    in_features = model.classifier[0].in_features
    model.classifier = nn.Sequential(
        nn.Linear(in_features, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.3),
        nn.Linear(1024, len(class_names))
    )
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()
except Exception as e:
    raise RuntimeError(...)
```

1.  **`models.mobilenet_v3_large()`**: Creates the MobileNetV3 Large neural network architecture.
2.  **`model.classifier = ...`**: Replaces the default ImageNet output layer (1000 classes) with our custom 2-layer classifier head (matching our 38 plant disease classes).
3.  **`torch.load(...)`**: Loads the trained weights from `plant_disease_model.pth` into memory.
4.  **`model.eval()`**: Puts the PyTorch model in **evaluation mode**. This disables training features like `Dropout` to ensure consistent predictions.

---

## 5. Image Preprocessing Pipeline (Lines 61–66)

```python
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

Before an uploaded image is fed into the deep learning model, it must match the training format:
*   **`Resize((224, 224))`**: Scales any input image dimensions to exactly 224x224 pixels.
*   **`ToTensor()`**: Converts raw pixel values (0–255) into floating point numbers scaled between `0.0` and `1.0`.
*   **`Normalize(...)`**: Standardizes color channels using standard ImageNet mean and standard deviation parameters.

---

## 6. Health Check Endpoint (Lines 68–74)

```python
@app.get("/")
def read_root():
    return {
        "status": "online",
        "model_loaded": True,
        "classes_count": len(class_names)
    }
```

A quick `GET` endpoint to verify that the server is online and the model is successfully loaded.

---

## 7. Prediction Endpoint `POST /predict` (Lines 76–111)

This is the primary endpoint called by the React frontend when a user uploads a leaf image.

### A. Input Validation & Image Parsing
```python
if not file.content_type.startswith("image/"):
    raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

image_bytes = await file.read()
image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
```
*   Ensures the uploaded file is an image file type.
*   Reads the byte stream and converts the image to standard 3-channel RGB format.

### B. Tensor Batching & Model Inference
```python
input_tensor = preprocess(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
    confidence, class_idx = torch.max(probabilities, dim=0)
```
*   **`unsqueeze(0)`**: Adds a batch dimension to transform the tensor shape from `[3, 224, 224]` to `[1, 3, 224, 224]`.
*   **`with torch.no_grad()`**: Disables gradient computation during inference to speed up processing and save RAM.
*   **`softmax(...)`**: Converts raw neural network output scores into percentage probabilities between 0 and 1.
*   **`torch.max(...)`**: Finds the highest probability score and its corresponding class index.

### C. Formatting the JSON Response
```python
class_name = class_names[class_idx.item()]
display_name = class_name.replace("___", " - ").replace("_", " ")
status = "Healthy" if "healthy" in class_name.lower() else "Disease Detected"

return {
    "disease": display_name,
    "raw_class": class_name,
    "confidence": float(confidence.item()),
    "status": status
}
```
*   Translates the numeric index back to the readable class name (e.g. `Tomato___Early_blight` $\rightarrow$ `Tomato - Early blight`).
*   Checks if the predicted label contains the keyword `"healthy"` to assign the health status.
*   Returns the final JSON payload back to the React UI.

---

## 8. Server Execution Entry Point (Lines 113–116)

```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
```

When you execute `python backend/main.py` directly, this block starts the **Uvicorn** server listening on port `8000` with auto-reload enabled.
