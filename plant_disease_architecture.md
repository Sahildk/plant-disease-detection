# Plant Disease Detection — System Architecture

## 1. Project Overview

The project is a web-based plant disease detection system. A user uploads an image of a plant leaf through the frontend, the backend sends the image to a trained deep learning model, and the system returns the predicted disease and confidence score.

The project is intentionally kept simple and focuses on image classification.

## 2. High-Level Architecture

```text
┌──────────────────────────┐
│        User              │
│  Uploads Plant Image     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│       Frontend           │
│   React / Next.js        │
│                          │
│ - Image Upload           │
│ - Image Preview          │
│ - Prediction Result      │
│ - Confidence Score       │
└────────────┬─────────────┘
             │ HTTP Request
             ▼
┌──────────────────────────┐
│        Backend           │
│        FastAPI           │
│                          │
│ - Receive Image          │
│ - Preprocess Image       │
│ - Run Model Inference    │
│ - Return Prediction      │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│    Trained ML Model      │
│                          │
│ Deep Learning Image      │
│ Classification Model     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│ Prediction Response      │
│                          │
│ Disease + Confidence     │
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│        Frontend          │
│ Display Results          │
└──────────────────────────┘
```

## 3. Main Components

### 3.1 Dataset

The project will use the **PlantVillage dataset** containing images of healthy and diseased plant leaves.

The dataset will be used to:

- Train the image classification model.
- Validate the model during training.
- Test the final model.

The dataset images will be organized according to their disease/class labels.

Example:

```text
dataset/
├── train/
│   ├── Tomato___Early_blight/
│   ├── Tomato___Late_blight/
│   ├── Tomato___healthy/
│   └── ...
├── validation/
│   └── ...
└── test/
    └── ...
```

### 3.2 Image Preprocessing

Before an image is passed to the model, it will be preprocessed to match the format used during training.

Basic preprocessing may include:

- Resizing the image.
- Converting it to the required image format.
- Normalizing pixel values.
- Converting the image into a tensor.

Image augmentation can be applied during training to improve the model's ability to handle variations in images.

### 3.3 Machine Learning Model

A deep learning image classification model will be trained using the PlantVillage dataset.

A transfer-learning model such as **MobileNetV3, EfficientNet-B0, or ResNet** can be used.

The model receives a plant image and outputs probabilities for the available disease classes.

```text
Plant Image
     ↓
Image Preprocessing
     ↓
Deep Learning Model
     ↓
Class Probabilities
     ↓
Highest Probability Class
     ↓
Predicted Disease
```

The final model will be saved as a model file and loaded by the backend during inference.

### 3.4 Backend

The backend will be developed using **Python and FastAPI**.

Its main responsibilities are:

1. Receive an uploaded image from the frontend.
2. Validate the uploaded file.
3. Preprocess the image.
4. Load/run the trained model.
5. Generate the prediction.
6. Calculate the confidence score.
7. Return the result to the frontend.

Example API:

```text
POST /predict
```

Example response:

```json
{
  "disease": "Tomato Early Blight",
  "confidence": 0.947
}
```

### 3.5 Frontend

The frontend will be developed using **Next.js / React with TypeScript**.

The main interface will contain:

- Project title and description.
- Image upload area.
- Image preview.
- Analyze button.
- Loading state.
- Prediction result.
- Confidence score.

Basic user flow:

```text
Upload Image
     ↓
Preview Image
     ↓
Click "Analyze"
     ↓
Send Image to API
     ↓
Wait for Prediction
     ↓
Display Disease + Confidence
```

## 4. Technology Stack

| Component | Technology |
|---|---|
| Frontend | Next.js / React |
| Language | TypeScript |
| Styling | Tailwind CSS |
| Backend | Python + FastAPI |
| ML Framework | TensorFlow/Keras or PyTorch |
| Dataset | PlantVillage |
| Model | Transfer Learning CNN |
| API | REST API |

## 5. Project Directory Structure

A simple project structure can be:

```text
plant-disease-detection/
│
├── frontend/
│   ├── app/
│   ├── components/
│   ├── public/
│   └── package.json
│
├── backend/
│   ├── main.py
│   ├── model/
│   │   └── disease_model
│   ├── preprocessing/
│   └── requirements.txt
│
├── training/
│   ├── dataset/
│   ├── train.py
│   ├── evaluate.py
│   └── requirements.txt
│
├── README.md
└── architecture.md
```

## 6. End-to-End Data Flow

```text
Plant Leaf Image
       │
       ▼
Frontend Upload
       │
       ▼
FastAPI /predict
       │
       ▼
Image Preprocessing
       │
       ▼
Trained Classification Model
       │
       ▼
Disease Prediction
       │
       ▼
Confidence Score
       │
       ▼
JSON Response
       │
       ▼
Frontend Result Screen
```

## 7. Expected Output

For an uploaded plant image, the system should display:

```text
Prediction: Tomato Early Blight

Confidence: 94.7%

Status: Disease Detected
```

If the model predicts a healthy plant:

```text
Prediction: Tomato Healthy

Confidence: 98.2%

Status: Healthy
```

## 8. Future Improvements

The initial version will remain focused on basic image classification. Possible future improvements include:

- Adding more plant disease classes.
- Improving model accuracy.
- Adding camera-based image capture.
- Deploying the model and frontend online.
- Adding prediction history.

These features are optional and are not required for the initial project.
