import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.models as models
from datasets import load_dataset
from PIL import Image

def main():
    print("Loading Hugging Face dataset: geraldmc/plantvillage-full...", flush=True)
    # Load dataset
    hf_dataset = load_dataset("geraldmc/plantvillage-full")
    train_split = hf_dataset["train"]
    
    # Extract class names
    print("Extracting class names...", flush=True)
    class_mapping = sorted(list(set(zip(train_split["class_idx"], train_split["class_label"]))), key=lambda x: x[0])
    class_names = [label for idx, label in class_mapping]
        
    print(f"Loaded dataset with {len(class_names)} classes.", flush=True)
    
    # Save labels.json early
    os.makedirs("backend/model", exist_ok=True)
    with open("backend/model/labels.json", "w") as f:
        json.dump(class_names, f, indent=2)
    print("Saved labels.json to backend/model/", flush=True)
    
    # Create a balanced subset (up to 200 images per class for robust training)
    print("Selecting subset of indices (up to 200 per class)...", flush=True)
    class_indices = {i: [] for i in range(len(class_names))}
    
    labels_column = train_split["class_idx"]
    for idx, label in enumerate(labels_column):
        if len(class_indices[label]) < 200:
            class_indices[label].append(idx)
            
    # Combine indices
    subset_indices = []
    for indices in class_indices.values():
        subset_indices.extend(indices)
        
    print(f"Total images in subset: {len(subset_indices)}", flush=True)
    
    # Split subset into train/val (85/15)
    import random
    random.seed(42)
    random.shuffle(subset_indices)
    split_idx = int(len(subset_indices) * 0.85)
    train_idxs = subset_indices[:split_idx]
    val_idxs = subset_indices[split_idx:]
    
    train_sub_dataset = train_split.select(train_idxs)
    val_sub_dataset = train_split.select(val_idxs)
    
    # Custom PyTorch Dataset wrapper
    class PlantVillageDatasetWrapper(Dataset):
        def __init__(self, hf_dataset, transform=None):
            self.hf_dataset = hf_dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.hf_dataset)
            
        def __getitem__(self, idx):
            item = self.hf_dataset[idx]
            img = item["image"]
            img = img.convert("RGB")
            label = item["class_idx"]
            if self.transform:
                img = self.transform(img)
            return img, label

    # Image transform for feature extraction (no augmentation needed for precomputed features)
    feature_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = PlantVillageDatasetWrapper(train_sub_dataset, transform=feature_transform)
    val_dataset = PlantVillageDatasetWrapper(val_sub_dataset, transform=feature_transform)
    
    # Setup MobileNetV3 Large backbone
    print("Downloading pretrained MobileNetV3 Large for feature extraction...", flush=True)
    backbone = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    backbone.eval()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = backbone.to(device)
    print(f"Using device for feature extraction: {device}", flush=True)
    
    # Feature extraction helper
    def extract_features(dataset, model_backbone, device_name):
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=0)
        all_features = []
        all_labels = []
        
        print(f"Extracting features for {len(dataset)} images...", flush=True)
        with torch.no_grad():
            for idx, (images, labels) in enumerate(loader):
                images = images.to(device_name)
                # Forward pass through backbone features, global average pool and flatten
                feats = model_backbone.features(images)
                feats = model_backbone.avgpool(feats)
                feats = torch.flatten(feats, 1)
                
                all_features.append(feats.cpu())
                all_labels.append(labels)
                if (idx + 1) % 15 == 0 or (idx + 1) == len(loader):
                    print(f"Processed batch {idx + 1}/{len(loader)}", flush=True)
                    
        return torch.cat(all_features, dim=0), torch.cat(all_labels, dim=0)
        
    # Extract features
    print("Extracting training features...", flush=True)
    train_features, train_labels = extract_features(train_dataset, backbone, device)
    
    print("Extracting validation features...", flush=True)
    val_features, val_labels = extract_features(val_dataset, backbone, device)
    
    print(f"Features ready. Shape: {train_features.shape}", flush=True)
    
    # Simple classifier head to train on precomputed features
    class ClassifierHead(nn.Module):
        def __init__(self, in_features, num_classes):
            super().__init__()
            self.classifier = nn.Sequential(
                nn.Linear(in_features, 1024),
                nn.Hardswish(),
                nn.Dropout(p=0.3),
                nn.Linear(1024, num_classes)
            )
        def forward(self, x):
            return self.classifier(x)
            
    in_features = backbone.classifier[0].in_features
    classifier_head = ClassifierHead(in_features, len(class_names)).to(device)
    
    # Feature Dataset & Dataloader
    class FeatureDataset(Dataset):
        def __init__(self, features, labels):
            self.features = features
            self.labels = labels
        def __len__(self):
            return len(self.features)
        def __getitem__(self, idx):
            return self.features[idx], self.labels[idx]
            
    feat_train_loader = DataLoader(FeatureDataset(train_features, train_labels), batch_size=64, shuffle=True)
    feat_val_loader = DataLoader(FeatureDataset(val_features, val_labels), batch_size=64, shuffle=False)
    
    # Loss, Optimizer and Scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_head.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.5)
    
    # Training Loop for Classifier Head (Runs in seconds!)
    epochs = 12
    print("Starting training of classifier head...", flush=True)
    for epoch in range(epochs):
        classifier_head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for images, labels in feat_train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = classifier_head(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        scheduler.step()
        epoch_loss = running_loss / len(train_features)
        epoch_acc = 100.0 * correct / total
        
        # Validation
        classifier_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for images, labels in feat_val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = classifier_head(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
                
        val_epoch_loss = val_loss / len(val_features)
        val_epoch_acc = 100.0 * val_correct / val_total
        
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}% | Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.2f}%", flush=True)
        
    # Copy trained weights back to the full model
    backbone.classifier = classifier_head.classifier
    
    # Save full model weights
    model_path = "backend/model/plant_disease_model.pth"
    torch.save(backbone.state_dict(), model_path)
    print(f"Model saved successfully to {model_path}", flush=True)

if __name__ == "__main__":
    main()
