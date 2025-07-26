import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import os
import copy
import time
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Parameters
data_dir = r"D:\df\df-main\data\Train"
input_size = 192  # Reduced resolution
batch_size = 32
num_classes = 2
num_epochs = 10  # Reduced for time constraint
learning_rate = 1e-3

# Transforms
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(input_size),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transforms = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Dataset loading with subsample
print("Loading dataset...")
full_dataset = datasets.ImageFolder(os.path.join(data_dir), transform=train_transforms)
print(f"Total images: {len(full_dataset)} | Classes: {full_dataset.classes}")

# Subsample 50% for faster training
indices = np.random.choice(len(full_dataset), size=int(0.5 * len(full_dataset)), replace=False)
subset = Subset(full_dataset, indices)

# Train-validation split
val_size = int(0.2 * len(subset))
train_size = len(subset) - val_size
train_set, val_set = torch.utils.data.random_split(subset, [train_size, val_size])
train_set.dataset.transform = train_transforms
val_set.dataset.transform = val_transforms

# DataLoaders
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Model setup
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Sequential(
    nn.Linear(model.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(256, num_classes)
)
model = model.to(device)

# Loss, optimizer, scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
scaler = GradScaler()

# Train function
def train_model(model, train_loader, val_loader, num_epochs):
    best_acc = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    history = {"train_acc": [], "val_acc": [], "train_loss": [], "val_loss": []}
    patience = 4
    wait = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 20)

        for phase in ['train', 'val']:
            model.train() if phase == 'train' else model.eval()
            loader = train_loader if phase == 'train' else val_loader

            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            loop = tqdm(loader, desc=f"{phase.capitalize()} [{epoch+1}]")
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'), autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                if phase == 'train':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)

            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples

            print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc.item())

            if phase == 'val':
                scheduler.step(epoch_loss)
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), "best_model.pth")
                    print("New best model saved.")
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        print("Early stopping triggered.")
                        model.load_state_dict(best_model_wts)
                        return history

    model.load_state_dict(best_model_wts)
    torch.save(model, "deepfake_efficientnetb0_final.pth")
    print("Training complete. Final model saved.")
    return history

# Plotting function
def plot_history(history):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.tight_layout()
    plt.show()

# Main
if __name__ == "__main__":
    if os.path.exists(data_dir):
        history = train_model(model, train_loader, val_loader, num_epochs)
        plot_history(history)
    else:
        print("Error: Dataset path not found.")
