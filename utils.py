# utils.py
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix

# Define transform (same preprocessing as EfficientNetB0 expects)
def preprocess_image(image_path, image_size=224):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet mean
                             std=[0.229, 0.224, 0.225])   # ImageNet std
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction function
def predict(model, image_path, device='cuda'):
    model.eval()
    model.to(device)
    image = preprocess_image(image_path).to(device)
    with torch.no_grad():
        outputs = model(image)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]
    return probs

# Plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(4, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

# Example usage
if __name__ == "__main__":
    import torchvision.models as models
    import os

    # Check available model files. Prioritize the best model saved during training.
    model_files = [
        "best_model.pth",
        "deepfake_efficientnetb0_final.pth", 
        "deepfake_efficientnetb0.pth"
    ]
    
    model_path = None
    for model_file in model_files:
        if os.path.exists(model_file):
            model_path = model_file
            print(f"Found model file: {model_path}")
            break
    
    if model_path is None:
        print("No trained model found. Please train the model first using model.py")
        print(f"Available files: {os.listdir('.')}")
        exit(1)

    # Load model with correct architecture
    model = models.efficientnet_b0(weights=None)
    # Modify classifier to match training setup
    model.classifier[1] = torch.nn.Sequential(
        torch.nn.Linear(model.classifier[1].in_features, 256),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 2)  # 2 classes: Fake, Real
    )
    
    try:
        # First try loading as full model
        try:
            model = torch.load(model_path, map_location='cuda', weights_only=False)
            print(f"Full model loaded successfully from {model_path}")
        except:
            # If that fails, try loading state dict
            model.load_state_dict(torch.load(model_path, map_location='cuda', weights_only=False))
            print(f"Model state dict loaded successfully from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)
    
    # Test with both fake and real images
    test_images = [
        ("data/Train/Fake/fake_1.jpg", "Fake"),
        ("data/Train/Real/real_1.jpg", "Real")
    ]
    class_names = ["Fake", "Real"]

    # Allow user to specify custom image path
    import sys
    if len(sys.argv) > 1:
        custom_image_path = sys.argv[1]
        print(f"\n🔍 Testing custom image: {custom_image_path}")
        if os.path.exists(custom_image_path):
            probs = predict(model, custom_image_path)
            predicted_class = class_names[np.argmax(probs)]
            confidence = np.max(probs) * 100
            print(f"Predicted probabilities: Fake={probs[0]:.4f}, Real={probs[1]:.4f}")
            print(f"Predicted class: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
        else:
            print(f"❌ Image not found: {custom_image_path}")
        exit(0)

    # Default test images
    for image_path, expected_class in test_images:
        if os.path.exists(image_path):
            print(f"\nTesting image: {image_path}")
            print(f"Expected class: {expected_class}")
            probs = predict(model, image_path)
            predicted_class = class_names[np.argmax(probs)]
            print(f"Predicted probabilities: {probs}")
            print(f"Predicted class: {predicted_class}")
            print(f"Correct prediction: {'✅' if predicted_class == expected_class else '❌'}")
        else:
            print(f"Test image not found: {image_path}")
            # Show available test images
            print("Available test images:")
            fake_dir = "data/Train/Fake"
            if os.path.exists(fake_dir):
                fake_images = [f for f in os.listdir(fake_dir) if f.endswith('.jpg')][:5]
                for img in fake_images:
                    print(f"  {os.path.join(fake_dir, img)}")
            else:
                print("  No fake images directory found")
