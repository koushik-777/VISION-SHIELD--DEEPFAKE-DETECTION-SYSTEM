import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set page config
st.set_page_config(
    page_title="VISION SHIELD - Deepfake Detection System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


@st.cache_resource
def load_pytorch_model():
    """Load the pre-trained PyTorch model"""
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        
        model_files = [
            os.path.join(script_dir, "best_model.pth"),
            os.path.join(script_dir, "deepfake_efficientnetb0_final.pth")
        ]
        
        model_path = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                break
        
        if model_path is None:
            
            available_files = [f for f in os.listdir(script_dir) if f.endswith('.pth')]
            return None, False, f"No trained model found. Available .pth files: {available_files}"
        
        
        try:
            model = torch.load(model_path, map_location='cpu', weights_only=False)
            model.eval()
            return model, True, f"Full model loaded from {model_path}"
        except:
            
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Sequential(
                nn.Linear(model.classifier[1].in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)  # 2 classes: Fake, Real
            )
            model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=False))
            model.eval()
            return model, True, f"Model state dict loaded from {model_path}"
            
    except Exception as e:
        return None, False, f"Error loading model: {str(e)}"

def predict_image(model, image, device='cpu'):
    """Make prediction on uploaded image"""                                  
    try:
        # Preprocess image
        transform = get_transforms()
        
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
        
        # Class names
        class_names = ["Fake", "Real"]
        prediction = class_names[predicted_class]
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].cpu().numpy(),
            'class_names': class_names
        }
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None

def create_confidence_chart(probabilities, class_names):
    """Create a confidence chart"""
    fig, ax = plt.subplots(figsize=(8, 4))
    
    # Create bar chart
    bars = ax.bar(class_names, probabilities * 100, 
                  color=['#ff6b6b', '#51cf66'], alpha=0.8)
    
    
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Prediction Confidence by Class')
    ax.set_ylim(0, 100)
    
    
    for bar, prob in zip(bars, probabilities):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{height:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    
    st.title("VISION SHEILD-Deepfake Detection System")
    st.markdown("Upload an image to check if it's real or a deepfake using our EfficientNet-B0 model.")
    
    # Sidebar
    st.sidebar.title("How it works")
    st.sidebar.markdown("""
    1. **Upload** an image using the file uploader
    2. **Wait** for the VISION SHIELD model to analyze it
    3. **Get results** showing if it's real or deepfake
    4. **View** the confidence scores and visualization
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model Information")
    st.sidebar.markdown("""
    - **Architecture**: EfficientNet-B0
    - 
    - **Classes**: Fake, Real
    - **Input Size**: 224×224 pixels
    - **Preprocessing**: ImageNet normalization
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Understanding Results")
    st.sidebar.markdown("""
    - **Prediction**: Final classification
    - **Confidence**: Model certainty (%)
    - **Probabilities**: Raw model outputs
    """)
    
    # Load model
    with st.spinner("Loading PyTorch model..."):
        model, model_loaded, load_message = load_pytorch_model()
    
    if not model_loaded or model is None:
        st.error(f"Failed to load the  model: {load_message}")
        st.info("Make sure you have trained the model using model.py first!")
        st.stop()
    
    st.success(f"{load_message}")
    
    st.success(f"✅ {load_message}")
    
    # Set device (CPU only as requested)
    device = 'cpu'
    model = model.to(device)
    
    # File uploader
    st.markdown("---")
    st.subheader("Upload Your Image")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'],
        help="Upload an image to analyze for deepfake detection"
    )
    
    if uploaded_file is not None:
        # Create columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Original Image")
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            
            # Image info
            st.info(f"""
            **Image Information:**
            - Size: {image.size[0]} × {image.size[1]} pixels
            - Mode: {image.mode}
            - Format: {uploaded_file.type}
            """)
        
        with col2:
            st.subheader("🔍 Analysis Results")
            
            # Analyze image
            with st.spinner("Analyzing image with PyTorch model..."):
                result = predict_image(model, image, device)
            
            if result is not None:
                # Display results
                st.markdown("###Detection Results")
                
                # Classification with colored badge
                if result['prediction'] == "Real":
                    st.success(f"**{result['prediction']}**")
                    result_color = "success"
                else:
                    st.error(f"**{result['prediction']}**")
                    result_color = "error"
                
                # Metrics
                col2a, col2b = st.columns(2)
                with col2a:
                    st.metric("Prediction", result['prediction'])
                with col2b:
                    st.metric("Confidence", f"{result['confidence']:.2f}%")
                
                # Progress bar for confidence
                st.progress(result['confidence'] / 100)
                
                # Detailed probabilities
                st.markdown("### Detailed Probabilities")
                prob_col1, prob_col2 = st.columns(2)
                
                with prob_col1:
                    fake_prob = result['probabilities'][0] * 100
                    st.metric("Fake Probability", f"{fake_prob:.2f}%")
                
                with prob_col2:
                    real_prob = result['probabilities'][1] * 100
                    st.metric("Real Probability", f"{real_prob:.2f}%")
                
                # Confidence chart
                st.markdown("### Confidence Visualization")
                fig = create_confidence_chart(result['probabilities'], result['class_names'])
                st.pyplot(fig)
                
                # Detailed explanation
                st.markdown("---")
                if result['prediction'] == "Fake":
                    st.warning(f"""
                    **This image appears to be a DEEPFAKE**
                    
                    - VISION SHIELD is **{result['confidence']:.1f}% confident** this is fake
                    - Fake probability: **{result['probabilities'][0]*100:.2f}%**
                    - Real probability: **{result['probabilities'][1]*100:.2f}%**
                    
                    The AI detected patterns suggesting this image may have been artificially generated or manipulated.
                    """)
                else:
                    st.success(f"""
                     **This image appears to be REAL**
                    
                    - VISION SHIELD is **{result['confidence']:.1f}% confident** this is real
                    - Real probability: **{result['probabilities'][1]*100:.2f}%**
                    - Fake probability: **{result['probabilities'][0]*100:.2f}%**
                    
                    The AI found patterns consistent with authentic, unmanipulated images.
                    """)
                
            else:
                st.error("Failed to analyze the image. Please try uploading a different image.")
    
    else:
        # Show example or instructions when no file is uploaded
        st.info("Please upload an image file to start the deepfake detection analysis.")
        
        # Example images section
        st.markdown("---")
        st.subheader("Try with Sample Images")
        st.markdown("You can test the system with images from your training dataset:")
        
        sample_col1, sample_col2 = st.columns(2)
        
        with sample_col1:
            if st.button("🔍 Analyze Sample Fake Image"):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                sample_paths = [
                    os.path.join(script_dir, "data", "Train", "Fake", "fake_1.jpg"),
                    os.path.join(script_dir, "data", "Train", "Fake", "fake_0.jpg"),
                    os.path.join(script_dir, "data", "Train", "Fake", "fake_10.jpg")
                ]
                
                sample_path = None
                for path in sample_paths:
                    if os.path.exists(path):
                        sample_path = path
                        break
                
                if sample_path:
                    sample_image = Image.open(sample_path)
                    st.image(sample_image, caption="Sample Fake Image", width=300)
                    
                    with st.spinner("Analyzing sample fake image..."):
                        result = predict_image(model, sample_image, device)
                    
                    if result is not None:
                        if result['prediction'] == "Fake":
                            st.success(f"Correctly identified as {result['prediction']} ({result['confidence']:.2f}% confidence)")
                        else:
                            st.warning(f"Incorrectly identified as {result['prediction']} ({result['confidence']:.2f}% confidence)")
                else:
                    st.error("Sample fake images not found in expected locations")
        
        with sample_col2:
            if st.button("🔍 Analyze Sample Real Image"):
                script_dir = os.path.dirname(os.path.abspath(__file__))
                sample_paths = [
                    os.path.join(script_dir, "data", "Train", "Real", "real_1.jpg"),
                    os.path.join(script_dir, "data", "Train", "Real", "real_0.jpg"),
                    os.path.join(script_dir, "data", "Train", "Real", "real_10.jpg")
                ]
                
                sample_path = None
                for path in sample_paths:
                    if os.path.exists(path):
                        sample_path = path
                        break
                
                if sample_path:
                    sample_image = Image.open(sample_path)
                    st.image(sample_image, caption="Sample Real Image", width=300)
                    
                    with st.spinner("Analyzing sample real image..."):
                        result = predict_image(model, sample_image, device)
                    
                    if result is not None:
                        if result['prediction'] == "Real":
                            st.success(f"Correctly identified as {result['prediction']} ({result['confidence']:.2f}% confidence)")
                        else:
                            st.warning(f"Incorrectly identified as {result['prediction']} ({result['confidence']:.2f}% confidence)")
                else:
                    st.error("Sample real images not found in expected locations")

    # Model details section
    st.markdown("---")
    with st.expander("Technical Details"):
        st.markdown(f"""
        **Model Architecture:** EfficientNet-B0 with custom classifier
        - Base model: Pre-trained EfficientNet-B0
        - Custom classifier: Linear(1280→256) → ReLU → Dropout(0.3) → Linear(256→2)
        - Classes: {['Fake', 'Real']}
        - Device: {device.upper()}
        
        **Preprocessing:**
        - Resize to 224×224 pixels
        - Convert to tensor
        - Normalize with ImageNet statistics
        
        **Training Details:**
        - Framework: PyTorch
        - Loss function: CrossEntropyLoss
        - Optimizer: Adam
        - Dataset: 140,002 images (70,001 Fake + 70,001 Real)
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        Powered by  EfficientNet-B0 | Built with Streamlit
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
