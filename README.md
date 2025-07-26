# 🛡️ VISION SHIELD - Advanced Deepfake Detection System

A comprehensive AI-powered deepfake detection system using PyTorch EfficientNet-B0 with support for both **image** and **video** analysis. Features multiple interfaces including web apps, command-line tools, and Python APIs.

## 🌟 Key Features

### 🖼️ **Image Detection**

- **🧠 EfficientNet-B0**: State-of-the-art CNN with 97.38% validation accuracy
- **📱 Web Interface**: Streamlit app with drag-and-drop upload
- **📊 Detailed Analysis**: Confidence scores, probability distributions, visual charts
- **🔧 Multiple Interfaces**: Web app, command-line tool, Python API
- **📈 Visualization**: Confidence charts and detailed result analysis

### 🎬 **Video Detection**

- **⏱️ Temporal Analysis**: Frame-by-frame analysis with consistency checking
- **👤 Face Detection**: Optional filtering to analyze only frames with faces
- **📊 Statistical Analysis**: Moving averages, pattern recognition, rapid change detection
- **🎯 Overall Verdict**: High-confidence assessment of video authenticity
- **📈 Comprehensive Reports**: Time-series plots, detailed metrics, downloadable results

### 🔧 **Technical Capabilities**

- **GPU Acceleration**: CUDA support for faster processing
- **Batch Processing**: Handle multiple files efficiently
- **Format Support**: Images (JPG, PNG, BMP, etc.) and Videos (MP4, AVI, MOV, etc.)
- **Cross-Platform**: Windows, Linux, macOS compatible

## 🚀 Quick Start

### 📱 **Image Detection Web App**

```bash
# Install dependencies
pip install -r requirements.txt

# Run the image detection app
streamlit run streamlit_pytorch_app.py
```

### 🎬 **Video Detection Web App**

```bash
# Run the video detection app
streamlit run streamlit_video_app.py
```

## 📋 Requirements

- **Python**: 3.8+
- **PyTorch**: 2.0+ with CUDA support (recommended)
- **OpenCV**: 4.0+ for video processing
- **Streamlit**: 1.28+ for web interfaces
- **Additional**: PIL, NumPy, Matplotlib, scikit-learn

## 📁 Project Structure

```
VISION-SHIELD/
├── 📱 WEB INTERFACES
│   ├── streamlit_pytorch_app.py    # Image detection web app
│   └── streamlit_video_app.py      # Video detection web app
├── 🧠 CORE MODELS & TRAINING
│   ├── model.py                    # Training script with GPU support
│   ├── best_model.pth             # Best checkpoint (97.38% accuracy)
│   └── deepfake_efficientnetb0_final.pth  # Final trained model
├── 🛠️ UTILITIES & TOOLS
│   ├── utils.py                   # Image testing utilities
│   ├── test_video.py             # Video analysis command-line tool
│   ├── video_deepfake_detector.py # Core video detection class
│   └── evaluate.py               # Model evaluation script
├── 📊 DATA & WEIGHTS
│   ├── data/
│   │   └── Train/
│   │       ├── Fake/             # Training fake images (70,001)
│   │       └── Real/             # Training real images (70,001)
│   └── weights/
│       └── Meso4_DF.h5          # Legacy MesoNet weights
├── 📚 DOCUMENTATION
│   ├── README.md                 # Main documentation (this file)
│   ├── VIDEO_DETECTION_README.md # Detailed video detection guide
│   └── requirements.txt          # Python dependencies
└── 🔧 CONFIGURATION
    └── app.py                    # Alternative launcher
```

## 🔧 How to Use

### 🖼️ **Image Detection**

#### **Web Interface (Recommended)**

1. Run `streamlit run streamlit_pytorch_app.py`
2. Open browser to `http://localhost:8501`
3. Upload image via drag-and-drop
4. Get instant results with confidence visualization

#### **Command Line**

```bash
# Single image analysis
python utils.py image.jpg

# Interactive mode
python utils.py

# Test with sample images
python -c "from utils import test_model; test_model()"
```

#### **Python API**

```python
from utils import load_model, predict_image
from PIL import Image

# Load model
model = load_model("best_model.pth")

# Analyze image
image = Image.open("test.jpg")
prediction, confidence, probabilities = predict_image(model, image)
print(f"Prediction: {prediction} ({confidence:.2f}% confidence)")
```

### 🎬 **Video Detection**

#### **Web Interface (Recommended)**

1. Run `streamlit run streamlit_video_app.py`
2. Open browser to `http://localhost:8502`
3. Upload video file (MP4, AVI, MOV, etc.)
4. Configure analysis parameters:
   - Frame skip interval
   - Maximum frames to analyze
   - Face detection on/off
5. Get comprehensive results with visualizations


## 📊 Model Architecture

### 🧠 **Core Model: EfficientNet-B0**

The system uses **EfficientNet-B0** with a custom classification head, achieving **97.38% validation accuracy**:

- **Base Architecture**: Pre-trained EfficientNet-B0 (ImageNet)
- **Input Size**: 224×224×3 RGB images
- **Custom Classifier**:
  - Linear(1280 → 256) + ReLU + Dropout(0.3)
  - Linear(256 → 2) for binary classification
- **Classes**: Real vs Fake
- **Training Dataset**: 140,002 images (70,001 Fake + 70,001 Real)

### 🎬 **Video Analysis Pipeline**

1. **Frame Extraction**: Extract frames at specified intervals
2. **Face Detection**: Optional Haar Cascade filtering
3. **Individual Classification**: EfficientNet-B0 analysis per frame
4. **Temporal Analysis**:
   - Moving average smoothing
   - Consistency scoring
   - Rapid change detection
   - Pattern recognition
5. **Statistical Aggregation**: Overall verdict with confidence levels

### 📈 **Performance Metrics**

| Metric                    | Value                         |
| ------------------------- | ----------------------------- |
| **Validation Accuracy**   | 97.38%                        |
| **Model Size**            | ~20MB                         |
| **Inference Speed (GPU)** | ~100ms per frame              |
| **Inference Speed (CPU)** | ~300ms per frame              |
| **Training Dataset**      | 140,002 images                |
| **Architecture**          | EfficientNet-B0 + Custom Head |

## 📈 Results Interpretation

### 🖼️ **Image Analysis Results**

- **Prediction**: Binary classification (Real/Fake)
- **Confidence**: Model certainty percentage (0-100%)
- **Probabilities**: Raw softmax outputs
  - **Fake Probability**: 0.0-1.0 (higher = more likely fake)
  - **Real Probability**: 0.0-1.0 (higher = more likely real)

### 🎬 **Video Analysis Results**

#### **Overall Verdict Categories**

- **"Likely Real Video"**: High confidence (>70% frames agree) - authentic content
- **"Likely Deepfake"**: High confidence (>70% frames agree) - fake content
- **"Uncertain/Mixed"**: Medium/Low confidence - requires manual review

#### **Confidence Levels**

- **High**: >70% of frames agree on classification
- **Medium**: 60-70% frame agreement
- **Low**: <60% frame agreement

#### **Temporal Analysis Metrics**

- **Consistency Score**: Temporal stability (0-1, higher = more consistent)
- **Rapid Changes**: Number of sudden prediction changes
- **Moving Average**: Smoothed probability over time
- **Pattern Recognition**: Detection of suspicious temporal patterns

#### **Statistical Outputs**

- **Frame-by-Frame Results**: Individual predictions with timestamps
- **Fake Percentage**: Proportion of frames classified as fake
- **Confidence Distributions**: Statistical analysis of prediction certainty
- **Face Detection Stats**: Frames with/without detected faces

## 🛠️ Installation & Setup

### 1️⃣ **Prerequisites**

- **Python 3.8+** (3.9+ recommended)
- **CUDA-compatible GPU** (optional but recommended for video analysis)
- **4GB+ RAM** (8GB+ recommended for large videos)

### 2️⃣ **Clone Repository**

```bash
git clone https://github.com/yourusername/vision-shield-deepfake-detection.git
cd vision-shield-deepfake-detection
```

### 3️⃣ **Install Dependencies**

```bash
# Install all required packages
pip install -r requirements.txt

# Verify PyTorch GPU support (optional)
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 4️⃣ **Quick Test**

```bash
# Test image detection
python utils.py

# Test video detection
python test_video.py --help

# Launch web interface
streamlit run streamlit_pytorch_app.py
```

## 🎯 Supported Formats

### 🖼️ **Image Formats**

- **Supported**: JPG, JPEG, PNG, BMP, TIFF, WEBP
- **Input Size**: Any resolution (auto-resized to 224×224)
- **Color**: RGB images (auto-converted if needed)
- **Recommended**: High-resolution face images for best accuracy

### 🎬 **Video Formats**

- **Fully Supported**: MP4, AVI, MOV, MKV, WMV
- **Limited Support**: FLV, WebM, 3GP
- **Recommended Settings**:
  - Resolution: 720p or higher
  - Duration: Under 5 minutes for faster analysis
  - Framerate: 24-30 FPS optimal
  - Quality: Avoid heavily compressed videos

## ⚡ Performance & Hardware

### 🖥️ **Hardware Recommendations**

| Hardware     | Image Detection | Video Detection | Notes                     |
| ------------ | --------------- | --------------- | ------------------------- |
| **RTX 3080** | Instant         | ~15 FPS         | Optimal performance       |
| **GTX 1650** | Instant         | ~10 FPS         | Good performance          |
| **CPU (i7)** | <1 second       | ~3 FPS          | Adequate for small videos |
| **CPU (i5)** | 1-2 seconds     | ~2 FPS          | Basic functionality       |

### 📊 **Video Analysis Scaling**

| Video Length | Frames Analyzed | Est. Time (GPU) | Est. Time (CPU) |
| ------------ | --------------- | --------------- | --------------- |
| 30 seconds   | 30 frames       | 3 seconds       | 10 seconds      |
| 2 minutes    | 120 frames      | 12 seconds      | 40 seconds      |
| 5 minutes    | 300 frames      | 30 seconds      | 100 seconds     |

_Based on default settings (frame_skip=30, max_frames=100)_

## 🚨 Troubleshooting

### ❗ **Common Issues**

#### **CUDA Out of Memory**

```
RuntimeError: CUDA out of memory
```

**Solutions:**

- Reduce `max_frames` parameter for video analysis
- Increase `frame_skip` interval
- Use CPU inference: `device = 'cpu'`

#### **Model Loading Errors**

```
Error loading model: No trained model found
```

**Solutions:**

- Ensure `best_model.pth` exists in project directory
- Train model first: `python model.py`
- Check file permissions and paths

#### **Video Processing Issues**

```
Error: Could not open video file
```

**Solutions:**

- Verify video format is supported
- Check file path and permissions
- Install additional codecs if needed
- Try converting to MP4 format

#### **Streamlit Port Conflicts**

```
Port 8501 is already in use
```

**Solutions:**

- Use different port: `streamlit run app.py --server.port 8502`
- Kill existing processes: `taskkill /f /im streamlit.exe` (Windows)
- Check for other Streamlit instances

### 🔍 **Debug Mode**



## 🚨 Important Limitations

### ⚠️ **Model Limitations**

- **Accuracy**: 97.38% validation accuracy (2.62% error rate is normal)
- **Domain**: Optimized for face-focused images and videos
- **Quality**: Performance depends on image/video resolution and lighting
- **Generalization**: May not detect all types of deepfakes (evolving field)

### ⚠️ **Technical Limitations**

- **Processing Speed**: Video analysis is computationally intensive
- **Memory**: Large videos may require significant RAM
- **Face Detection**: Face-focused content provides better results
- **Format Support**: Some video codecs may not be supported

### ⚠️ **Ethical Considerations**

- **False Positives**: Always verify results with multiple methods
- **Human Review**: Critical applications require manual verification
- **Bias**: Model trained on specific datasets may have biases
- **Evolution**: Deepfake technology evolves rapidly, model may become outdated

## 🤝 Contributing

### 🛠️ **Development Setup**

```bash
# Clone repository
git clone https://github.com/yourusername/vision-shield-deepfake-detection.git
cd vision-shield-deepfake-detection

# Install development dependencies
pip install -r requirements.txt
pip install black pytest flake8  # Development tools



# Code formatting
black *.py
```

### 📝 **How to Contribute**

1. **Fork** the repository on GitHub
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Make** your changes and test thoroughly
4. **Commit** changes (`git commit -m 'Add amazing feature'`)
5. **Push** to the branch (`git push origin feature/amazing-feature`)
6. **Submit** a Pull Request

### � **Bug Reports**

When reporting bugs, please include:

- **System information** (OS, Python version, GPU model)
- **Error messages** (full traceback)
- **Steps to reproduce** the issue
- **Sample files** if relevant (images/videos that cause issues)
- **Expected vs actual behavior**

### � **Feature Requests**

- **Describe** the use case and expected behavior
- **Explain** why this feature would be valuable
- **Provide** examples or mockups if applicable
- **Consider** performance implications for video analysis

## 🔗 Advanced Usage

### 🧪 **Model Training**

```bash
# Train new model (requires dataset)
python model.py

# Evaluate model performance
python evaluate.py

# Compare different model checkpoints
python utils.py --model best_model.pth
python utils.py --model deepfake_efficientnetb0_final.pth
```

### � **Custom Configuration**

```python
# Custom video analysis parameters
from video_deepfake_detector import VideoDeepfakeDetector

detector = VideoDeepfakeDetector(model_path="custom_model.pth")

# Advanced analysis with custom settings
results = detector.analyze_video(
    video_path="video.mp4",
    frame_skip=15,           # Analyze every 15th frame
    max_frames=500,          # Analyze up to 500 frames
    face_detection=True,     # Only analyze frames with faces
    temporal_window=10       # Moving average window size
)
```

### 📊 **Batch Processing**

```python
# Process multiple videos
import os
from video_deepfake_detector import VideoDeepfakeDetector

detector = VideoDeepfakeDetector()
video_folder = "path/to/videos"

for video_file in os.listdir(video_folder):
    if video_file.endswith(('.mp4', '.avi', '.mov')):
        video_path = os.path.join(video_folder, video_file)
        results = detector.analyze_video(video_path)
        print(f"{video_file}: {results['overall_verdict']['verdict']}")
```

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### 📜 **License Summary**

- ✅ **Commercial use** allowed
- ✅ **Modification** allowed
- ✅ **Distribution** allowed
- ✅ **Private use** allowed
- ❗ **No warranty** provided
- ❗ **No liability** assumed

## 🙏 Acknowledgments

### 📚 **Research & Papers**

- **EfficientNet**: Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks.
- **Original MesoNet**: Afchar, D., Nozick, V., Yamagishi, J., & Echizen, I. (2018). MesoNet: a Compact Facial Video Forgery Detection Network.
- **Deepfake Detection Research**: Various papers and datasets from the computer vision community

### 🛠️ **Technologies & Frameworks**

- **PyTorch**: The PyTorch team for the deep learning framework
- **OpenCV**: Intel Corporation for computer vision libraries
- **Streamlit**: Streamlit Inc. for the web application framework
- **EfficientNet**: Google Research for the base architecture
- **Python Community**: For all the amazing open-source libraries

### 🎯 **Datasets & Resources**

- **Training Dataset**: Contributors to the deepfake detection datasets
- **Face Detection**: Haar Cascade classifiers from OpenCV
- **GPU Computing**: NVIDIA for CUDA toolkit and GPU support



- **📚 Documentation**: Check this README and `VIDEO_DETECTION_README.md`
- **🐛 Issues**: [GitHub Issues](https://github.com/yourusername/vision-shield-deepfake-detection/issues)
- **💡 Discussions**: [GitHub Discussions](https://github.com/yourusername/vision-shield-deepfake-detection/discussions)
- **📧 Email**: koushikaadhy@gmail.com

### 🔍 **Troubleshooting Steps**

1. **Check** the troubleshooting section above
2. **Search** existing GitHub issues
3. **Provide** detailed information when creating new issues
4. **Test** with sample images/videos first

### 🌟 **Stay Updated**

- **⭐ Star** this repository for updates
- **👀 Watch** for new releases and features
- **🔔 Follow** for announcements

---

## 🏆 **Project Status**

| Feature                | Status          | Notes                                      |
| ---------------------- | --------------- | ------------------------------------------ |
| **Image Detection**    | ✅ **Complete** | 97.38% accuracy, multiple interfaces       |
| **Video Detection**    | ✅ **Complete** | Full temporal analysis pipeline            |
| **Web Interfaces**     | ✅ **Complete** | Streamlit apps for both image and video    |
| **Command Line Tools** | ✅ **Complete** | Full CLI support with all options          |
| **GPU Acceleration**   | ✅ **Complete** | CUDA support for faster processing         |
| **Documentation**      | ✅ **Complete** | Comprehensive guides and examples          |
| **Model Training**     | ✅ **Complete** | Full training pipeline with early stopping |
| **Batch Processing**   | ✅ **Complete** | Multiple file processing capabilities      |

---

**⚡ Made with ❤️ using Python, PyTorch, and Streamlit**

**🛡️ VISION SHIELD - Protecting against deepfake deception with advanced AI**

> ⚠️ **Important Note**: This tool is designed for research and educational purposes. Always verify results with multiple methods and human review for critical applications. Deepfake detection is an evolving field, and no tool is 100% accurate.
