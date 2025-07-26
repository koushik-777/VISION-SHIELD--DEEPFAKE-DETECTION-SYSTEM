# 🛡️ VISION SHIELD – Deepfake Detection System  
🔗 **Demo (For Showcase Only)**

⚠️ *This is a demo deployment. For full functionality, clone the repo and run locally.*

---

## 🌟 Key Features

### 🖼️ Image Detection
- **EfficientNet-B0 (PyTorch)** – 97.38% validation accuracy
- **Streamlit Web App** – Drag-and-drop support
- **Rich Outputs** – Prediction, confidence score, probability chart
- **Multiple Interfaces** – Web app, CLI, Python API

### 🎬 Video Detection
- **Frame-by-frame classification**
- **Optional face-only analysis**
- **Temporal smoothing & pattern detection**
- **Statistical reports + final verdict**

---
## 🛠️ Tech Stack

- **Python 3.9+**
- **PyTorch** – Deep learning framework for model training and inference
- **Torchvision** – Pretrained models and image transforms
- **Streamlit** – Interactive web app interface
- **OpenCV** – Video and image processing (for video detection)
- **NumPy** – Numerical operations
- **Pillow** – Image handling
- **Matplotlib & Seaborn** – Visualization and plotting

## 🚀 Quick Usage (Local Setup)

```bash
git clone https://github.com/yourusername/vision-shield-deepfake-detection.git
cd vision-shield-deepfake-detection
pip install -r requirements.txt
```

### 🔍 Image Detection

```bash
streamlit run streamlit_pytorch_app.py      # Web app
python utils.py image.jpg                   # CLI
```

### 🎥 Video Detection

```bash
streamlit run streamlit_video_app.py
python test_video.py --video path/to/video.mp4
```

---

## 🧠 Model Overview

- **Architecture:** EfficientNet-B0 + Custom Classifier
- **Classes:** Real vs Fake
- **Accuracy:** 97.38%
- **Dataset:** 140,002 face images (balanced real/fake)

---

## 📈 Performance Snapshot

| Mode         | GPU (RTX 3080) | CPU (i7)   |
|--------------|----------------|------------|
| Image        | Instant        | <1 sec     |
| Video (5min) | ~30 sec        | ~100 sec   |

---

## 📊 Output Summary

- **Image:** Prediction + confidence + softmax distribution
- **Video:**
  - Frame-wise verdict
  - Confidence trend graph
  - Verdict: Real / Fake / Uncertain

---

## 🧪 Advanced Use

- Custom scripts for training, batch processing, and CLI tools
- Python APIs for model integration
- Supports JPG, PNG, MP4, AVI, and more

---

## ⚠️ Limitations

- Accuracy depends on face visibility, video quality
- Trained on specific datasets – may not generalize to all fakes
- Human review advised for critical use

---

## 📄 License

MIT License – Free for commercial and personal use

---

💡 **Reminder:**  
👉 *Clone and run locally for full model functionality. The hosted demo is limited to UI
