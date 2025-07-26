import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from video_deepfake_detector import VideoDeepfakeDetector
import tempfile
import json
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Video Deepfake Detection System",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_resource
def load_video_detector():
    """Load the video deepfake detector"""
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
            return None, f"No trained model found. Available .pth files: {available_files}"
        
        detector = VideoDeepfakeDetector(model_path=model_path)
        return detector, f"Model loaded from {os.path.basename(model_path)}"
        
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def create_results_visualization(results):
    """Create visualization plots for the analysis results"""
    if not results['frame_results']:
        return None
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Video Analysis Results", fontsize=16, fontweight='bold')
    
    # Plot 1: Fake probability over time
    timestamps = [r['timestamp'] for r in results['frame_results']]
    fake_probs = [r['fake_prob'] for r in results['frame_results']]
    
    axes[0, 0].plot(timestamps, fake_probs, 'b-', alpha=0.7, linewidth=2)
    axes[0, 0].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Decision Threshold')
    axes[0, 0].set_xlabel('Time (seconds)')
    axes[0, 0].set_ylabel('Fake Probability')
    axes[0, 0].set_title('Deepfake Probability Over Time')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Distribution of predictions
    predictions = [r['prediction'] for r in results['frame_results']]
    pred_counts = {pred: predictions.count(pred) for pred in set(predictions)}
    
    colors = ['#ff6b6b' if pred == 'Fake' else '#51cf66' for pred in pred_counts.keys()]
    axes[0, 1].bar(pred_counts.keys(), pred_counts.values(), color=colors, alpha=0.8)
    axes[0, 1].set_ylabel('Number of Frames')
    axes[0, 1].set_title('Frame Predictions Distribution')
    
    # Plot 3: Confidence distribution
    confidences = [r['confidence'] for r in results['frame_results']]
    axes[1, 0].hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 0].set_xlabel('Confidence Score')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Confidence Score Distribution')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Moving average (if available)
    if 'temporal_analysis' in results and 'moving_averages' in results['temporal_analysis']:
        moving_avg = results['temporal_analysis']['moving_averages']
        moving_timestamps = timestamps[:len(moving_avg)]
        
        axes[1, 1].plot(moving_timestamps, moving_avg, 'g-', linewidth=2, label='Moving Average')
        axes[1, 1].plot(timestamps, fake_probs, 'b-', alpha=0.3, label='Raw Probabilities')
        axes[1, 1].axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold')
        axes[1, 1].set_xlabel('Time (seconds)')
        axes[1, 1].set_ylabel('Fake Probability')
        axes[1, 1].set_title('Temporal Analysis (Moving Average)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, 'Temporal Analysis\nNot Available', 
                       ha='center', va='center', transform=axes[1, 1].transAxes,
                       fontsize=12, style='italic')
        axes[1, 1].set_title('Temporal Analysis')
    
    plt.tight_layout()
    return fig

def main():
    st.title("VISION-SHIELD Deepfake Detection System")
    st.markdown("Upload a video to analyze for deepfake content using advanced temporal analysis and face detection.")
    
    # Sidebar
    st.sidebar.title(" Analysis Settings")
    
    # Load detector
    with st.spinner("Loading video detection model..."):
        detector, load_message = load_video_detector()
    
    if detector is None:
        st.error(f" Failed to load model: {load_message}")
        st.info(" Make sure you have trained the model using model.py first!")
        st.stop()
    
    st.success(f"✅ {load_message}")
    
    # Device info
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        st.info(f" Using GPU: {torch.cuda.get_device_name()}")
    else:
        st.info("💻 Using CPU for inference")
    
    # Analysis parameters
    st.sidebar.markdown("###  Analysis Parameters")
    frame_skip = st.sidebar.slider(
        "Frame Skip Interval", 
        min_value=1, max_value=120, value=30,
        help="Analyze every Nth frame (30 = ~1 FPS for 30fps video)"
    )
    
    max_frames = st.sidebar.number_input(
        "Max Frames to Analyze", 
        min_value=10, max_value=1000, value=100,
        help="Limit analysis to this many frames"
    )
    
    face_detection = st.sidebar.checkbox(
        "Enable Face Detection", 
        value=True,
        help="Only analyze frames containing detected faces"
    )
    
    st.sidebar.markdown("### How it works")
    st.sidebar.markdown("""
    1. **Upload** a video file
    2. **Configure** analysis parameters  
    3. **Wait** for frame-by-frame analysis
    4. **Review** comprehensive results with:
       - Frame-by-frame predictions
       - Temporal consistency analysis
       - Statistical summaries
       - Visualization plots
    """)
    
    # File uploader
    st.markdown("---")
    st.subheader("Upload Video File")
    
    uploaded_file = st.file_uploader(
        "Choose a video file",
        type=['mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'],
        help="Upload a video file to analyze for deepfake content"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{uploaded_file.name.split(".")[-1]}') as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_video_path = tmp_file.name
        
        # Display video info
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Video Information")
            
            # Get video properties
            cap = cv2.VideoCapture(temp_video_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = total_frames / fps if fps > 0 else 0
                cap.release()
                
                st.info(f"""
                **File:** {uploaded_file.name}  
                **Duration:** {duration:.2f} seconds  
                **Resolution:** {width}×{height}  
                **FPS:** {fps:.2f}  
                **Total Frames:** {total_frames:,}  
                **File Size:** {uploaded_file.size / (1024*1024):.2f} MB
                """)
                
                # Estimate analysis time
                frames_to_analyze = min(total_frames // frame_skip, max_frames)
                estimated_time = frames_to_analyze * 0.1  # Rough estimate: 0.1 sec per frame
                st.warning(f"⏱️ Estimated analysis time: ~{estimated_time:.1f} seconds for {frames_to_analyze} frames")
            
            # Display first frame as preview
            cap = cv2.VideoCapture(temp_video_path)
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                st.image(frame_rgb, caption="Video Preview (First Frame)", use_column_width=True)
            cap.release()
        
        with col2:
            st.subheader("Start Analysis")
            
            if st.button("Analyze Video for Deepfakes", type="primary"):
                try:
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Start analysis
                    status_text.text("🔄 Initializing video analysis...")
                    
                    # Analyze video
                    results = detector.analyze_video(
                        video_path=temp_video_path,
                        frame_skip=frame_skip,
                        max_frames=max_frames,
                        face_detection=face_detection
                    )
                    
                    progress_bar.progress(1.0)
                    status_text.text("Analysis complete!")
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("📊 Analysis Results")
                    
                    if 'statistics' in results and results['frame_results']:
                        stats = results['statistics']
                        verdict = results.get('overall_verdict', {})
                        
                        # Overall verdict
                        st.markdown("### 🎯 Overall Assessment")
                        
                        verdict_text = verdict.get('verdict', 'Unknown')
                        confidence_level = verdict.get('confidence_level', 'Unknown')
                        fake_percentage = verdict.get('fake_percentage', 0)
                        
                        if 'Deepfake' in verdict_text:
                            st.error(f" **{verdict_text}** (Confidence: {confidence_level})")
                        elif 'Real' in verdict_text:
                            st.success(f" **{verdict_text}** (Confidence: {confidence_level})")
                        else:
                            st.warning(f" **{verdict_text}** (Confidence: {confidence_level})")
                        
                        # Key metrics
                        st.markdown("###  Key Metrics")
                        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                        
                        with metric_col1:
                            st.metric("Fake Frames", f"{stats['fake_frames']}", f"{fake_percentage:.1f}%")
                        
                        with metric_col2:
                            st.metric("Real Frames", f"{stats['real_frames']}", f"{stats['real_percentage']:.1f}%")
                        
                        with metric_col3:
                            st.metric("Analyzed Frames", f"{stats['total_analyzed_frames']}")
                        
                        with metric_col4:
                            st.metric("Frames with Faces", f"{stats['frames_with_faces']}")
                        
                        # Detailed statistics
                        st.markdown("### Detailed Statistics")
                        detail_col1, detail_col2 = st.columns(2)
                        
                        with detail_col1:
                            st.markdown(f"""
                            **Confidence Scores:**
                            - Average Fake Confidence: {stats['avg_fake_confidence']:.3f}
                            - Maximum Fake Confidence: {stats['max_fake_confidence']:.3f}
                            - Average Real Confidence: {stats['avg_real_confidence']:.3f}
                            - Maximum Real Confidence: {stats['max_real_confidence']:.3f}
                            """)
                        
                        with detail_col2:
                            if 'temporal_analysis' in results:
                                temporal = results['temporal_analysis']
                                st.markdown(f"""
                                **Temporal Analysis:**
                                - Consistency Score: {temporal['consistency_score']:.3f}
                                - Pattern: {temporal['temporal_pattern']}
                                - Rapid Changes: {temporal['rapid_changes']}
                                - Standard Deviation: {temporal['std_deviation']:.3f}
                                """)
                        
                        # Visualization
                        st.markdown("### Analysis Visualization")
                        
                        fig = create_results_visualization(results)
                        if fig:
                            st.pyplot(fig)
                        
                        # Frame-by-frame results (sample)
                        st.markdown("### Frame-by-Frame Results (Sample)")
                        
                        # Show first 10 frame results
                        sample_results = results['frame_results'][:10]
                        
                        frame_data = []
                        for frame_result in sample_results:
                            frame_data.append({
                                'Frame': frame_result['frame_number'],
                                'Timestamp (s)': f"{frame_result['timestamp']:.2f}",
                                'Prediction': frame_result['prediction'],
                                'Confidence': f"{frame_result['confidence']:.3f}",
                                'Fake Prob': f"{frame_result['fake_prob']:.3f}",
                                'Real Prob': f"{frame_result['real_prob']:.3f}",
                                'Faces': frame_result.get('faces_detected', 'N/A')
                            })
                        
                        st.dataframe(frame_data)
                        
                        if len(results['frame_results']) > 10:
                            st.info(f"Showing first 10 results. Total analyzed frames: {len(results['frame_results'])}")
                        
                        # Download results
                        st.markdown("### Download Results")
                        
                        # Prepare JSON download
                        results_json = json.dumps(results, indent=2, default=str)
                        st.download_button(
                            label="Download Full Results (JSON)",
                            data=results_json,
                            file_name=f"video_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json"
                        )
                        
                    else:
                        st.warning("No frames were analyzed. Try adjusting the analysis parameters.")
                
                except Exception as e:
                    st.error(f" Error during analysis: {str(e)}")
                    st.info("Try reducing the number of frames to analyze or check the video format.")
                
                finally:
                    # Clean up temporary file
                    try:
                        os.unlink(temp_video_path)
                    except:
                        pass
    
    else:
        # Instructions when no file is uploaded
        st.info("Please upload a video file to start the deepfake detection analysis.")
        
        st.markdown("---")
        st.subheader("Supported Features")
        
        feature_col1, feature_col2 = st.columns(2)
        
        with feature_col1:
            st.markdown("""
            **🎯 Detection Capabilities:**
            - Frame-by-frame deepfake analysis
            - Temporal consistency checking
            - Face detection filtering
            - Statistical analysis
            - Confidence scoring
            """)
        
        with feature_col2:
            st.markdown("""
            **Analysis Outputs:**
            - Overall video verdict
            - Time-series probability plots
            - Confidence distributions
            - Temporal pattern analysis
            - Downloadable JSON results
            """)
        
        st.markdown("---")
        st.subheader("Supported Video Formats")
        st.markdown("""
        **Common formats:** MP4, AVI, MOV, MKV, WMV, FLV, WebM
        
        **Recommendations:**
        - Use videos with clear faces for best results
        - Higher resolution videos may provide better detection accuracy
        - Shorter videos (under 1 minute) will analyze faster
        """)

    # Technical details
    st.markdown("---")
    with st.expander("Technical Details"):
        st.markdown(f"""
        **Video Analysis Pipeline:**
        - **Frame Extraction:** Extract frames at specified intervals
        - **Face Detection:** Optional filtering using Haar cascades
        - **Individual Prediction:** Each frame analyzed with EfficientNet-B0
        - **Temporal Analysis:** Statistical analysis across time
        - **Pattern Detection:** Identify suspicious temporal patterns
        
        **Model Information:**
        - Architecture: EfficientNet-B0 with custom classifier
        - Input Size: 224×224 pixels
        - Classes: Fake, Real
        - Device: {device.upper()}
        
        **Analysis Features:**
        - Moving average smoothing
        - Variance and consistency scoring
        - Rapid change detection
        - Statistical aggregation
        """)

if __name__ == "__main__":
    main()
