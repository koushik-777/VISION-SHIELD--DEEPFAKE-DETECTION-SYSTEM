import torch
import torch.nn as nn
from torchvision import models, transforms
import cv2
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import json
from datetime import datetime

class VideoDeepfakeDetector:
    def __init__(self, model_path="best_model.pth", device=None):
        """
        Initialize video deepfake detector
        
        Args:
            model_path (str): Path to the trained model
            device (str): Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
        self.class_names = ["Fake", "Real"]
        
    def _load_model(self, model_path):
        """Load the trained PyTorch model"""
        try:
            # Try loading as full model first
            model = torch.load(model_path, map_location=self.device, weights_only=False)
            model.eval()
            return model.to(self.device)
        except:
            # Create model architecture and load state dict
            model = models.efficientnet_b0(weights=None)
            model.classifier[1] = nn.Sequential(
                nn.Linear(model.classifier[1].in_features, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 2)
            )
            model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=False))
            model.eval()
            return model.to(self.device)
    
    def _get_transforms(self):
        """Get image preprocessing transforms"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def predict_frame(self, frame):
        """
        Predict if a single frame is fake or real
        
        Args:
            frame (numpy.ndarray): Video frame as BGR image
            
        Returns:
            dict: Prediction results
        """
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Apply transforms
        image_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        return {
            'prediction': self.class_names[predicted_class],
            'confidence': confidence,
            'fake_prob': probabilities[0][0].item(),
            'real_prob': probabilities[0][1].item()
        }
    
    def analyze_video(self, video_path, frame_skip=30, max_frames=None, face_detection=True):
        """
        Analyze a video for deepfake content
        
        Args:
            video_path (str): Path to the video file
            frame_skip (int): Analyze every Nth frame (default: 30 for ~1 FPS)
            max_frames (int): Maximum number of frames to analyze
            face_detection (bool): Only analyze frames with detected faces
            
        Returns:
            dict: Comprehensive analysis results
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Initialize video capture
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"🎬 Analyzing video: {os.path.basename(video_path)}")
        print(f"📊 Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s duration")
        
        # Initialize face detector if requested
        if face_detection:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Analysis results
        results = {
            'video_info': {
                'path': video_path,
                'total_frames': total_frames,
                'fps': fps,
                'duration': duration,
                'analyzed_at': datetime.now().isoformat()
            },
            'frame_results': [],
            'statistics': {},
            'temporal_analysis': {}
        }
        
        frame_count = 0
        analyzed_frames = 0
        frames_with_faces = 0
        
        # Progress bar
        frames_to_analyze = min(total_frames // frame_skip, max_frames or float('inf'))
        pbar = tqdm(total=int(frames_to_analyze), desc="Analyzing frames")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Skip frames based on frame_skip parameter
                if frame_count % frame_skip != 0:
                    frame_count += 1
                    continue
                
                # Check max_frames limit
                if max_frames and analyzed_frames >= max_frames:
                    break
                
                # Face detection filter
                should_analyze = True
                faces_detected = 0
                
                if face_detection:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    faces_detected = len(faces)
                    
                    if faces_detected == 0:
                        should_analyze = False
                    else:
                        frames_with_faces += 1
                
                if should_analyze:
                    # Predict frame
                    frame_result = self.predict_frame(frame)
                    
                    # Add frame metadata
                    frame_result.update({
                        'frame_number': frame_count,
                        'timestamp': frame_count / fps if fps > 0 else 0,
                        'faces_detected': faces_detected
                    })
                    
                    results['frame_results'].append(frame_result)
                    analyzed_frames += 1
                    pbar.update(1)
                
                frame_count += 1
                
        finally:
            cap.release()
            pbar.close()
        
        # Calculate statistics
        if results['frame_results']:
            fake_predictions = [r for r in results['frame_results'] if r['prediction'] == 'Fake']
            real_predictions = [r for r in results['frame_results'] if r['prediction'] == 'Real']
            
            # Basic statistics
            results['statistics'] = {
                'total_analyzed_frames': analyzed_frames,
                'frames_with_faces': frames_with_faces,
                'fake_frames': len(fake_predictions),
                'real_frames': len(real_predictions),
                'fake_percentage': len(fake_predictions) / analyzed_frames * 100,
                'real_percentage': len(real_predictions) / analyzed_frames * 100,
                'avg_fake_confidence': np.mean([r['fake_prob'] for r in results['frame_results']]),
                'avg_real_confidence': np.mean([r['real_prob'] for r in results['frame_results']]),
                'max_fake_confidence': max([r['fake_prob'] for r in results['frame_results']]),
                'max_real_confidence': max([r['real_prob'] for r in results['frame_results']])
            }
            
            # Temporal analysis
            self._perform_temporal_analysis(results)
            
            # Overall verdict
            results['overall_verdict'] = self._determine_overall_verdict(results['statistics'])
        
        return results
    
    def _perform_temporal_analysis(self, results):
        """Perform temporal analysis to detect patterns over time"""
        frame_results = results['frame_results']
        
        if len(frame_results) < 3:
            return
        
        # Analyze temporal consistency
        fake_probs = [r['fake_prob'] for r in frame_results]
        timestamps = [r['timestamp'] for r in frame_results]
        
        # Calculate moving averages
        window_size = min(5, len(fake_probs))
        moving_avg = []
        for i in range(len(fake_probs) - window_size + 1):
            avg = np.mean(fake_probs[i:i + window_size])
            moving_avg.append(avg)
        
        # Detect suspicious patterns
        variance = np.var(fake_probs)
        std_dev = np.std(fake_probs)
        
        # Count rapid changes (potential artifacts)
        rapid_changes = 0
        threshold = 0.3  # 30% change
        for i in range(1, len(fake_probs)):
            if abs(fake_probs[i] - fake_probs[i-1]) > threshold:
                rapid_changes += 1
        
        results['temporal_analysis'] = {
            'variance': variance,
            'std_deviation': std_dev,
            'rapid_changes': rapid_changes,
            'consistency_score': 1.0 - (variance * 2),  # Higher is more consistent
            'moving_averages': moving_avg,
            'temporal_pattern': 'inconsistent' if rapid_changes > len(fake_probs) * 0.1 else 'consistent'
        }
    
    def _determine_overall_verdict(self, stats):
        """Determine overall video verdict based on statistics"""
        fake_percentage = stats['fake_percentage']
        avg_fake_confidence = stats['avg_fake_confidence']
        
        if fake_percentage > 70:
            confidence_level = "Very High"
            verdict = "Deepfake"
        elif fake_percentage > 50:
            confidence_level = "High"
            verdict = "Likely Deepfake"
        elif fake_percentage > 30:
            confidence_level = "Medium"
            verdict = "Possibly Deepfake"
        else:
            confidence_level = "Low"
            verdict = "Likely Real"
        
        return {
            'verdict': verdict,
            'confidence_level': confidence_level,
            'fake_percentage': fake_percentage,
            'avg_fake_confidence': avg_fake_confidence
        }
    
    def save_results(self, results, output_path):
        """Save analysis results to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"📄 Results saved to: {output_path}")
    
    def create_analysis_plots(self, results, save_path=None):
        """Create visualization plots for the analysis"""
        if not results['frame_results']:
            print("No frame results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Video Deepfake Analysis: {os.path.basename(results['video_info']['path'])}", 
                     fontsize=16, fontweight='bold')
        
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"📊 Plots saved to: {save_path}")
        
        plt.show()
    
    def print_summary(self, results):
        """Print a formatted summary of the analysis"""
        print("\n" + "="*60)
        print("🎬 VIDEO DEEPFAKE ANALYSIS SUMMARY")
        print("="*60)
        
        # Video info
        video_info = results['video_info']
        print(f"📁 File: {os.path.basename(video_info['path'])}")
        print(f"⏱️  Duration: {video_info['duration']:.2f} seconds")
        print(f"🎞️  Total Frames: {video_info['total_frames']}")
        print(f"📊 FPS: {video_info['fps']:.2f}")
        
        if 'statistics' in results:
            stats = results['statistics']
            print(f"\n📈 ANALYSIS STATISTICS:")
            print(f"   Analyzed Frames: {stats['total_analyzed_frames']}")
            print(f"   Frames with Faces: {stats['frames_with_faces']}")
            print(f"   Fake Frames: {stats['fake_frames']} ({stats['fake_percentage']:.1f}%)")
            print(f"   Real Frames: {stats['real_frames']} ({stats['real_percentage']:.1f}%)")
            print(f"   Average Fake Confidence: {stats['avg_fake_confidence']:.3f}")
            print(f"   Maximum Fake Confidence: {stats['max_fake_confidence']:.3f}")
        
        if 'overall_verdict' in results:
            verdict = results['overall_verdict']
            print(f"\n🎯 OVERALL VERDICT:")
            print(f"   Classification: {verdict['verdict']}")
            print(f"   Confidence Level: {verdict['confidence_level']}")
            print(f"   Fake Percentage: {verdict['fake_percentage']:.1f}%")
        
        if 'temporal_analysis' in results:
            temporal = results['temporal_analysis']
            print(f"\n⏰ TEMPORAL ANALYSIS:")
            print(f"   Consistency Score: {temporal['consistency_score']:.3f}")
            print(f"   Pattern: {temporal['temporal_pattern']}")
            print(f"   Rapid Changes: {temporal['rapid_changes']}")
        
        print("="*60)

def main():
    """Example usage of the video deepfake detector"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Video Deepfake Detection')
    parser.add_argument('video_path', help='Path to the video file')
    parser.add_argument('--model', default='best_model.pth', help='Path to the model file')
    parser.add_argument('--skip', type=int, default=30, help='Frame skip interval')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to analyze')
    parser.add_argument('--no-face-detection', action='store_true', help='Disable face detection')
    parser.add_argument('--output', help='Output JSON file path')
    parser.add_argument('--plot', help='Save plots to this path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = VideoDeepfakeDetector(model_path=args.model)
    
    # Analyze video
    results = detector.analyze_video(
        video_path=args.video_path,
        frame_skip=args.skip,
        max_frames=args.max_frames,
        face_detection=not args.no_face_detection
    )
    
    # Print summary
    detector.print_summary(results)
    
    # Save results
    if args.output:
        detector.save_results(results, args.output)
    
    # Create plots
    detector.create_analysis_plots(results, save_path=args.plot)

if __name__ == "__main__":
    main()
