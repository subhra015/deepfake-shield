import cv2
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Tuple
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from collections import Counter

class VideoDeepfakeDetector:
    def __init__(self, model, device, sample_rate=1):
        """
        model: loaded PyTorch model
        device: torch device
        sample_rate: extract 1 frame every N seconds
        """
        self.model = model
        self.device = device
        self.sample_rate = sample_rate
        
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    
    def extract_frames(self, video_path: str, max_frames=30) -> List[np.ndarray]:
        """Extract frames from video."""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        frame_indices = []
        
        count = 0
        while cap.isOpened() and len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sample every N seconds
            if count % int(fps * self.sample_rate) == 0:
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                frame_indices.append(count)
            
            count += 1
        
        cap.release()
        return frames, frame_indices, fps, count
    
    def predict_frame(self, frame: np.ndarray) -> Tuple[float, float]:
        """Predict single frame."""
        tensor = self.transform(image=frame)["image"].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
        
        return float(probs[0]), float(probs[1])  # fake_prob, real_prob
    
    def analyze_video(self, video_path: str) -> Dict:
        """Full video analysis with temporal voting."""
        frames, indices, fps, total_frames = self.extract_frames(video_path)
        
        if not frames:
            return {"error": "No frames extracted"}
        
        # Predict all frames
        predictions = []
        for i, frame in enumerate(frames):
            fake_prob, real_prob = self.predict_frame(frame)
            predictions.append({
                "frame_idx": indices[i],
                "fake_prob": fake_prob,
                "real_prob": real_prob,
                "is_fake": fake_prob > real_prob
            })
        
        # Temporal analysis
        fake_votes = sum(1 for p in predictions if p["is_fake"])
        real_votes = len(predictions) - fake_votes
        
        # Consistency check (low variance = real, high variance = suspicious)
        fake_probs = [p["fake_prob"] for p in predictions]
        consistency = 1.0 - np.std(fake_probs)  # higher = more consistent
        
        # Temporal voting: weight by consistency
        if consistency > 0.8:
            # High consistency → trust majority vote
            final_fake_prob = np.mean(fake_probs)
        else:
            # Low consistency → suspicious, boost fake probability
            final_fake_prob = np.mean(fake_probs) + 0.1
        
        final_fake_prob = min(final_fake_prob, 0.99)
        
        return {
            "total_frames": total_frames,
            "analyzed_frames": len(frames),
            "fps": fps,
            "fake_votes": fake_votes,
            "real_votes": real_votes,
            "consistency_score": round(consistency, 3),
            "final_fake_probability": round(final_fake_prob, 3),
            "final_prediction": "fake" if final_fake_prob > 0.5 else "real",
            "confidence": round(max(final_fake_prob, 1 - final_fake_prob), 3),
            "frame_predictions": predictions,
            "temporal_analysis": {
                "stable": consistency > 0.8,
                "suspicious_flicker": consistency < 0.6,
                "recommendation": "Deepfake detected" if final_fake_prob > 0.7 else "Likely real" if consistency > 0.8 else "Uncertain - manual review"
            }
        }
