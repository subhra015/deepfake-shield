import os
import sys
import tempfile
import traceback
from pathlib import Path
from io import BytesIO
from typing import List, Dict, Any

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

# ─── Config ───────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "deepfake_best.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
MAX_FILE_SIZE_MB = 50
MAX_VIDEO_FRAMES = 32

# ─── Model ──────────────────────────────────────────────────────────────────────
class DeepFakeDetector(nn.Module):
    def __init__(self, num_classes=2, dropout=0.3):
        super().__init__()
        self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = self.backbone.classifier[-1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Flatten(1),
            nn.Dropout(dropout),
            nn.Linear(in_features, num_classes)
        )
    def forward(self, x):
        return self.backbone(x)

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(title="DeepFake Shield API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Transforms ─────────────────────────────────────────────────────────────────
transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ─── Model Load ─────────────────────────────────────────────────────────────────
model = None

def load_model():
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    model = DeepFakeDetector()
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    model.eval()

@app.on_event("startup")
async def startup_event():
    load_model()

# ─── Helpers ──────────────────────────────────────────────────────────────────
def safe_image(file_bytes: bytes) -> Image.Image:
    try:
        img = Image.open(BytesIO(file_bytes))
        img = ImageOps.exif_transpose(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception:
        raise HTTPException(400, "Invalid or corrupted image file")

def infer_batch(tensors: List[torch.Tensor]) -> np.ndarray:
    batch = torch.stack(tensors).to(DEVICE)
    with torch.no_grad():
        outputs = model(batch)
        probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    return probs

# ─── Endpoints ──────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    
    image = safe_image(contents)
    tensor = transform(image)
    prob = float(infer_batch([tensor])[0])
    is_fake = prob > 0.5
    
    return {
        "prediction": "deepfake" if is_fake else "real",
        "confidence": round(prob if is_fake else 1 - prob, 4),
        "deepfake_probability": round(prob, 4)
    }

@app.post("/predict/video")
async def predict_video(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds {MAX_FILE_SIZE_MB}MB limit")
    
    suffix = Path(file.filename).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(contents)
        temp_path = tmp.name
    
    try:
        cap = cv2.VideoCapture(temp_path)
        if not cap.isOpened():
            raise HTTPException(400, "Cannot decode video. Re-encode with H.264/mp4v.")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if total_frames == 0:
            raise HTTPException(400, "Video contains no frames")
        
        num_samples = min(MAX_VIDEO_FRAMES, total_frames)
        indices = np.linspace(0, total_frames - 1, num_samples, dtype=int)
        
        tensors: List[torch.Tensor] = []
        valid = 0
        skipped = 0
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret or frame is None or frame.size == 0:
                skipped += 1
                continue
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb).convert("RGB")
                tensors.append(transform(pil))
                valid += 1
            except Exception:
                skipped += 1
                continue
        
        cap.release()
        
        if valid == 0:
            raise HTTPException(400, "No valid frames could be extracted")
        
        probs = infer_batch(tensors)
        avg_prob = float(np.mean(probs))
        max_prob = float(np.max(probs))
        is_fake = avg_prob > 0.5
        
        return {
            "prediction": "deepfake" if is_fake else "real",
            "confidence": round(max_prob if is_fake else 1 - avg_prob, 4),
            "deepfake_probability": round(avg_prob, 4),
            "frames_processed": valid,
            "frames_skipped": skipped,
            "total_frames": total_frames,
            "fps": round(fps, 2) if fps else None
        }
        
    except HTTPException:
        raise
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, "Video processing failed")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
