import streamlit as st
import torch
import timm
import numpy as np
import cv2
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.evaluation.gradcam import GradCAM

st.set_page_config(page_title="DeepFake Shield | AI Detection", page_icon="🛡️", layout="wide")

# Dark professional theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    * { font-family: 'Inter', sans-serif; }
    
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    .header-card {
        background: rgba(30, 41, 59, 0.8);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(148, 163, 184, 0.1);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .upload-zone {
        background: rgba(15, 23, 42, 0.6);
        border: 2px dashed rgba(148, 163, 184, 0.3);
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        transition: all 0.3s;
    }
    
    .result-card {
        background: rgba(30, 41, 59, 0.9);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(148, 163, 184, 0.1);
    }
    
    .fake-glow { 
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(239, 68, 68, 0.05));
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    .real-glow {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(34, 197, 94, 0.05));
        border: 1px solid rgba(34, 197, 94, 0.3);
    }
    
    .metric-pill {
        background: rgba(15, 23, 42, 0.8);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        font-size: 0.875rem;
    }
    
    .confidence-ring {
        width: 120px;
        height: 120px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0 auto;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
    }
    
    .footer {
        text-align: center;
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 3rem;
        padding-top: 1rem;
        border-top: 1px solid rgba(148, 163, 184, 0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model = timm.create_model("convnext_tiny.fb_in22k", pretrained=False, num_classes=2)
    model_path = Path(__file__).parent.parent / "models" / "deepfake_best.pth"
    ckpt = torch.load(str(model_path), map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model.eval().to(device)
    target_layer = model.stages[-1].blocks[-1].conv_dw
    gradcam = GradCAM(model, target_layer)
    return model, gradcam, device

def preprocess(image):
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    img = np.array(image.convert("RGB"))
    tensor = transform(image=img)["image"].unsqueeze(0)
    return tensor, img

# Header
st.markdown("""
<div class="header-card">
    <div style="display: flex; align-items: center; justify-content: space-between;">
        <div>
            <h1 style="margin: 0; font-size: 2rem; font-weight: 700; color: #f8fafc;">
                🛡️ DeepFake Shield
            </h1>
            <p style="margin: 0.5rem 0 0 0; color: #94a3b8; font-size: 0.95rem;">
                Enterprise-grade deepfake detection with neural explainability
            </p>
        </div>
        <div style="text-align: right;">
            <div class="metric-pill">⚡ ConvNeXt-Tiny</div>
            <div class="metric-pill" style="margin-top: 0.5rem;">🎯 95.05% Accuracy</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Main layout
left, right = st.columns([2, 1])

with left:
    uploaded = st.file_uploader(
        "📤 Drop image or click to browse",
        type=["jpg", "jpeg", "png"],
        help="Supported: JPG, PNG up to 10MB"
    )
    
    if uploaded:
        image = Image.open(uploaded)
        tensor, img_array = preprocess(image)
        model, gradcam, device = load_model()
        tensor = tensor.to(device)
        
        with st.spinner("🔍 Neural analysis in progress..."):
            start_time = time.time()
            with torch.no_grad():
                logits = model(tensor)
                probs = torch.softmax(logits, dim=1)[0]
            fake_prob = probs[0].item()
            real_prob = probs[1].item()
            pred_class = 0 if fake_prob > real_prob else 1
            overlay, _ = gradcam.generate(tensor, img_array, target_class=pred_class)
            inference_time = (time.time() - start_time) * 1000
        
        # Results grid
        r1, r2 = st.columns(2)
        with r1:
            st.markdown("### 📷 Source")
            st.image(image, use_container_width=True)
            st.caption(f"{image.size[0]}×{image.size[1]} • {uploaded.size/1024:.0f}KB")
        
        with r2:
            st.markdown("### 🔥 Attention Map")
            st.image(overlay, use_container_width=True)
            st.caption("Red = High suspicion • Blue = Cleared")
        
        # Metrics bar
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric("Fake Probability", f"{fake_prob*100:.1f}%")
        with m2:
            st.metric("Real Probability", f"{real_prob*100:.1f}%")
        with m3:
            st.metric("Inference", f"{inference_time:.0f}ms")
        with m4:
            st.metric("Model Confidence", f"{max(fake_prob, real_prob)*100:.1f}%")

with right:
    if uploaded:
        confidence = max(fake_prob, real_prob) * 100
        is_fake = fake_prob > real_prob
        
        result_class = "fake-glow" if is_fake else "real-glow"
        result_icon = "🔴" if is_fake else "🟢"
        result_text = "SYNTHETIC DETECTED" if is_fake else "AUTHENTIC"
        result_color = "#ef4444" if is_fake else "#22c55e"
        
        st.markdown(f"""
        <div class="result-card {result_class}">
            <div style="text-align: center;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{result_icon}</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: {result_color}; letter-spacing: 0.05em;">
                    {result_text}
                </div>
                <div style="font-size: 2.5rem; font-weight: 700; color: #f8fafc; margin: 1rem 0;">
                    {confidence:.1f}%
                </div>
                <div style="color: #94a3b8; font-size: 0.875rem;">
                    confidence score
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Probability visualization
        st.markdown("#### Distribution")
        st.progress(fake_prob if is_fake else real_prob)
        
        # Risk assessment
        risk_level = "CRITICAL" if confidence > 90 else "HIGH" if confidence > 75 else "MODERATE" if confidence > 60 else "LOW"
        risk_color = "#ef4444" if risk_level in ["CRITICAL", "HIGH"] else "#f59e0b" if risk_level == "MODERATE" else "#22c55e"
        
        st.markdown(f"""
        <div style="margin-top: 1.5rem;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                <span style="color: #94a3b8; font-size: 0.875rem;">Risk Assessment</span>
                <span style="color: {risk_color}; font-weight: 600; font-size: 0.875rem;">{risk_level}</span>
            </div>
            <div style="background: rgba(15, 23, 42, 0.8); border-radius: 6px; height: 6px; overflow: hidden;">
                <div style="width: {confidence}%; height: 100%; background: {risk_color}; border-radius: 6px; transition: width 0.5s;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Explanation
        st.markdown("""
        <div style="margin-top: 2rem; padding-top: 1rem; border-top: 1px solid rgba(148, 163, 184, 0.1);">
            <div style="color: #94a3b8; font-size: 0.8rem; line-height: 1.6;">
                <b style="color: #e2e8f0;">Analysis Method</b><br>
                GradCAM highlights regions that most influenced the model's decision. 
                Focus on facial boundaries, eyes, and artifacts for synthetic detection.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-card" style="text-align: center; padding: 3rem 1rem;">
            <div style="font-size: 3rem; margin-bottom: 1rem;">📤</div>
            <div style="color: #94a3b8; font-size: 0.95rem;">
                Upload an image to begin<br>neural analysis
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # How it works
        st.markdown("### How it works")
        steps = [
            ("1", "Upload", "Any face image"),
            ("2", "Process", "ConvNeXt analyzes"),
            ("3", "Explain", "GradCAM visualizes"),
        ]
        for num, title, desc in steps:
            st.markdown(f"""
            <div style="display: flex; align-items: center; gap: 0.75rem; margin: 0.75rem 0;">
                <div style="width: 28px; height: 28px; background: rgba(59, 130, 246, 0.2); border-radius: 50%; display: flex; align-items: center; justify-content: center; color: #60a5fa; font-weight: 600; font-size: 0.75rem;">{num}</div>
                <div>
                    <div style="color: #e2e8f0; font-weight: 600; font-size: 0.875rem;">{title}</div>
                    <div style="color: #64748b; font-size: 0.75rem;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    DeepFake Shield v1.0 • Built with PyTorch + ConvNeXt • Trained on 60K+ frames
</div>
""", unsafe_allow_html=True)