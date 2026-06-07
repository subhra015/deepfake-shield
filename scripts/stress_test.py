import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
import io
import cv2

URL_IMAGE = "http://127.0.0.1:8000/predict/image"
URL_VIDEO = "http://127.0.0.1:8000/predict/video"
TEST_IMG = "test_image.jpg"
TEST_VID = "test_video.mp4"

def make_degraded_images(source_path: str):
    img = Image.open(source_path).convert("RGB")
    variants = {"original": img}
    
    # JPEG compression artifacts
    for quality in [80, 50, 20, 10]:
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        variants[f"jpeg_q{quality}"] = Image.open(buf)
    
    # Gaussian noise
    arr = np.array(img).astype(np.float32)
    for sigma in [5, 15, 25, 40]:
        noisy = arr + np.random.normal(0, sigma, arr.shape)
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        variants[f"noise_s{sigma}"] = Image.fromarray(noisy)
    
    # Blur
    for radius in [1, 3, 5]:
        variants[f"blur_r{radius}"] = img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    # Brightness / Contrast shifts
    for factor in [0.5, 1.5, 2.0]:
        variants[f"brightness_{factor}"] = ImageEnhance.Brightness(img).enhance(factor)
        variants[f"contrast_{factor}"] = ImageEnhance.Contrast(img).enhance(factor)
    
    # Low resolution (downscale-upscale)
    small = img.resize((56, 56), Image.BICUBIC)
    variants["lowres"] = small.resize((224, 224), Image.BICUBIC)
    
    return variants

def test_image_robustness():
    if not Path(TEST_IMG).exists():
        # Create a simple test image
        img = Image.new("RGB", (640, 480), (180, 160, 140))
        img.save(TEST_IMG)
        print(f"Created {TEST_IMG}")
    
    variants = make_degraded_images(TEST_IMG)
    results = []
    
    print(f"\n{'Variant':<<20} {'Status':<<8} {'Prediction':<<10} {'Confidence':<<12} {'DF Prob':<<10}")
    print("-" * 70)
    
    for name, img in variants.items():
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=95)
        files = {"file": (f"{name}.jpg", buf.getvalue(), "image/jpeg")}
        
        try:
            r = requests.post(URL_IMAGE, files=files, timeout=10)
            if r.status_code == 200:
                d = r.json()
                print(f"{name:<20} {r.status_code:<8} {d['prediction']:<10} {d['confidence']:<12.4f} {d['deepfake_probability']:<10.4f}")
                results.append({"variant": name, "ok": True, **d})
            else:
                print(f"{name:<20} {r.status_code:<8} ERROR")
                results.append({"variant": name, "ok": False})
        except Exception as e:
            print(f"{name:<20} {'FAIL':<<8} {str(e)[:30]}")
            results.append({"variant": name, "ok": False, "error": str(e)})
    
    # Summary
    ok_count = sum(1 for r in results if r["ok"])
    print(f"\n✅ Passed: {ok_count}/{len(results)}")
    
    confidences = [r["confidence"] for r in results if r.get("confidence")]
    if confidences:
        print(f"📊 Confidence range: {min(confidences):.4f} - {max(confidences):.4f}")
        print(f"📊 Confidence std:   {np.std(confidences):.4f}")
        if np.std(confidences) > 0.15:
            print("⚠️  High variance under degradation — model may need more robust training data")

def test_video_endpoint():
    print(f"\n{'='*50}")
    print("VIDEO ENDPOINT TEST")
    print(f"{'='*50}")
    
    if not Path(TEST_VID).exists():
        print(f"❌ {TEST_VID} not found. Run make_test_video.py first.")
        return
    
    with open(TEST_VID, "rb") as f:
        files = {"file": (TEST_VID, f, "video/mp4")}
        r = requests.post(URL_VIDEO, files=files, timeout=30)
    
    print(f"Status: {r.status_code}")
    if r.status_code == 200:
        d = r.json()
        print(f"Prediction: {d['prediction']}")
        print(f"Confidence: {d['confidence']}")
        print(f"Frames: {d['frames_processed']}/{d['total_frames']} (skipped {d.get('frames_skipped', 0)})")
    else:
        print(f"❌ Error: {r.text}")

if __name__ == "__main__":
    test_image_robustness()
    test_video_endpoint()
