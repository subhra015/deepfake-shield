import requests
from pathlib import Path

url = "http://127.0.0.1:8000/predict/image"
image_path = Path("DF2.jpg")

if not image_path.exists():
    print("❌ No image found")
    exit(1)

print(f"📤 Uploading: {image_path}")

with open(image_path, "rb") as f:
    response = requests.post(url, files={"file": f})

print(f"Status: {response.status_code}")

if response.status_code == 200:
    result = response.json()
    print("\n" + "=" * 50)
    print("🛡️  DeepFake Shield API Result")
    print("=" * 50)
    print(f"Prediction:    {result['prediction'].upper()}")
    print(f"Confidence:    {result['confidence']*100:.1f}%")
    print(f"Fake prob:     {result['probabilities']['fake']*100:.1f}%")
    print(f"Real prob:     {result['probabilities']['real']*100:.1f}%")
    if 'inference_time_ms' in result:
        print(f"Inference:     {result['inference_time_ms']}ms")
    print("=" * 50)
else:
    print(f"❌ Error {response.status_code}: {response.text[:500]}")
