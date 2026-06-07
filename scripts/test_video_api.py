import requests
import sys
from pathlib import Path

URL = "http://127.0.0.1:8000/predict/video"

def test_video(video_path: str):
    path = Path(video_path)
    if not path.exists():
        print(f"❌ File not found: {video_path}")
        sys.exit(1)
    print(f"📤 Uploading: {path.name} ({path.stat().st_size / 1024 / 1024:.2f} MB)")
    with open(path, "rb") as f:
        files = {"file": (path.name, f, "video/mp4")}
        try:
            response = requests.post(URL, files=files, timeout=60)
        except requests.exceptions.ConnectionError:
            print("❌ Cannot connect to API. Is the server running on http://127.0.0.1:8000?")
            sys.exit(1)
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("✅ SUCCESS:")
        print(f"   Prediction: {data['prediction']}")
        print(f"   Confidence: {data['confidence']}")
        print(f"   Frames processed: {data['frames_processed']} / {data['total_frames']}")
        print(f"   DeepFake Probability: {data['deepfake_probability']}")
    else:
        print("❌ ERROR:")
        try:
            print(response.json())
        except:
            print(response.text)

if __name__ == "__main__":
    video = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"
    test_video(video)
