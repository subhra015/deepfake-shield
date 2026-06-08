# DeepFake Shield

Real-time deepfake detection API using ConvNeXt-Tiny.
95.05% accuracy | AUC 0.9502 | Hardened against JPEG, noise, blur, and low-res inputs.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| /health | GET | Server and model status |
| /predict/image | POST | Single image detection |
| /predict/video | POST | Video frame-sampling detection |

## Setup

pip install -r requirements.txt
python -m uvicorn src.api.main:app --reload

## Model Weights

Place deepfake_best.pth (318 MB) in models/ before starting.
