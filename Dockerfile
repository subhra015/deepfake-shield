FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

# Auto-download model from GitHub Releases on startup
RUN mkdir -p models
ENV MODEL_URL="https://github.com/subhra015/deepfake-shield/releases/download/v1.0.0/deepfake_best.pth"
RUN wget -O models/deepfake_best.pth $MODEL_URL

EXPOSE 7860
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "7860"]
