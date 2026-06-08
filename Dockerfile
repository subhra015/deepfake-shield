FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxext6 libxrender-dev libgomp1 wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY scripts/ ./scripts/

RUN mkdir -p models

EXPOSE 7860

# Download model at runtime, then start server
CMD wget -q -O models/deepfake_best.pth https://github.com/subhra015/deepfake-shield/releases/download/v1.0.0/deepfake_best.pth && \
    uvicorn src.api.main:app --host 0.0.0.0 --port 7860
