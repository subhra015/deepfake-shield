"""Pydantic request/response models for the inference API."""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class ImageUploadRequest(BaseModel):
    """Single image via base64 string or reference to uploaded file."""

    image_base64: Optional[str] = Field(None, description="Base64-encoded image bytes")
    filename: Optional[str] = Field(None, description="Original filename for logging")


class BatchImageUploadRequest(BaseModel):
    """Batch base64 image upload schema."""

    images_base64: List[str] = Field(..., min_length=1, max_length=32)


class ProbabilityDict(BaseModel):
    """Class probability distribution."""

    real: float = Field(..., ge=0.0, le=1.0)
    fake: float = Field(..., ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Single image prediction response."""

    label: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: ProbabilityDict
    gradcam_base64: Optional[str] = None
    request_id: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""

    predictions: List[PredictionResponse]
    request_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_name: str
    device: str
    version: str = "1.0.0"
    model_loaded: bool = True


class MetricsResponse(BaseModel):
    """Prometheus-style metrics summary."""

    inference_count: int
    total_latency_ms: float
    avg_latency_ms: float


class ErrorResponse(BaseModel):
    """Standard error response."""

    detail: str
    request_id: Optional[str] = None
