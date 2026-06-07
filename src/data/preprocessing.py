"""Face detection and alignment using MTCNN."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN

from src.exceptions import DataError
from src.utils.device import get_device
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.preprocessing")


class FaceExtractor:
    """Detect and crop faces from images using MTCNN.

    Args:
        image_size: Output face crop size.
        margin: Pixel margin around detected bounding box.
        device: Torch device for MTCNN inference.
    """

    def __init__(
        self,
        image_size: int = 224,
        margin: int = 20,
        device: Optional[str] = None,
    ) -> None:
        resolved = get_device(device or "cuda")
        self.device = resolved
        self.image_size = image_size
        self.margin = margin
        self.detector = MTCNN(
            image_size=image_size,
            margin=margin,
            keep_all=False,
            device=str(resolved),
        )

    def detect_and_crop(
        self, image: Union[np.ndarray, str]
    ) -> Optional[torch.Tensor]:
        """Detect the primary face and return a cropped RGB tensor.

        Args:
            image: RGB numpy array or path to image file.

        Returns:
            Cropped face tensor (C, H, W) or None if no face detected.
        """
        rgb = self._load_rgb(image)
        face_tensor = self.detector(rgb)
        if face_tensor is None:
            logger.warning("No face detected in image")
            return None
        return face_tensor

    def align_face(
        self,
        image: Union[np.ndarray, str],
        landmarks: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return an aligned face crop as RGB numpy array.

        Args:
            image: Input RGB image or path.
            landmarks: Optional facial landmarks (unused; MTCNN handles alignment).

        Returns:
            Aligned face RGB array of shape (H, W, 3).
        """
        tensor = self.detect_and_crop(image)
        if tensor is None:
            raise DataError("Cannot align face: no face detected")
        array = tensor.permute(1, 2, 0).cpu().numpy()
        array = (array * 255.0).clip(0, 255).astype(np.uint8)
        return array

    def batch_extract(
        self, images: List[Union[np.ndarray, str]]
    ) -> Tuple[List[torch.Tensor], List[int]]:
        """Extract faces from a batch of images.

        Args:
            images: List of RGB arrays or file paths.

        Returns:
            Tuple of (face_tensors, valid_indices) for successful detections.
        """
        faces: List[torch.Tensor] = []
        valid_indices: List[int] = []
        for idx, image in enumerate(images):
            face = self.detect_and_crop(image)
            if face is not None:
                faces.append(face)
                valid_indices.append(idx)
        return faces, valid_indices

    @staticmethod
    def _load_rgb(image: Union[np.ndarray, str]) -> np.ndarray:
        """Load image as RGB numpy array."""
        if isinstance(image, str):
            bgr = cv2.imread(image)
            if bgr is None:
                raise DataError(f"Could not read image: {image}")
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if image.ndim != 3 or image.shape[2] != 3:
            raise DataError("Expected RGB image with shape (H, W, 3)")
        return image
