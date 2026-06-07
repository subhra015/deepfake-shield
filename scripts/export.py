"""ONNX export CLI entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from src.models.classifier import DeepfakeClassifier
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.export")


def main() -> None:
    """Export PyTorch checkpoint to ONNX and verify numerically."""
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--onnx_out", default="models/model.onnx", help="Output ONNX path")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("config", {})
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    image_size = int(data_cfg.get("image_size", 224))

    model = DeepfakeClassifier(
        backbone_name=model_cfg.get("backbone", "convnext_tiny"),
        num_classes=int(model_cfg.get("num_classes", 2)),
        dropout=float(model_cfg.get("dropout", 0.3)),
        use_attention=bool(model_cfg.get("use_attention", True)),
        pretrained=False,
    )
    model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    model.eval()

    onnx_path = Path(args.onnx_out)
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    dummy = torch.randn(1, 3, image_size, image_size)
    torch.onnx.export(
        model,
        dummy,
        str(onnx_path),
        opset_version=args.opset,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
    )

    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        pt_out = model(dummy).numpy()
    ort_out = session.run(None, {"input": dummy.numpy()})[0]

    max_diff = float(np.max(np.abs(pt_out - ort_out)))
    if max_diff > 1e-5:
        raise RuntimeError(f"ONNX mismatch. max_diff={max_diff}")

    pt_tmp = onnx_path.with_suffix(".pth.tmp")
    torch.save(model.state_dict(), pt_tmp)
    pt_size_mb = pt_tmp.stat().st_size / (1024 * 1024)
    onnx_size_mb = onnx_path.stat().st_size / (1024 * 1024)
    pt_tmp.unlink(missing_ok=True)

    logger.info("Exported ONNX: %s", onnx_path.resolve())
    logger.info("PyTorch state_dict size: %.2f MB", pt_size_mb)
    logger.info("ONNX model size: %.2f MB", onnx_size_mb)


if __name__ == "__main__":
    main()
