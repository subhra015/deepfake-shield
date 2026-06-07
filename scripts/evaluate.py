"""Evaluation CLI entry point."""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.dataset import DeepfakeDataset, collect_images_from_dir
from src.data.datamodule import build_eval_transform
from src.evaluation.explainability import GradCAM
from src.evaluation.metrics import (
    compute_auc,
    compute_eer,
    compute_f1_at_threshold,
    plot_confusion_matrix,
    plot_roc_curve,
)
from src.evaluation.robustness import run_robustness_suite
from src.inference.predictor import DeepfakePredictor
from src.utils.logging import setup_logging

logger = setup_logging(name="deepfake.evaluate")


def main() -> None:
    """Run evaluation on test set."""
    parser = argparse.ArgumentParser(description="Evaluate Deepfake Shield model")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--test_dir", required=True, help="Test directory with real/ and fake/")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path("results") / f"evaluation_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    predictor = DeepfakePredictor(args.checkpoint, image_size=args.image_size)
    transform = build_eval_transform(args.image_size)

    paths, labels = collect_images_from_dir(args.test_dir)
    dataset = DeepfakeDataset(paths, labels, transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    y_true: list[int] = []
    y_scores: list[float] = []
    y_pred: list[int] = []

    model = predictor.get_model()
    device = predictor.device
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels_t = batch["label"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = (probs >= 0.5).long()

            y_true.extend(labels_t.cpu().tolist())
            y_scores.extend(probs.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())

    y_true_np = np.array(y_true)
    y_scores_np = np.array(y_scores)
    y_pred_np = np.array(y_pred)

    auc_value = compute_auc(y_true_np, y_scores_np)
    eer_value = compute_eer(y_true_np, y_scores_np)
    f1_value = compute_f1_at_threshold(y_true_np, y_scores_np, threshold=0.5)

    plot_roc_curve(y_true_np, y_scores_np, out_dir / "roc_curve.png")
    plot_confusion_matrix(y_true_np, y_pred_np, out_dir / "confusion_matrix.png")

    cam = GradCAM(model, model.get_cam_target_layer())
    idxs_correct = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t == p]
    idxs_incorrect = [i for i, (t, p) in enumerate(zip(y_true, y_pred)) if t != p]
    random.shuffle(idxs_correct)
    random.shuffle(idxs_incorrect)
    sample_idxs = idxs_correct[:5] + idxs_incorrect[:5]

    for index, sample_idx in enumerate(sample_idxs):
        path = paths[sample_idx]
        label = y_true[sample_idx]
        image = predictor._load_image(path)
        tensor = predictor._preprocess(image)
        cam.save_visualization(
            tensor, image, out_dir / f"gradcam_{index}_true{label}.png", target_class=1
        )

    robustness = run_robustness_suite(model, dataloader, device)

    metrics = {
        "auc": auc_value,
        "eer": eer_value,
        "f1_at_0.5": f1_value,
        "robustness": robustness,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / "metrics.txt").write_text(
        f"AUC: {auc_value:.6f}\nEER: {eer_value:.6f}\nF1@0.5: {f1_value:.6f}\n",
        encoding="utf-8",
    )

    logger.info("Saved evaluation results to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
