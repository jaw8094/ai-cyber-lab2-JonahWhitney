"""Evaluate a trained SVM phishing URL classifier on the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from joblib import load
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

try:
    from .data import load_and_split_feature_matrix
except ImportError:
    from data import load_and_split_feature_matrix


def evaluate_svm_model(
    model_path: str | Path,
    dataset_path: str | Path,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    deduplicate: bool = True,
) -> tuple[dict[str, float], list[int], list[int]]:
    """Load trained model and evaluate it on the test split."""

    _, x_test, _, y_test = load_and_split_feature_matrix(
        file_path=dataset_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
        deduplicate=deduplicate,
    )

    model = load(model_path)
    y_pred = model.predict(x_test)

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
    }
    return metrics, y_test, y_pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate trained SVM model on phishing URL test split"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "svm_model.joblib",
        help="Path to trained model artifact.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "phishing_site_urls.csv",
        help="Path to dataset CSV/TSV/JSONL.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "metrics.json",
        help="Path to save evaluation metrics JSON.",
    )
    parser.add_argument(
        "--confusion-matrix-out",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "results"
        / "confusion_matrix.png",
        help="Path to save confusion matrix image.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication in data loading.",
    )
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model artifact not found: {args.model}")

    metrics, y_test, y_pred = evaluate_svm_model(
        model_path=args.model,
        dataset_path=args.dataset,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        deduplicate=not args.no_deduplicate,
    )

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.confusion_matrix_out.parent.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=["Benign (0)", "Phishing (1)"]
    )
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(args.confusion_matrix_out, dpi=150)
    plt.close(fig)

    print("Evaluation complete.")
    print(f"Model loaded from: {args.model}")
    print(f"Metrics saved to: {args.metrics_out}")
    print(f"Confusion matrix saved to: {args.confusion_matrix_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
