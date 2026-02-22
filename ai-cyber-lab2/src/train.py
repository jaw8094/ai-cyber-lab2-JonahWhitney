"""Train an SVM phishing URL classifier using engineered URL features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from joblib import dump
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

try:
    from .data import load_and_split_feature_matrix
except ImportError:
    from data import load_and_split_feature_matrix


def train_svm_model(
    dataset_path: str | Path,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    deduplicate: bool = True,
    max_train_samples: int | None = None,
) -> tuple[object, dict[str, float]]:
    """Load features/labels, train SVM on train split, and evaluate on test split."""

    x_train, x_test, y_train, y_test = load_and_split_feature_matrix(
        file_path=dataset_path,
        train_ratio=train_ratio,
        random_seed=random_seed,
        deduplicate=deduplicate,
    )

    if max_train_samples is not None:
        if max_train_samples <= 0:
            raise ValueError("max_train_samples must be positive when provided.")
        x_train = x_train[:max_train_samples]
        y_train = y_train[:max_train_samples]

    model = make_pipeline(
        StandardScaler(),
        LinearSVC(random_state=random_seed, class_weight="balanced", max_iter=5000),
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "train_size": float(len(x_train)),
        "test_size": float(len(x_test)),
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SVM model for phishing URL data")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(__file__).resolve().parent / "phishing_site_urls.csv",
        help="Path to dataset CSV/TSV/JSONL.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "svm_model.joblib",
        help="Path to save trained model artifact.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "results" / "metrics.json",
        help="Path to save metrics JSON.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable deduplication in data loading.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on train samples for faster iteration.",
    )
    args = parser.parse_args()

    model, metrics = train_svm_model(
        dataset_path=args.dataset,
        train_ratio=args.train_ratio,
        random_seed=args.seed,
        deduplicate=not args.no_deduplicate,
        max_train_samples=args.max_train_samples,
    )

    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)

    dump(model, args.model_out)
    args.metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("Training complete.")
    print(f"Model saved to: {args.model_out}")
    print(f"Metrics saved to: {args.metrics_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
