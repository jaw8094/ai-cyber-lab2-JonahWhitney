"""Data loading utilities for phishing-vs-benign classification."""

from __future__ import annotations

import csv
import json
import random
import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlsplit


def _normalize_label(value: object) -> int:
    """Convert common phishing/benign labels to binary integers.

    Returns:
        1 for phishing and 0 for benign.
    """

    text = str(value).strip().lower()

    phishing_values = {"1", "phishing", "phish", "malicious", "true", "yes", "bad"}
    benign_values = {
        "0",
        "benign",
        "legitimate",
        "safe",
        "false",
        "no",
        "good",
    }

    if text in phishing_values:
        return 1
    if text in benign_values:
        return 0

    raise ValueError(
        f"Unsupported label value: {value!r}. Use phishing/benign or 1/0 style labels."
    )


def _iter_rows(path: Path) -> Iterable[dict[str, object]]:
    """Yield records from a CSV/TSV/JSONL file."""

    suffix = path.suffix.lower()

    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            yield from reader
        return

    if suffix in {".jsonl", ".ndjson"}:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return

    raise ValueError(
        f"Unsupported file format: {path.suffix!r}. Expected .csv, .tsv, .jsonl, or .ndjson"
    )


def _clean_input_text(value: object) -> str:
    """Apply basic text cleaning for URL/email-like inputs."""

    text = str(value).strip().lower()
    # Normalize odd spacing and remove accidental line breaks/tabs.
    text = re.sub(r"\s+", "", text)
    # Remove common URL prefixes to reduce duplicate surface forms.
    text = re.sub(r"^https?://", "", text)
    text = re.sub(r"^www\.", "", text)
    # Normalize trailing slash variants.
    text = text.rstrip("/")
    return text


def load_phishing_dataset(
    file_path: str | Path,
    text_column: str = "URL",
    label_column: str = "Label",
    deduplicate: bool = True,
) -> tuple[list[str], list[int]]:
    """Load URL/email phishing dataset from disk.

    Args:
        file_path: Path to a CSV/TSV/JSONL file containing records.
        text_column: Column containing URL or email text.
        label_column: Column containing phishing/benign labels.
        deduplicate: If True, drop duplicate cleaned inputs (first occurrence kept).

    Returns:
        A tuple of (`inputs`, `labels`) where labels are binary (1=phishing, 0=benign).
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    inputs: list[str] = []
    labels: list[int] = []
    seen_inputs: set[str] = set()

    for i, row in enumerate(_iter_rows(path), start=1):
        if text_column not in row:
            raise KeyError(f"Missing text column {text_column!r} in row {i}")
        if label_column not in row:
            raise KeyError(f"Missing label column {label_column!r} in row {i}")

        text = _clean_input_text(row[text_column])
        if not text:
            raise ValueError(f"Empty input text in row {i}")

        if deduplicate and text in seen_inputs:
            continue

        inputs.append(text)
        labels.append(_normalize_label(row[label_column]))
        seen_inputs.add(text)

    if not inputs:
        raise ValueError(f"No data rows were found in {path}")

    return inputs, labels


def _extract_url_features(url: str) -> list[float]:
    """Extract a compact numeric feature vector from a URL-like string."""

    try:
        parsed = urlsplit(f"http://{url}")
        host = parsed.netloc
        path = parsed.path
        query = parsed.query
    except ValueError:
        # Fallback for malformed samples that break URL parsing.
        first_slash = url.find("/")
        if first_slash == -1:
            host = url
            path = ""
        else:
            host = url[:first_slash]
            path = url[first_slash:]
        query_split = path.split("?", maxsplit=1)
        path = query_split[0]
        query = query_split[1] if len(query_split) == 2 else ""

    digits = sum(ch.isdigit() for ch in url)
    letters = sum(ch.isalpha() for ch in url)
    special = sum(not ch.isalnum() for ch in url)
    host_parts = [p for p in host.split(".") if p]

    has_ipv4 = 1.0 if re.search(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", host) else 0.0
    has_suspicious_keyword = 1.0 if re.search(
        r"(login|verify|update|secure|account|bank|signin|password|confirm)",
        url,
    ) else 0.0

    return [
        float(len(url)),
        float(len(host)),
        float(len(path)),
        float(len(query)),
        float(len(host_parts)),
        float(url.count(".")),
        float(url.count("-")),
        float(url.count("@")),
        float(url.count("?")),
        float(url.count("=")),
        float(url.count("/")),
        float(digits),
        float(letters),
        float(special),
        float(path.count("//")),
        1.0 if "https" in url else 0.0,
        has_ipv4,
        has_suspicious_keyword,
    ]


def build_feature_matrix(inputs: list[str]) -> list[list[float]]:
    """Convert cleaned URL strings into a numeric feature matrix."""

    if not inputs:
        raise ValueError("Cannot build features from an empty input list.")
    return [_extract_url_features(text) for text in inputs]


def load_feature_matrix_and_labels(
    file_path: str | Path,
    text_column: str = "URL",
    label_column: str = "Label",
    deduplicate: bool = True,
) -> tuple[list[list[float]], list[int]]:
    """Load dataset and return numeric features (X) and labels (y)."""

    inputs, labels = load_phishing_dataset(
        file_path=file_path,
        text_column=text_column,
        label_column=label_column,
        deduplicate=deduplicate,
    )
    features = build_feature_matrix(inputs)
    return features, labels


def split_train_test(
    inputs: list[str],
    labels: list[int],
    train_ratio: float = 0.7,
    random_seed: int = 42,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Split inputs/labels into train and test sets."""

    if len(inputs) != len(labels):
        raise ValueError("Inputs and labels must have the same length.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
    if not inputs:
        raise ValueError("Cannot split an empty dataset.")

    indices = list(range(len(inputs)))
    rng = random.Random(random_seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    x_train = [inputs[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_test = [inputs[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    return x_train, x_test, y_train, y_test


def _oversample_minority_class(
    x_train: list[object], y_train: list[int], random_seed: int = 42
) -> tuple[list[object], list[int]]:
    """Randomly oversample the minority class in the training split only."""

    if len(x_train) != len(y_train):
        raise ValueError("x_train and y_train must have the same length.")
    if not x_train:
        raise ValueError("Cannot rebalance an empty training set.")

    class_to_indices: dict[int, list[int]] = {}
    for idx, label in enumerate(y_train):
        class_to_indices.setdefault(label, []).append(idx)

    if len(class_to_indices) < 2:
        return x_train, y_train

    majority_count = max(len(indices) for indices in class_to_indices.values())
    rng = random.Random(random_seed)

    balanced_indices: list[int] = []
    for indices in class_to_indices.values():
        balanced_indices.extend(indices)
        needed = majority_count - len(indices)
        if needed > 0:
            balanced_indices.extend(rng.choices(indices, k=needed))

    rng.shuffle(balanced_indices)
    x_balanced = [x_train[i] for i in balanced_indices]
    y_balanced = [y_train[i] for i in balanced_indices]
    return x_balanced, y_balanced


def split_feature_matrix_train_test(
    features: list[list[float]],
    labels: list[int],
    train_ratio: float = 0.7,
    random_seed: int = 42,
    balance_train: bool = True,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Split feature matrix/labels into train and test sets.

    If `balance_train` is True, the minority class is oversampled in training data.
    """

    if len(features) != len(labels):
        raise ValueError("Features and labels must have the same length.")
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1 (exclusive).")
    if not features:
        raise ValueError("Cannot split an empty dataset.")

    indices = list(range(len(features)))
    rng = random.Random(random_seed)
    rng.shuffle(indices)

    split_idx = int(len(indices) * train_ratio)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    x_train = [features[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_test = [features[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]

    if balance_train:
        x_train, y_train = _oversample_minority_class(
            x_train=x_train, y_train=y_train, random_seed=random_seed
        )

    return x_train, x_test, y_train, y_test


def load_and_split_phishing_dataset(
    file_path: str | Path,
    text_column: str = "URL",
    label_column: str = "Label",
    deduplicate: bool = True,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    balance_train: bool = True,
) -> tuple[list[str], list[str], list[int], list[int]]:
    """Load dataset from disk and return a shuffled train/test split.

    If `balance_train` is True, the minority class is oversampled in training data.
    """

    inputs, labels = load_phishing_dataset(
        file_path=file_path,
        text_column=text_column,
        label_column=label_column,
        deduplicate=deduplicate,
    )
    x_train, x_test, y_train, y_test = split_train_test(
        inputs=inputs,
        labels=labels,
        train_ratio=train_ratio,
        random_seed=random_seed,
    )

    if balance_train:
        x_train, y_train = _oversample_minority_class(
            x_train=x_train, y_train=y_train, random_seed=random_seed
        )

    return x_train, x_test, y_train, y_test


def load_and_split_feature_matrix(
    file_path: str | Path,
    text_column: str = "URL",
    label_column: str = "Label",
    deduplicate: bool = True,
    train_ratio: float = 0.7,
    random_seed: int = 42,
    balance_train: bool = True,
) -> tuple[list[list[float]], list[list[float]], list[int], list[int]]:
    """Load dataset, featurize it, and return a shuffled train/test split.

    If `balance_train` is True, the minority class is oversampled in training data.
    """

    features, labels = load_feature_matrix_and_labels(
        file_path=file_path,
        text_column=text_column,
        label_column=label_column,
        deduplicate=deduplicate,
    )
    return split_feature_matrix_train_test(
        features=features,
        labels=labels,
        train_ratio=train_ratio,
        random_seed=random_seed,
        balance_train=balance_train,
    )
