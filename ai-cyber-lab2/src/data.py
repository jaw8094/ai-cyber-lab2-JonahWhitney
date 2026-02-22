"""Data loading utilities for phishing-vs-benign classification."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable


def _normalize_label(value: object) -> int:
    """Convert common phishing/benign labels to binary integers.

    Returns:
        1 for phishing and 0 for benign.
    """

    text = str(value).strip().lower()

    phishing_values = {"1", "phishing", "phish", "malicious", "true", "yes"}
    benign_values = {"0", "benign", "legitimate", "safe", "false", "no"}

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


def load_phishing_dataset(
    file_path: str | Path,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[int]]:
    """Load URL/email phishing dataset from disk.

    Args:
        file_path: Path to a CSV/TSV/JSONL file containing records.
        text_column: Column containing URL or email text.
        label_column: Column containing phishing/benign labels.

    Returns:
        A tuple of (`inputs`, `labels`) where labels are binary (1=phishing, 0=benign).
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    inputs: list[str] = []
    labels: list[int] = []

    for i, row in enumerate(_iter_rows(path), start=1):
        if text_column not in row:
            raise KeyError(f"Missing text column {text_column!r} in row {i}")
        if label_column not in row:
            raise KeyError(f"Missing label column {label_column!r} in row {i}")

        text = str(row[text_column]).strip()
        if not text:
            raise ValueError(f"Empty input text in row {i}")

        inputs.append(text)
        labels.append(_normalize_label(row[label_column]))

    if not inputs:
        raise ValueError(f"No data rows were found in {path}")

    return inputs, labels
