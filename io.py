"""
io.py — Custom I/O utilities for numcompute_stream.

Provides:
  - load_csv(path)        : Load a CSV file into a NumPy array + header
  - split_chunks(X, y, n) : Split dataset into n equal streaming chunks
  - iter_chunks(X, y, n)  : Generator that yields (X_chunk, y_chunk) one at a time

No pandas or sklearn — pure Python + NumPy only.
"""

import numpy as np


def load_csv(path, delimiter=",", skip_header=True, target_col=-1):
    """
    Load a CSV file into NumPy arrays X (features) and y (labels).

    Parameters
    ----------
    path        : str   — file path to the CSV
    delimiter   : str   — column separator (default ',')
    skip_header : bool  — whether the first row is a header (default True)
    target_col  : int   — column index of the label; -1 means last column

    Returns
    -------
    X      : np.ndarray, shape (n_samples, n_features)  — feature matrix
    y      : np.ndarray, shape (n_samples,)              — label vector
    header : list of str or None                         — column names if present
    """
    if not isinstance(path, str):
        raise TypeError(f"path must be a string, got {type(path)}")

    rows = []
    header = None

    with open(path, "r") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue  # skip blank lines
            parts = line.split(delimiter)
            if i == 0 and skip_header:
                header = [p.strip() for p in parts]
                continue
            try:
                rows.append([float(p.strip()) for p in parts])
            except ValueError as e:
                raise ValueError(f"Non-numeric value on line {i+1}: {e}")

    if len(rows) == 0:
        raise ValueError("CSV file is empty or contains only a header.")

    data = np.array(rows, dtype=np.float64)

    # Separate features and target
    n_cols = data.shape[1]
    col_idx = target_col % n_cols  # handle negative indexing
    feature_cols = [c for c in range(n_cols) if c != col_idx]

    X = data[:, feature_cols]
    y = data[:, col_idx]

    return X, y, header


def split_chunks(X, y, n_chunks):
    """
    Split arrays X and y into n_chunks roughly equal parts.

    Parameters
    ----------
    X        : np.ndarray, shape (n_samples, n_features)
    y        : np.ndarray, shape (n_samples,)
    n_chunks : int — number of chunks to split into

    Returns
    -------
    X_chunks : list of np.ndarray
    y_chunks : list of np.ndarray
    """
    if not isinstance(n_chunks, int) or n_chunks < 1:
        raise ValueError(f"n_chunks must be a positive integer, got {n_chunks}")
    if X.shape[0] != y.shape[0]:
        raise ValueError(f"X and y must have the same number of rows. Got {X.shape[0]} vs {y.shape[0]}")
    if n_chunks > X.shape[0]:
        raise ValueError(f"n_chunks ({n_chunks}) cannot exceed number of samples ({X.shape[0]})")

    X_chunks = np.array_split(X, n_chunks)
    y_chunks = np.array_split(y, n_chunks)
    return X_chunks, y_chunks


def iter_chunks(X, y, n_chunks):
    """
    Generator that yields (X_chunk, y_chunk) one at a time.
    Memory-efficient alternative to split_chunks for large datasets.

    Parameters
    ----------
    X        : np.ndarray
    y        : np.ndarray
    n_chunks : int

    Yields
    ------
    (X_chunk, y_chunk) : tuple of np.ndarray
    """
    X_chunks, y_chunks = split_chunks(X, y, n_chunks)
    for xc, yc in zip(X_chunks, y_chunks):
        yield xc, yc