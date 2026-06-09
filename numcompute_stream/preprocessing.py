"""
preprocessing.py — Streaming-compatible data preprocessing for numcompute_stream.

Classes
-------
StandardScaler   : Incremental z-score normalisation (Welford-based)
MinMaxScaler     : Incremental min-max scaling
Imputer          : Online missing-value imputation (mean or constant)
OneHotEncoder    : Incrementally expanding one-hot encoding

All transformers implement:
  .partial_fit(X)          → update internal statistics
  .transform(X)            → apply the learned transform
  .fit_transform(X)        → partial_fit then transform in one call
"""

import numpy as np
from .stats import RunningStats
import warnings

# ---------------------------------------------------------------------------
# StandardScaler
# ---------------------------------------------------------------------------

class StandardScaler:
    """
    Incremental StandardScaler using Welford's online algorithm.

    Transforms features to zero mean and unit variance.
    Safe against zero-variance columns (returns 0 instead of NaN).

    Example
    -------
    scaler = StandardScaler()
    for X_chunk in chunks:
        scaler.partial_fit(X_chunk)
    X_scaled = scaler.transform(X_new)
    """

    def __init__(self):
        self._stats = None
        self.n_features_ = None

    def partial_fit(self, X):
        """
        Update mean and variance estimates with X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        """
        X = self._validate(X)
        if self._stats is None:
            self._stats = RunningStats(X.shape[1])
            self.n_features_ = X.shape[1]
        self._stats.update(X)
        if X.shape[0] == 0:
            return self  # Do absolutely nothing and return safely
        return self

    def transform(self, X):
        """
        Apply z-score normalisation using running mean and std.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        X_scaled : np.ndarray, same shape
        """
        if self._stats is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = self._validate(X)
        std = self._stats.std
        # Avoid division by zero for zero-variance features
        std_safe = np.where(std == 0, 1.0, std)
        return (X - self._stats.mean) / std_safe
        if X.shape[0] == 0:
            return X

    def fit_transform(self, X):
        """partial_fit then transform."""
        return self.partial_fit(X).transform(X)

    @property
    def mean_(self):
        if self._stats is None:
            raise RuntimeError("Not fitted yet.")
        return self._stats.mean

    @property
    def var_(self):
        if self._stats is None:
            raise RuntimeError("Not fitted yet.")
        return self._stats.variance

    def _validate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError(f"Expected 2-D array, got shape {X.shape}")
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        return X


# ---------------------------------------------------------------------------
# MinMaxScaler
# ---------------------------------------------------------------------------

class MinMaxScaler:
    """
    Incremental Min-Max scaler.  Scales features to [0, 1].

    Tracks running min and max across all chunks seen so far.
    """

    def __init__(self):
        self._min = None
        self._max = None
        self.n_features_ = None

    def partial_fit(self, X):
        X = self._validate(X)
        chunk_min = np.nanmin(X, axis=0)
        chunk_max = np.nanmax(X, axis=0)
        if self._min is None:
            self._min = chunk_min
            self._max = chunk_max
            self.n_features_ = X.shape[1]
        else:
            self._min = np.minimum(self._min, chunk_min)
            self._max = np.maximum(self._max, chunk_max)
        return self

    def transform(self, X):
        if self._min is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = self._validate(X)
        range_ = self._max - self._min
        range_safe = np.where(range_ == 0, 1.0, range_)
        return (X - self._min) / range_safe

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)

    def _validate(self, X):
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.n_features_ is not None and X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        return X


# ---------------------------------------------------------------------------
# Imputer
# ---------------------------------------------------------------------------

class Imputer:
    """
    Incremental missing-value imputer.

    Strategies
    ----------
    'mean'     : replace NaN with running column mean
    'constant' : replace NaN with a fixed fill_value
    """

    def __init__(self, strategy="mean", fill_value=0.0):
        if strategy not in ("mean", "constant"):
            raise ValueError(f"strategy must be 'mean' or 'constant', got '{strategy}'")
        self.strategy   = strategy
        self.fill_value = fill_value
        self._stats     = None
        self.n_features_ = None

    def partial_fit(self, X):
        X_arr = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self._stats is None:
            self._stats = RunningStats(X_arr.shape[1])
            self.n_features_ = X_arr.shape[1]
        if self.strategy == "mean":
            # Only update stats on non-NaN values
            mask = ~np.isnan(X_arr).any(axis=1)
            if mask.any():
                self._stats.update(X_arr[mask])
        return self

    def transform(self, X):
        X_arr = np.array(np.atleast_2d(np.asarray(X, dtype=np.float64)), copy=True)
        nan_mask = np.isnan(X_arr)
        if not nan_mask.any():
            return X_arr
        if self.strategy == "mean":
            if self._stats is None:
                raise RuntimeError("Call partial_fit before transform.")
            fill = self._stats.mean
        else:
            fill = np.full(X_arr.shape[1], self.fill_value)
        # Replace NaN column-wise
        for j in range(X_arr.shape[1]):
            col_mask = nan_mask[:, j]
            X_arr[col_mask, j] = fill[j]
        return X_arr

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)


# ---------------------------------------------------------------------------
# OneHotEncoder
# ---------------------------------------------------------------------------

class OneHotEncoder:
    """
    Incremental One-Hot Encoder for integer-valued categorical columns.

    Discovers new categories as chunks arrive and expands the output width.
    """

    def __init__(self):
        self.categories_ = []   # list of sorted arrays, one per feature
        self.n_features_in_ = None

    def partial_fit(self, X):
        """
        Discover categories in X (integer array of categorical columns).

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_cat_features)
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        n_features = X.shape[1]

        if self.n_features_in_ is None:
            self.n_features_in_ = n_features
            self.categories_ = [np.array([], dtype=np.int64)] * n_features

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f"Expected {self.n_features_in_} categorical features, got {X.shape[1]}"
            )

        for j in range(n_features):
            new_cats = np.unique(X[:, j])
            merged   = np.union1d(self.categories_[j], new_cats)
            self.categories_[j] = merged
        return self

    def transform(self, X):
        """
        One-hot encode X.

        Returns
        -------
        encoded : np.ndarray, shape (n_samples, total_categories)
        """
        if self.n_features_in_ is None:
            raise RuntimeError("Call partial_fit before transform.")
        X = np.atleast_2d(np.asarray(X, dtype=np.int64))
        n = X.shape[0]
        cols = []
        for j, cats in enumerate(self.categories_):
            col_enc = np.zeros((n, len(cats)), dtype=np.float64)
            for k, cat in enumerate(cats):
                col_enc[:, k] = (X[:, j] == cat).astype(np.float64)
            cols.append(col_enc)
        return np.hstack(cols) if cols else np.zeros((n, 0))

    def fit_transform(self, X):
        return self.partial_fit(X).transform(X)