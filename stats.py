"""
stats.py — Streaming statistical functions for numcompute_stream.

All statistics support chunk-wise updates so they never need to see
the full dataset at once.

Classes
-------
RunningStats   : Welford online algorithm for mean and variance
StreamQuantile : Approximate streaming quantiles (P²-inspired histogram)
StreamHistogram: Sliding-window or cumulative histogram

Standalone functions
--------------------
chunk_mean(X)     : Mean along axis 0 for a single chunk
chunk_variance(X) : Variance along axis 0 for a single chunk
"""

import numpy as np


# ---------------------------------------------------------------------------
# Standalone chunk functions (no state)
# ---------------------------------------------------------------------------

def chunk_mean(X):
    """
    Compute column-wise mean of a 2-D array chunk.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)

    Returns
    -------
    mean : np.ndarray, shape (d,)
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] == 0:
        raise ValueError("chunk_mean received an empty chunk.")
    return np.nanmean(X, axis=0)


def chunk_variance(X, ddof=0):
    """
    Compute column-wise variance of a 2-D array chunk.

    Parameters
    ----------
    X    : np.ndarray, shape (n, d)
    ddof : int — delta degrees of freedom (0 = population, 1 = sample)

    Returns
    -------
    var : np.ndarray, shape (d,)
    """
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    if X.shape[0] == 0:
        raise ValueError("chunk_variance received an empty chunk.")
    return np.nanvar(X, axis=0, ddof=ddof)


# ---------------------------------------------------------------------------
# RunningStats — Welford's online algorithm
# ---------------------------------------------------------------------------

class RunningStats:
    """
    Incrementally compute mean and variance using Welford's algorithm.

    Welford's algorithm is numerically stable and handles one chunk at
    a time without storing raw data.

    Usage
    -----
    rs = RunningStats(n_features=4)
    rs.update(X_chunk)      # call for each chunk
    print(rs.mean)          # current running mean
    print(rs.variance)      # current running variance (population)
    print(rs.std)           # current running std
    """

    def __init__(self, n_features):
        """
        Parameters
        ----------
        n_features : int — number of features (columns)
        """
        if not isinstance(n_features, int) or n_features < 1:
            raise ValueError(f"n_features must be a positive int, got {n_features}")
        self.n_features = n_features
        self.reset()

    def reset(self):
        """Reset all accumulators to zero."""
        self.n = 0                                      # total samples seen
        self._mean = np.zeros(self.n_features)          # running mean
        self._M2   = np.zeros(self.n_features)          # sum of squared deviations

    def update(self, X):
        """
        Update running statistics with a new chunk.

        Parameters
        ----------
        X : np.ndarray, shape (n_chunk, n_features)
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if X.ndim != 2 or X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected shape (n, {self.n_features}), got {X.shape}"
            )

        # Use Chan's parallel algorithm to merge chunk stats into running stats
        if X.shape[0] == 0:
         return self
        n_b   = X.shape[0]
        mean_b = np.nanmean(X, axis=0)
        M2_b   = np.nanvar(X, axis=0) * n_b  # sum of squared deviations for chunk

        n_a   = self.n
        mean_a = self._mean
        M2_a  = self._M2

        n_total = n_a + n_b
        delta   = mean_b - mean_a
        new_mean = mean_a + delta * (n_b / n_total)
        new_M2   = M2_a + M2_b + (delta ** 2) * (n_a * n_b / n_total)

        self.n      = n_total
        self._mean  = new_mean
        self._M2    = new_M2

    @property
    def mean(self):
        """Running mean, shape (n_features,)."""
        return self._mean.copy()

    @property
    def variance(self):
        """Population variance, shape (n_features,). Returns 0 if n < 2."""
        if self.n < 2:
            return np.zeros(self.n_features)
        return self._M2 / self.n

    @property
    def std(self):
        """Standard deviation, shape (n_features,)."""
        return np.sqrt(self.variance)

    def update_stats(self, X_chunk):
        """Alias for update() — satisfies the assignment's update_stats API."""
        self.update(X_chunk)


# ---------------------------------------------------------------------------
# StreamHistogram — cumulative histogram with optional sliding window
# ---------------------------------------------------------------------------

class StreamHistogram:
    """
    Maintain a per-feature histogram that updates incrementally.

    Parameters
    ----------
    n_features : int
    n_bins     : int    — number of histogram bins
    window     : int or None — if set, only the last `window` samples count
    """

    def __init__(self, n_features, n_bins=20, window=None):
        self.n_features = n_features
        self.n_bins = n_bins
        self.window = window
        self._buffer = []   # stores chunks for sliding window
        self._all_data = [] # stores all data for cumulative mode

    def update(self, X):
        """Add a chunk to the histogram."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        if self.window is not None:
            self._buffer.append(X)
            # Keep only the last `window` samples
            total = sum(c.shape[0] for c in self._buffer)
            while total - self._buffer[0].shape[0] >= self.window:
                total -= self._buffer[0].shape[0]
                self._buffer.pop(0)
        else:
            self._all_data.append(X)

    def update_stats(self, X_chunk):
        """Alias for update() to satisfy assignment API."""
        self.update(X_chunk)

    def get_histogram(self, feature_idx=0):
        """
        Compute histogram counts and bin edges for a feature.

        Returns
        -------
        counts : np.ndarray, shape (n_bins,)
        edges  : np.ndarray, shape (n_bins+1,)
        """
        if feature_idx >= self.n_features:
            raise ValueError(f"feature_idx {feature_idx} out of range for {self.n_features} features")
        source = self._buffer if self.window is not None else self._all_data
        if not source:
            raise RuntimeError("No data has been added yet.")
        data = np.concatenate(source, axis=0)[:, feature_idx]
        counts, edges = np.histogram(data, bins=self.n_bins)
        return counts, edges


# ---------------------------------------------------------------------------
# StreamQuantile — approximate quantiles via a sorted sample
# ---------------------------------------------------------------------------

class StreamQuantile:
    """
    Approximate streaming quantiles using reservoir sampling.

    Maintains a reservoir of at most `max_samples` observations and
    computes quantiles from it.
    """

    def __init__(self, n_features, max_samples=1000):
        self.n_features  = n_features
        self.max_samples = max_samples
        self._reservoir  = None
        self._n_seen     = 0

    def update(self, X):
        """Update reservoir with a new chunk."""
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        n = X.shape[0]

        if self._reservoir is None:
            # First chunk: just take up to max_samples
            self._reservoir = X[:self.max_samples].copy()
        else:
            # Reservoir sampling: randomly replace existing entries
            for i in range(n):
                self._n_seen += 1
                if self._reservoir.shape[0] < self.max_samples:
                    self._reservoir = np.vstack([self._reservoir, X[i]])
                else:
                    j = np.random.randint(0, self._n_seen)
                    if j < self.max_samples:
                        self._reservoir[j] = X[i]
        self._n_seen += n

    def update_stats(self, X_chunk):
        self.update(X_chunk)

    def quantile(self, q):
        """
        Compute per-feature quantiles.

        Parameters
        ----------
        q : float or array-like — quantile(s) in [0, 1]

        Returns
        -------
        result : np.ndarray — quantile values, shape (n_features,) or (len(q), n_features)
        """
        if self._reservoir is None:
            raise RuntimeError("No data has been added yet.")
        q = np.asarray(q)
        return np.quantile(self._reservoir, q, axis=0)