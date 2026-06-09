"""
metrics.py — Streaming classification metrics for numcompute_stream.

All metric classes maintain running totals and support:
  .update(y_true_chunk, y_pred_chunk)
  .result()   → current metric value
  .reset()    → clear all accumulators

Classes
-------
Accuracy           : Running fraction of correct predictions
Precision          : TP / (TP + FP) per class (binary or macro)
Recall             : TP / (TP + FN) per class
F1Score            : Harmonic mean of precision and recall
ConfusionMatrix    : Accumulating confusion matrix
RollingAccuracy    : Accuracy over the last N predictions (sliding window)
AUCAccumulator     : Approximate streaming AUC via trapezoid rule on buckets
"""

import numpy as np


def _check_binary(y_true, y_pred):
    """Validate that labels are 1-D and same length."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"y_true shape {y_true.shape} != y_pred shape {y_pred.shape}"
        )
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Accuracy
# ---------------------------------------------------------------------------

class Accuracy:
    """
    Streaming accuracy: cumulative correct / cumulative total.

    Usage
    -----
    acc = Accuracy()
    acc.update(y_true_chunk, y_pred_chunk)
    print(acc.result())   # → float in [0, 1]
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._correct = 0
        self._total   = 0

    def update(self, y_true, y_pred):
        y_true, y_pred = _check_binary(y_true, y_pred)
        self._correct += int(np.sum(y_true == y_pred))
        self._total   += len(y_true)

    def result(self):
        if self._total == 0:
            return 0.0
        return self._correct / self._total


# ---------------------------------------------------------------------------
# Precision, Recall, F1 (binary, positive class = 1)
# ---------------------------------------------------------------------------

class Precision:
    """
    Streaming precision = TP / (TP + FP).
    Uses positive class = 1 by default.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tp = 0
        self._fp = 0

    def update(self, y_true, y_pred):
        y_true, y_pred = _check_binary(y_true, y_pred)
        self._tp += int(np.sum((y_pred == 1) & (y_true == 1)))
        self._fp += int(np.sum((y_pred == 1) & (y_true == 0)))

    def result(self):
        denom = self._tp + self._fp
        return self._tp / denom if denom > 0 else 0.0


class Recall:
    """
    Streaming recall = TP / (TP + FN).
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tp = 0
        self._fn = 0

    def update(self, y_true, y_pred):
        y_true, y_pred = _check_binary(y_true, y_pred)
        self._tp += int(np.sum((y_pred == 1) & (y_true == 1)))
        self._fn += int(np.sum((y_pred == 0) & (y_true == 1)))

    def result(self):
        denom = self._tp + self._fn
        return self._tp / denom if denom > 0 else 0.0


class F1Score:
    """
    Streaming F1 = 2 * P * R / (P + R).
    Internally maintains Precision and Recall objects.
    """

    def __init__(self):
        self._precision = Precision()
        self._recall    = Recall()

    def reset(self):
        self._precision.reset()
        self._recall.reset()

    def update(self, y_true, y_pred):
        self._precision.update(y_true, y_pred)
        self._recall.update(y_true, y_pred)

    def result(self):
        p = self._precision.result()
        r = self._recall.result()
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """
    Accumulating confusion matrix for binary or multi-class problems.

    Parameters
    ----------
    n_classes : int — number of classes
    """

    def __init__(self, n_classes=2):
        if n_classes < 2:
            raise ValueError("n_classes must be >= 2")
        self.n_classes = n_classes
        self.reset()

    def reset(self):
        self._matrix = np.zeros((self.n_classes, self.n_classes), dtype=np.int64)

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
        if y_true.shape != y_pred.shape:
            raise ValueError("y_true and y_pred must have the same length")
        # Vectorised accumulation
        valid = (y_true >= 0) & (y_true < self.n_classes) & \
                (y_pred >= 0) & (y_pred < self.n_classes)
        np.add.at(self._matrix, (y_true[valid], y_pred[valid]), 1)

    def result(self):
        """Return the accumulated confusion matrix."""
        return self._matrix.copy()


# ---------------------------------------------------------------------------
# Rolling (sliding-window) Accuracy
# ---------------------------------------------------------------------------

class RollingAccuracy:
    """
    Accuracy over the last `window` individual predictions.

    Uses a circular buffer for memory efficiency.

    Parameters
    ----------
    window : int — number of recent predictions to keep
    """

    def __init__(self, window=100):
        if window < 1:
            raise ValueError("window must be >= 1")
        self.window = window
        self.reset()

    def reset(self):
        self._buf_true = np.empty(self.window, dtype=np.int64)
        self._buf_pred = np.empty(self.window, dtype=np.int64)
        self._ptr   = 0
        self._count = 0

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.int64).ravel()
        y_pred = np.asarray(y_pred, dtype=np.int64).ravel()
        for yt, yp in zip(y_true, y_pred):
            self._buf_true[self._ptr] = yt
            self._buf_pred[self._ptr] = yp
            self._ptr   = (self._ptr + 1) % self.window
            self._count = min(self._count + 1, self.window)

    def result(self):
        if self._count == 0:
            return 0.0
        t = self._buf_true[:self._count]
        p = self._buf_pred[:self._count]
        return float(np.mean(t == p))


# ---------------------------------------------------------------------------
# AUC Accumulator (approximate, binary classification)
# ---------------------------------------------------------------------------

class AUCAccumulator:
    """
    Approximate streaming AUC using score bucketing.

    Maintains histogram counts for positive and negative classes
    over predicted probability buckets.  AUC is estimated via the
    trapezoid rule on the accumulated ROC curve.

    Parameters
    ----------
    n_bins : int — number of probability buckets
    """

    def __init__(self, n_bins=100):
        self.n_bins = n_bins
        self.reset()

    def reset(self):
        self._pos_hist = np.zeros(self.n_bins)   # counts for y=1
        self._neg_hist = np.zeros(self.n_bins)   # counts for y=0

    def update(self, y_true, y_scores):
        """
        Parameters
        ----------
        y_true   : array-like of int {0, 1}
        y_scores : array-like of float in [0, 1] — predicted probabilities
        """
        y_true   = np.asarray(y_true,   dtype=np.int64).ravel()
        y_scores = np.asarray(y_scores, dtype=np.float64).ravel()
        # Clip and bucket scores
        buckets = np.clip(
            (y_scores * self.n_bins).astype(np.int64), 0, self.n_bins - 1
        )
        np.add.at(self._pos_hist, buckets[y_true == 1], 1)
        np.add.at(self._neg_hist, buckets[y_true == 0], 1)

    def result(self):
        """
        Compute approximate AUC from accumulated histograms.

        Returns
        -------
        auc : float in [0, 1]
        """
        # Build ROC curve: sweep threshold from high to low
        total_pos = self._pos_hist.sum()
        total_neg = self._neg_hist.sum()
        if total_pos == 0 or total_neg == 0:
            return 0.5  # undefined; return chance level

        tpr = np.cumsum(self._pos_hist[::-1]) / total_pos
        fpr = np.cumsum(self._neg_hist[::-1]) / total_neg
        # Prepend (0, 0)
        tpr = np.concatenate([[0.0], tpr])
        fpr = np.concatenate([[0.0], fpr])
        # Trapezoid rule
        auc = float(np.trapz(tpr, fpr))
        return np.clip(auc, 0.0, 1.0)