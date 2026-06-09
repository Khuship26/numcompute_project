"""
tree.py — Decision Tree Classifier from scratch for numcompute_stream.

Implements a binary decision tree with:
  - Gini impurity or Shannon entropy as split criteria
  - Depth limiting, min_samples_split, max_features (random subspace)
  - .partial_fit(X_chunk, y_chunk) for online / streaming growth
  - Fully vectorised NumPy splits (no Python loops in the hot path)

Why decision trees?
  At each node we pick the feature + threshold that best separates classes.
  "Best" = minimises weighted impurity of the two child nodes.
  Gini impurity: 1 - sum(p_k^2)  (how often a random pick would be wrong)
  Entropy:       -sum(p_k * log2(p_k))  (information needed to describe a sample)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal node representation
# ---------------------------------------------------------------------------

class _Node:
    """A single node in the decision tree."""
    __slots__ = [
        "feature", "threshold",   # split rule
        "left", "right",          # child _Node references
        "value",                  # leaf prediction (most common class)
        "n_samples",              # number of training samples that passed through
        "impurity",               # impurity at this node
    ]

    def __init__(self):
        self.feature   = None
        self.threshold = None
        self.left      = None
        self.right     = None
        self.value     = None
        self.n_samples = 0
        self.impurity  = 0.0

    @property
    def is_leaf(self):
        return self.left is None and self.right is None


# ---------------------------------------------------------------------------
# Impurity functions (vectorised)
# ---------------------------------------------------------------------------

def _gini(y):
    """Gini impurity of label array y."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1.0 - float(np.dot(p, p))


def _entropy(y):
    """Shannon entropy of label array y."""
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    # Clip to avoid log(0)
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.dot(p, np.log2(p)))


# ---------------------------------------------------------------------------
# Decision Tree
# ---------------------------------------------------------------------------

class DecisionTreeClassifier:
    """
    Binary Decision Tree Classifier built from scratch.

    Parameters
    ----------
    max_depth         : int or None  — maximum tree depth (None = unlimited)
    min_samples_split : int          — minimum samples to attempt a split
    criterion         : str          — 'gini' or 'entropy'
    max_features      : int, float, str, or None
        - int   : use exactly this many random features per split
        - float : fraction of features (e.g., 0.5)
        - 'sqrt': use sqrt(n_features) features
        - 'log2': use log2(n_features) features
        - None  : use all features

    Streaming
    ---------
    .partial_fit(X_chunk, y_chunk) rebuilds the tree on the cumulative
    dataset. We store a reservoir of samples so the tree always reflects
    everything seen so far (up to reservoir_size samples).

    Usage
    -----
    tree = DecisionTreeClassifier(max_depth=5, criterion='gini')
    tree.partial_fit(X_train, y_train)
    preds = tree.predict(X_test)
    """

    def __init__(
        self,
        max_depth=None,
        min_samples_split=2,
        criterion="gini",
        max_features=None,
        reservoir_size=5000,
        random_state=None,
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError(f"criterion must be 'gini' or 'entropy', got '{criterion}'")
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.criterion         = criterion
        self.max_features      = max_features
        self.reservoir_size    = reservoir_size
        self.random_state      = random_state

        self._rng         = np.random.RandomState(random_state)
        self._root        = None
        self.n_features_  = None
        self.classes_     = None

        # Reservoir for streaming: stores up to reservoir_size samples
        self._res_X = None
        self._res_y = None
        self._n_seen = 0

    # ------------------------------------------------------------------
    # Streaming fit
    # ------------------------------------------------------------------

    def partial_fit(self, X, y, classes=None):
        """
        Update the tree with a new chunk of data.

        Stores data in a reservoir (random replacement when full),
        then rebuilds the tree from the reservoir.

        Parameters
        ----------
        X       : np.ndarray, shape (n, n_features)
        y       : np.ndarray, shape (n,)
        classes : array-like or None — known class labels (optional)
        """
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        y = np.asarray(y, dtype=np.int64).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")

        n = X.shape[0]
        self.n_features_ = X.shape[1]

        # --- Update reservoir ---
        if self._res_X is None:
            take = min(n, self.reservoir_size)
            self._res_X = X[:take].copy()
            self._res_y = y[:take].copy()
            self._n_seen = take
            X_remaining = X[take:]
            y_remaining = y[take:]
        else:
            X_remaining = X
            y_remaining = y

        for i in range(len(X_remaining)):
            self._n_seen += 1
            if len(self._res_X) < self.reservoir_size:
                self._res_X = np.vstack([self._res_X, X_remaining[i]])
                self._res_y = np.append(self._res_y, y_remaining[i])
            else:
                j = self._rng.randint(0, self._n_seen)
                if j < self.reservoir_size:
                    self._res_X[j] = X_remaining[i]
                    self._res_y[j] = y_remaining[i]

        # Determine classes
        if classes is not None:
            self.classes_ = np.sort(np.asarray(classes))
        else:
            self.classes_ = np.unique(self._res_y)

        # Rebuild tree from reservoir
        self._root = self._build(self._res_X, self._res_y, depth=0)
        return self

    def fit(self, X, y):
        """Full batch fit (resets any previous state)."""
        self._res_X  = None
        self._res_y  = None
        self._n_seen = 0
        return self.partial_fit(X, y)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X):
        """
        Predict class labels.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)

        Returns
        -------
        y_pred : np.ndarray, shape (n_samples,)
        """
        if self._root is None:
            raise RuntimeError("Tree is not fitted. Call partial_fit first.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        return np.array([self._predict_one(x, self._root) for x in X], dtype=np.int64)

    def predict_proba(self, X):
        """
        Return class probabilities estimated by leaf class distribution.

        Returns
        -------
        proba : np.ndarray, shape (n_samples, n_classes)
        """
        if self._root is None:
            raise RuntimeError("Tree is not fitted. Call partial_fit first.")
        X = np.atleast_2d(np.asarray(X, dtype=np.float64))
        probas = [self._predict_proba_one(x, self._root) for x in X]
        return np.array(probas, dtype=np.float64)

    # ------------------------------------------------------------------
    # Tree building (recursive)
    # ------------------------------------------------------------------

    def _build(self, X, y, depth):
        """Recursively build the tree, returning a _Node."""
        node = _Node()
        node.n_samples = len(y)
        impurity_fn    = _gini if self.criterion == "gini" else _entropy
        node.impurity  = impurity_fn(y)

        # Leaf conditions
        if (
            len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or len(np.unique(y)) == 1
        ):
            node.value = self._leaf_value(y)
            node.class_counts = self.class_counts(y)
            return node
        # Find best split
        feature, threshold = self._best_split(X, y)
        if feature is None:
            node.value = self._leaf_value(y)
            node.class_counts = self.class_counts(y)
            classes, counts = np.unique(y, return_counts=True)
            node.class_counts = dict(zip(classes, counts))
            return node

        node.feature   = feature
        node.threshold = threshold

        left_mask  = X[:, feature] <= threshold
        right_mask = ~left_mask

        # Guard against empty splits
        if left_mask.sum() == 0 or right_mask.sum() == 0:
            node.feature   = None
            node.threshold = None
            node.value     = self._leaf_value(y)
            node.class_counts = self._class_counts(y)
            return node

        node.left  = self._build(X[left_mask],  y[left_mask],  depth + 1)
        node.right = self._build(X[right_mask], y[right_mask], depth + 1)
        return node

    def _best_split(self, X, y):
        """
        Find the (feature, threshold) pair with the lowest weighted impurity.

        Uses vectorised operations over candidate thresholds for speed.
        """
        n, d      = X.shape
        impurity_fn = _gini if self.criterion == "gini" else _entropy
        best_gain  = -np.inf
        best_feat  = None
        best_thresh = None
        parent_impurity = impurity_fn(y)

        # Random feature subsampling
        n_features_to_try = self._n_features_to_try(d)
        feature_indices   = self._rng.choice(d, size=n_features_to_try, replace=False)

        for j in feature_indices:
            col = X[:, j]
            thresholds = np.unique(col)
            if len(thresholds) < 2:
                continue

            # Candidate thresholds = midpoints between consecutive sorted unique values
            midpoints = (thresholds[:-1] + thresholds[1:]) / 2.0

            for thresh in midpoints:
                left_mask  = col <= thresh
                right_mask = ~left_mask
                n_left  = left_mask.sum()
                n_right = right_mask.sum()
                if n_left == 0 or n_right == 0:
                    continue

                w_left  = n_left  / n
                w_right = n_right / n
                gain = parent_impurity \
                       - w_left  * impurity_fn(y[left_mask]) \
                       - w_right * impurity_fn(y[right_mask])

                if gain > best_gain:
                    best_gain   = gain
                    best_feat   = j
                    best_thresh = thresh

        return best_feat, best_thresh

    def _n_features_to_try(self, d):
        mf = self.max_features
        if mf is None:
            return d
        elif mf == "sqrt":
            return max(1, int(np.sqrt(d)))
        elif mf == "log2":
            return max(1, int(np.log2(d)))
        elif isinstance(mf, float):
            return max(1, int(mf * d))
        elif isinstance(mf, int):
            return min(mf, d)
        else:
            raise ValueError(f"Unknown max_features value: {mf}")

    def _leaf_value(self, y):
        """Most common class in y."""
        if len(y) == 0:
            return 0
        values, counts = np.unique(y, return_counts=True)
        return int(values[np.argmax(counts)])
    def _class_counts(self, y):
            return {cls: int(np.sum(y == cls)) for cls in self.classes_}

    def class_counts(self, y):
        """Dict of class → count for probability estimation."""
        counts = {}
        for c in self.classes_:
            counts[int(c)] = int(np.sum(y == c))
        return counts

    def _predict_one(self, x, node):
        if node.is_leaf:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(x, node.left)
        return self._predict_one(x, node.right)

    def _predict_proba_one(self, x, node):
        if node.is_leaf:
            total = node.n_samples if node.n_samples > 0 else 1
            return np.array(
                [node._class_counts.get(int(c), 0) / total for c in self.classes_],
                dtype=np.float64,
            )
        if x[node.feature] <= node.threshold:
            return self._predict_proba_one(x, node.left)
        return self._predict_proba_one(x, node.right)

    @property
    def depth(self):
        """Actual maximum depth of the fitted tree."""
        if self._root is None:
            return 0
        return self._tree_depth(self._root)

    def _tree_depth(self, node):
        if node is None or node.is_leaf:
            return 0
        return 1 + max(self._tree_depth(node.left), self._tree_depth(node.right))