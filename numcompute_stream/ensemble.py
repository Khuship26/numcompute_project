import numpy as np
from typing import Optional, List, Dict, Any
# Assuming your tree module contains your implemented DecisionTreeClassifier
from .tree import DecisionTreeClassifier

class EnsembleClassifier:
    """
    An online/streaming ensemble classifier implementing Online Bagging (Oza & Russell).
    
    This framework adapts traditional bagging to streaming settings by using a 
    Poisson distribution (\lambda=1) to simulate bootstrap sampling on incremental chunks.
    Designed from scratch using pure NumPy with vectorized operations.
    """
    def __init__(
        self, 
        n_estimators: int = 10, 
        max_depth: int = 5, 
        min_samples_split: int = 2,
        max_features: Optional[str] = "sqrt"
    ) -> None:
        """
        Initializes the streaming ensemble classifier.
        
        Args:
            n_estimators: Number of base decision tree models.
            max_depth: Maximum depth limit for each individual tree.
            min_samples_split: Minimum number of samples required to split a node.
            max_features: Strategy for splitting feature subsets ('sqrt', 'log2', or None).
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        
        # Initialize the collection of base streaming trees   
        self.estimators: List[DecisionTreeClassifier] = [
             DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features
            ) for _ in range(self.n_estimators)
        ]
        
        # Track unique classes observed across the entire stream for consistency
        self.classes_: np.ndarray = np.array([], dtype=int)
        
        # Track unique classes observed across the entire stream for consistency
        self.classes_: np.ndarray = np.array([], dtype=int)
        self.is_fitted_ = False

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Enforces strict input shape rules and checks types to guarantee numerical stability.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Expected X to be a numpy.ndarray, got {type(X)}")
        
        if len(X.shape) != 2:
            raise ValueError(f"X must be a 2D matrix of shape (n_samples, n_features), got shape {X.shape}")
            
        if np.isnan(X).any():
            raise ValueError("X contains NaN values. Ensure the pipeline's Imputer runs before the estimator.")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"Expected y to be a numpy.ndarray, got {type(y)}")
            if len(y.shape) != 1:
                raise ValueError(f"y must be a 1D array of shape (n_samples,), got shape {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Length mismatch: X has {X.shape[0]} samples, y has {y.shape[0]} samples.")

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray, classes: Optional[np.ndarray] = None) -> 'EnsembleClassifier':
        """
        Incrementally updates the ensemble using an incoming streaming data chunk.
        
        Args:
            X_chunk: Input feature matrix for the chunk, shape (n_samples, n_features).
            y_chunk: Target labels for the chunk, shape (n_samples,).
            classes: Optional full array of target classes expected over the entire stream.
            
        Returns:
            self: The updated EnsembleClassifier instance.
        """
        # 1. Structural Validation
        self._validate_input(X_chunk, y_chunk)
        
        # 2. Update global known labels/classes dynamically or statically
        if classes is not None:
            self.classes_ = np.unique(classes)
        elif self.classes_.size == 0:
            self.classes_ = np.unique(y_chunk)
        else:
            self.classes_ = np.unique(np.concatenate([self.classes_, y_chunk]))
            
        n_samples = X_chunk.shape[0]
        
        # 3. Handle edge case: Chunk contains zero-variance data or a single sample
        if n_samples == 0:
            return self

        # 4. Core Online Bagging Logic (Vectorized per Tree)
        # Instead of allocating massive individual rows, generate a Poisson weight matrix
        # Shape: (n_estimators, n_samples)
        poisson_weights = np.random.poisson(lam=1.0, size=(self.n_estimators, n_samples))
        
        for idx, tree in enumerate(self.estimators):
            # Extract sample updates allocated to this specific tree
            tree_weights = poisson_weights[idx]
            
            # Identify samples that are chosen at least once (weight > 0)
            active_indices = np.where(tree_weights > 0)[0]
            
            if active_indices.size > 0:
                X_sub = X_chunk[active_indices]
                y_sub = y_chunk[active_indices]
                w_sub = tree_weights[active_indices]
                
                # Update the base tree model.
                # NOTE: Ensure your tree.py partial_fit accepts an optional sample_weight array.
                # If your tree.py treats rows as independent updates, you can repeat rows 
                # based on weights to keep it robust:
                X_expanded = np.repeat(X_sub, w_sub, axis=0)
                y_expanded = np.repeat(y_sub, w_sub, axis=0)
                
                tree.partial_fit(X_expanded, y_expanded, classes=self.classes_)
                
        self.is_fitted_ = True
        return self

    def predict(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Predicts classes for an incoming data chunk by aggregating votes from all trees.
        
        Args:
            X_chunk: Input feature matrix, shape (n_samples, n_features).
            
        Returns:
            Vectorized 1D array of class predictions, shape (n_samples,).
        """
        # 1. Validation & State Checks
        if not self.is_fitted_:
            raise RuntimeError("Cannot predict on a model that has not yet been fitted with partial_fit.")
        self._validate_input(X_chunk)
        
        n_samples = X_chunk.shape[0]
        if n_samples == 0:
            return np.array([], dtype=int)
            
        # 2. Gather predictions across all independent base trees
        # Matrix shape: (n_estimators, n_samples)
        predictions = np.zeros((self.n_estimators, n_samples), dtype=int)
        for idx, tree in enumerate(self.estimators):
            predictions[idx] = tree.predict(X_chunk)
            
        # 3. Vectorized Voting Process
        # We find the most frequent value along axis 0 without using scikit-learn or scipy.
        final_predictions = np.zeros(n_samples, dtype=int)
        
        # Loop over samples for final aggregation (unavoidable for mode collection in pure NumPy,
        # but kept highly efficient)
        for i in range(n_samples):
            sample_votes = predictions[:, i]
            # Use fast counting via bincount
            counts = np.bincount(sample_votes)
            # Find the index of the highest occurring vote (resolving ties deterministically)
            final_predictions[i] = np.argmax(counts)
            
        return final_predictions

    def predict_proba(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Computes predictive class probabilities for an incoming data chunk.
        
        Args:
            X_chunk: Input feature matrix, shape (n_samples, n_features).
            
        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted_:
            raise RuntimeError("Cannot predict probabilities on a model that has not been fitted.")
        self._validate_input(X_chunk)
        
        n_samples = X_chunk.shape[0]
        n_classes = len(self.classes_)
        
        if n_samples == 0:
            return np.empty((0, n_classes))
            
        # Map classes to indices for consistent matrix placement
        class_to_idx = {cls: idx for idx, cls in enumerate(self.classes_)}
        
        # Accumulator for categorical votes: shape (n_samples, n_classes)
        prob_accumulator = np.zeros((n_samples, n_classes))
        
        for tree in self.estimators:
            # If the base tree implements predict_proba, call it directly:
            if hasattr(tree, 'predict_proba'):
                prob_accumulator += tree.predict_proba(X_chunk)
            else:
                # Fallback: create hard single-vote probabilities from standard predict outputs
                preds = tree.predict(X_chunk)
                for i, pred in enumerate(preds):
                    if pred in class_to_idx:
                        prob_accumulator[i, class_to_idx[pred]] += 1.0
                        
        # Normalize the vote sums across axis 1 to produce proper probabilities
        # Handle the numerical stability edge case of zero-sum divisions gracefully
        row_sums = prob_accumulator.sum(axis=1, keepdims=True)
        # Avoid ZeroDivisionError if an unseen state prevents votes
        row_sums[row_sums == 0] = 1.0 
        
        return prob_accumulator / row_sums

    def get_params(self) -> Dict[str, Any]:
        """
        Returns parameters for reproducibility tracking and architectural verification.
        """
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "max_features": self.max_features
        }