import numpy as np
from typing import List, Tuple, Any, Dict, Optional, Union

class Pipeline:
    """
    A streaming machine learning pipeline built from scratch using pure NumPy.
    Chains a sequence of data transformers with a final learning estimator.
    
    All intermediate transformer steps must implement both `.partial_fit(X_chunk)` 
    and `.transform(X_chunk)`. The final step (estimator) must implement 
    `.partial_fit(X_chunk, y_chunk)` and `.predict(X_chunk)`.
    """
    def __init__(self, steps: List[Tuple[str, Any]]) -> None:
        """
        Initializes the streaming pipeline structure.
        
        Args:
            steps: List of (name, transform_or_estimator_instance) tuples 
                   specifying the sequence of execution.
                   Example: [('scaler', StandardScaler()), ('model', EnsembleClassifier())]
        """
        if not steps:
            raise ValueError("Pipeline must contain at least one step execution block.")
            
        self.steps = steps
        self._validate_steps()
        
    def _validate_steps(self) -> None:
        """
        Validates the API consistency and structural setup of the pipeline steps
        to ensure compatibility under streaming conditions.
        """
        names, estimators = zip(*self.steps)
        
        # 1. Enforce unique naming constraints
        if len(set(names)) != len(names):
            raise ValueError(f"Pipeline step names must be completely unique. Received names: {names}")
            
        # 2. Validate all intermediate components are transformers
        for name, step in self.steps[:-1]:
            if not hasattr(step, "partial_fit") or not hasattr(step, "transform"):
                raise TypeError(
                    f"Intermediate pipeline step '{name}' ({type(step).__name__}) "
                    f"must implement both '.partial_fit()' and '.transform()' for streaming compatibility."
                )
                
        # 3. Validate the final component is an estimator
        final_name, final_step = self.steps[-1]
        if not hasattr(final_step, "partial_fit") or not hasattr(final_step, "predict"):
            raise TypeError(
                f"The final pipeline estimator '{final_name}' ({type(final_step).__name__}) "
                f"must implement '.partial_fit()' and '.predict()' to serve as an endpoint."
            )

    @property
    def named_steps(self) -> Dict[str, Any]:
        """
        Returns a dictionary mapping step names to their respective instances.
        Allows for easy extraction of intermediate statistical states (e.g., extracting means from a scaler).
        """
        return dict(self.steps)

    def _validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Verifies input types and structural shapes to guarantee matrix operations do not fail down the line.
        """
        if not isinstance(X, np.ndarray):
            raise TypeError(f"Pipeline expects feature inputs as a numpy.ndarray, got {type(X)}")
            
        if len(X.shape) != 2:
            raise ValueError(f"X must be a 2D matrix of shape (n_samples, n_features), got shape {X.shape}")

        if y is not None:
            if not isinstance(y, np.ndarray):
                raise TypeError(f"Pipeline expects target inputs as a numpy.ndarray, got {type(y)}")
            if len(y.shape) != 1:
                raise ValueError(f"y target array must be 1D with shape (n_samples,), got shape {y.shape}")
            if X.shape[0] != y.shape[0]:
                raise ValueError(f"Row count mismatch: X has {X.shape[0]} rows, but y has {y.shape[0]} rows.")

    def partial_fit(self, X_chunk: np.ndarray, y_chunk: np.ndarray, **fit_params: Any) -> 'Pipeline':
        """
        Incrementally fits the intermediate transformers and the final estimator 
        using a streaming chunk of training data.
        
        Args:
            X_chunk: Current feature matrix slice, shape (n_samples, n_features).
            y_chunk: Current target class vector slice, shape (n_samples,).
            fit_params: Optional keyword arguments routed directly to the final estimator.
            
        Returns:
            self: The updated Pipeline instance.
        """
        self._validate_input(X_chunk, y_chunk)
        
        # Handle zero-variance or empty streaming chunk edge cases safely
        if X_chunk.shape[0] == 0:
            return self
            
        X_transformed = X_chunk.copy()
        
        # 1. Sequentially partial_fit and transform through all intermediate transformers
        for name, step in self.steps[:-1]:
            # Update the transformer's internal running state (e.g., updating running mean/variance)
            step.partial_fit(X_transformed)
            # Route data forward through the updated transformer
            X_transformed = step.transform(X_transformed)
            
        # 2. Final incremental adaptation step for your decision tree or ensemble
        final_name, final_estimator = self.steps[-1]
        final_estimator.partial_fit(X_transformed, y_chunk, **fit_params)
        
        return self

    def predict(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Transforms a feature chunk through all intermediate steps and generates 
        vectorized class predictions using the final estimator.
        
        Args:
            X_chunk: Feature matrix to evaluate, shape (n_samples, n_features).
            
        Returns:
            A 1D numpy array of class predictions, shape (n_samples,).
        """
        self._validate_input(X_chunk)
        
        if X_chunk.shape[0] == 0:
            return np.array([], dtype=int)
            
        X_transformed = X_chunk.copy()
        
        # 1. Sequence the data chunk through all data transformers using .transform()
        for name, step in self.steps[:-1]:
            X_transformed = step.transform(X_transformed)
            
        # 2. Generate predictions using the terminal estimator model
        final_name, final_estimator = self.steps[-1]
        return final_estimator.predict(X_transformed)

    def predict_proba(self, X_chunk: np.ndarray) -> np.ndarray:
        """
        Transforms a feature chunk and generates probabilistic allocations across class targets.
        
        Args:
            X_chunk: Feature matrix to evaluate, shape (n_samples, n_features).
            
        Returns:
            A 2D probability matrix of shape (n_samples, n_classes).
        """
        self._validate_input(X_chunk)
        
        final_name, final_estimator = self.steps[-1]
        if not hasattr(final_estimator, "predict_proba"):
            raise AttributeError(f"The terminal estimator '{final_name}' does not implement 'predict_proba'.")
            
        if X_chunk.shape[0] == 0:
            return np.empty((0, len(getattr(final_estimator, 'classes_', []))))
            
        X_transformed = X_chunk.copy()
        
        # 1. Transform features through intermediate transformers
        for name, step in self.steps[:-1]:
            X_transformed = step.transform(X_transformed)
            
        # 2. Compute probabilities from the final model
        return final_estimator.predict_proba(X_transformed)