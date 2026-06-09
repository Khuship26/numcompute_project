import numpy as np
import time
import sys
from typing import Dict, Any, List, Optional

class StreamTrainer:
    """
    Manages incremental learning workflows, real-time logging, metrics tracking,
    and performance monitoring across streaming data chunks.
    
    Implements a strict 'Prequential Evaluation' design pattern (Test-then-Train)
    to ensure performance tracking is scientifically unbiased.
    """
    def __init__(self, pipeline: Any, metrics_manager: Any) -> None:
        """
        Initializes the StreamTrainer.
        
        Args:
            pipeline: A modular pipeline instance supporting .partial_fit() and .predict()
            metrics_manager: A metrics tracker from your metrics.py supporting .update() and .result()
        """
        self.pipeline = pipeline
        self.metrics_manager = metrics_manager
        
        # History structures for performance analytics
        self.history: Dict[str, List[Any]] = {
            "chunk_id": [],
            "chunk_size": [],
            "accuracy": [],
            "cumulative_accuracy": [],
            "fit_time": [],
            "memory_bytes": []
        }
        
    def fit_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, Any]:
        """
        Updates the internal pipeline state using a single data chunk.
        
        Args:
            X_chunk: Input features for the chunk, shape (n_samples, n_features).
            y_chunk: Target labels for the chunk, shape (n_samples,).
            
        Returns:
            A dictionary containing processing metadata for this specific chunk.
        """
        if X_chunk.shape[0] == 0:
            return {"status": "empty_chunk"}
            
        start_time = time.perf_counter()
        
        # Incremental pipeline adjustment
        self.pipeline.partial_fit(X_chunk, y_chunk)
        
        fit_time = time.perf_counter() - start_time
        
        # Safely evaluate system resource consumption (in bytes)
        memory_footprint = X_chunk.nbytes + y_chunk.nbytes
        
        return {
            "fit_time": fit_time,
            "memory_bytes": memory_footprint
        }
        
    def score_chunk(self, X_chunk: np.ndarray, y_chunk: np.ndarray) -> Dict[str, float]:
        """
        Evaluates the existing model on unseen data prior to updating weights.
        
        Args:
            X_chunk: Input features, shape (n_samples, n_features).
            y_chunk: Target labels, shape (n_samples,).
        """
        if X_chunk.shape[0] == 0:
            return {"accuracy": 0.0}
            
        # 1. Generate predictions from the current historical state of the pipeline
        y_pred = self.pipeline.predict(X_chunk)
        
        # 2. Extract accuracy for this individual snapshot
        chunk_correct = np.sum(y_pred == y_chunk)
        chunk_accuracy = chunk_correct / X_chunk.shape[0]
        
        # 3. Permanently accumulate state within your streaming metrics.py module
        self.metrics_manager.update(y_chunk, y_pred)
        
        return {
            "accuracy": chunk_accuracy,
            "cumulative_accuracy": self.metrics_manager.result().get("accuracy", chunk_accuracy)
        }

    def process_stream(self, data_generator) -> Dict[str, List[Any]]:
        """
        Orchestrates an entire data stream sequentially from an iterator or custom data loader.
        
        Args:
            data_generator: An iterable returning chunks of (X_chunk, y_chunk) from your io.py module.
        """
        for chunk_idx, (X_chunk, y_chunk) in enumerate(data_generator):
            # Strict validation checks to satisfy robust API consistency requirements
            if not isinstance(X_chunk, np.ndarray) or not isinstance(y_chunk, np.ndarray):
                raise TypeError("Stream components must strictly emerge as numpy arrays.")
                
            if X_chunk.shape[0] != y_chunk.shape[0]:
                raise ValueError(f"Shape discrepancy at chunk {chunk_idx}: X={X_chunk.shape}, y={y_chunk.shape}")

            # --- Step A: Prequential Evaluation (Predict first) ---
            # If it's the very first chunk and the model is totally uninitialized, 
            # we skip prediction scoring to avoid a RuntimeError.
           # Alternative super-simple fix for line 108:
           # 🌟 FIX: Safely check if the pipeline has an initialized model step
            model_step = self.pipeline.steps[-1][1] if hasattr(self.pipeline, 'steps') else list(self.pipeline.named_steps.values())[-1]

            if chunk_idx > 0 or getattr(model_step, 'is_fitted_', False) or getattr(model_step, 'classes_', np.array([])).size > 0:
                scores = self.score_chunk(X_chunk, y_chunk)
                current_acc = scores["accuracy"]
                cum_acc = scores["cumulative_accuracy"]
            else:
                current_acc = 0.0
                cum_acc = 0.0

            # --- Step B: Progressive Model Adaptation (Train second) ---
            fit_metrics = self.fit_chunk(X_chunk, y_chunk)
            
            # Save metrics to history for tracking over time
            self.history["chunk_id"].append(chunk_idx)
            self.history["chunk_size"].append(X_chunk.shape[0])
            self.history["accuracy"].append(current_acc)
            self.history["cumulative_accuracy"].append(cum_acc)
            self.history["fit_time"].append(fit_metrics.get("fit_time", 0.0))
            self.history["memory_bytes"].append(fit_metrics.get("memory_bytes", 0))
            
            # Print streaming logs for real-time validation in the notebook
            print(f"[Chunk {chunk_idx:02d}] Size: {X_chunk.shape[0]} | "
                  f"Chunk Acc: {current_acc:.4f} | Cum Acc: {cum_acc:.4f} | "
                  f"Fit Time: {fit_metrics.get('fit_time', 0.0):.4f}s")
                  
        return self.history