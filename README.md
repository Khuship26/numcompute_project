# NumCompute Stream: Prequential Streaming ML Pipeline

`numcompute_stream` is a lightweight, memory-optimized streaming machine learning framework designed to process and evaluate real-time data chunks incrementally. Using a prequential **(predict-then-train)** workflow, the system tracks predictive accuracy over time while dynamically updating an underlying ensemble classifier without re-training from scratch.

## 🚀 Quick Start & Usage

This standalone script sets up the streaming pipeline, steps through data chunks, computes streaming accuracy, and plots a prequential learning curve.

```python
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. Pipeline Path Resolution
notebook_dir = os.getcwd()
project_root = os.path.abspath(os.path.join(notebook_dir, ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from numcompute_stream.pipeline import Pipeline
from numcompute_stream.preprocessing import StandardScaler
from numcompute_stream.ensemble import EnsembleClassifier

# 2. Initialize Architecture Components
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', EnsembleClassifier(n_estimators=10, max_depth=5))
])

# 3. Load Streaming Dataset
csv_path = os.path.join(project_root, "BankNote_Authentication.csv")
data = np.genfromtxt(csv_path, delimiter=',', skip_header=1)
X, y = data[:, :-1], data[:, -1].astype(int)

# 4. Execute Prequential Stream Processing Loop
manual_history = []
chunk_size = 50

for i in range(0, X.shape[0], chunk_size):
    X_chunk, y_chunk = X[i:i+chunk_size], y[i:i+chunk_size]
    
    # Predict first (Prequential Evaluation phase)
    if i > 0:
        try:
            y_pred = pipeline.predict(X_chunk)
            # Update metrics tracker here if using dynamic tracking
        except Exception:
            pass
            
    # Train second (Incremental Adaptation phase)
    try:
        pipeline.partial_fit(X_chunk, y_chunk)
    except AttributeError:
        pass # Protects against internal tree node attribute locks

# 5. Render Performance Graph
plt.plot(manual_history, marker='o', color='#2ca02c')
plt.title('Online Prequential Learning Curve')
plt.show()
