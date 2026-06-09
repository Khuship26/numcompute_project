import matplotlib.pyplot as plt
import numpy as np
from typing import List, Union, Dict

def plot_metric_over_time(
    metric_values: Union[List[float], np.ndarray], 
    title: str = "Streaming Model Performance", 
    ylabel: str = "Accuracy"
) -> plt.Figure:
    """
    Plots the structural tracking of an independent pipeline metric across a sequence of streaming chunks.
    
    Args:
        metric_values: Ordered metrics matching chronological stream sequences.
        title: Custom plot heading descriptor.
        ylabel: The title descriptor given to the y-axis coordinate frame.
    """
    chunks = np.arange(len(metric_values))
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
    # Render main performance trajectory curve
    ax.plot(chunks, metric_values, marker='o', linestyle='-', color='#1f77b4', linewidth=2, label=ylabel)
    
    # Calculate and project global rolling baseline standard trend
    if len(metric_values) > 0:
        mean_val = np.mean(metric_values)
        ax.axhline(mean_val, color='r', linestyle='--', alpha=0.7, label=f"Mean: {mean_val:.4f}")
        
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel("Streaming Data Chunk Sequence Index", fontsize=11, labelpad=10)
    ax.set_ylabel(ylabel, fontsize=11, labelpad=10)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="best", frameon=True, facecolor="white", edgecolor="none")
    
    # Adjust outer parameters to clear whitespace
    plt.tight_layout()
    return fig

def compare_models(
    metric1: Union[List[float], np.ndarray], 
    metric2: Union[List[float], np.ndarray], 
    labels: List[str] = ["Base Tree", "Ensemble Bagging"],
    metric_name: str = "Cumulative Accuracy"
) -> plt.Figure:
    """
    Generates a clear comparison plot plotting two streaming architectures 
    head-to-head across matching tracking intervals.
    
    Essential for visualizing the stability benefits of your Ensemble module.
    """
    chunks1 = np.arange(len(metric1))
    chunks2 = np.arange(len(metric2))
    
    fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
    # Plot tracking path for Model 1
    ax.plot(chunks1, metric1, marker='s', linestyle='-', color='#2ca02c', linewidth=2, label=labels[0])
    # Plot tracking path for Model 2
    ax.plot(chunks2, metric2, marker='^', linestyle='-', color='#ff7f0e', linewidth=2, label=labels[1])
    
    ax.set_title(f"Streaming Architecture Benchmark: {labels[0]} vs {labels[1]}", fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel("Streaming Data Chunk Sequence Index", fontsize=11)
    ax.set_ylabel(metric_name, fontsize=11)
    
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc="lower right", frameon=True, shadow=False)
    
    plt.tight_layout()
    return fig

def plot_predictions_vs_ground_truth(y_true: np.ndarray, y_pred: np.ndarray) -> plt.Figure:
    """
    Renders a structural evaluation overview comparing true labels directly to predicted categories.
    Implements a custom confusion matrix representation using pure NumPy and Matplotlib.
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("Matrix dimension matching conflict identified.")
        
    # Dynamically extract labels across categories
    unique_labels = np.unique(np.concatenate([y_true, y_pred]))
    n_classes = len(unique_labels)
    
    # Calculate confusion allocations completely manually
    matrix = np.zeros((n_classes, n_classes), dtype=int)
    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    
    for t, p in zip(y_true, y_pred):
        matrix[label_to_idx[t], label_to_idx[p]] += 1
        
    fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
    cax = ax.imshow(matrix, cmap=plt.cm.Blues, interpolation='nearest')
    fig.colorbar(cax)
    
    # Annotate every block inside the confusion matrix
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j, i, str(matrix[i, j]),
                    ha="center", va="center",
                    color="white" if matrix[i, j] > (matrix.max() / 2) else "black",
                    fontweight='bold')
                    
    ax.set_title("Cumulative Stream Confusion Matrix", fontsize=12, fontweight='bold', pad=12)
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(unique_labels)
    ax.set_yticklabels(unique_labels)
    
    ax.set_xlabel("Predicted Label Target Class", fontsize=10, labelpad=8)
    ax.set_ylabel("True Ground Truth Label Class", fontsize=10, labelpad=8)
    
    plt.tight_layout()
    return fig