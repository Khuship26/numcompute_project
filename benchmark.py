import time
import numpy as np
from numcompute_stream.preprocessing import StandardScaler

def run_benchmark():
    # 1. Generate a large mock data stream (10,000 samples, 4 features)
    print("⏳ Generating 10,000 streaming rows for benchmarking...")
    np.random.seed(42)
    X_stream = np.random.randn(10000, 4)
    chunk_size = 50
    
    scaler_loop = StandardScaler()
    scaler_vector = StandardScaler()
    
    # ============================================================
    # BENCHMARK 1: LOOP-BASED PROCESSING (Row-by-Row)
    # ============================================================
    print("\n🏃 Running Loop-Based Benchmark (Row-by-Row)...")
    start_time = time.perf_counter()
    
    # Process the stream row by row using a standard python loop
    for row in X_stream:
        # Reshape to keep 2D array structure expected by the scaler
        row_2d = row.reshape(1, -1)
        scaler_loop.partial_fit(row_2d)
        _ = scaler_loop.transform(row_2d)
        
    loop_time = time.perf_counter() - start_time
    print(f"⏱️ Loop Execution Time: {loop_time:.4f} seconds")

    # ============================================================
    # BENCHMARK 2: VECTORIZED PROCESSING (Chunk-by-Chunk)
    # ============================================================
    print("\n⚡ Running Vectorized Benchmark (Chunk-by-Chunk)...")
    start_time = time.perf_counter()
    
    # Process the stream in vectorized blocks of 50 using NumPy slicing
    for i in range(0, X_stream.shape[0], chunk_size):
        chunk = X_stream[i:i+chunk_size]
        scaler_vector.partial_fit(chunk)
        _ = scaler_vector.transform(chunk)
        
    vector_time = time.perf_counter() - start_time
    print(f"⏱️ Vectorized Execution Time: {vector_time:.4f} seconds")

    # ============================================================
    # SPEEDUP SUMMARY
    # ============================================================
    print("\n" + "="*50)
    print("🏆 BENCHMARK RESULTS SUMMARY")
    print("="*50)
    speedup = loop_time / vector_time
    print(f"• Loop-based approach: {loop_time:.4f}s")
    print(f"• Vectorized chunk approach: {vector_time:.4f}s")
    print(f"🚀 Vectorization Speedup Factor: {speedup:.2f}x faster")
    print("="*50)

if __name__ == "__main__":
    run_benchmark()