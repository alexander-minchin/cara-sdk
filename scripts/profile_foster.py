import numpy as np
import cara_py
import time

def main():
    print("--- Optimized Profiling Foster Conjunctions ---")
    
    r1 = np.array([7000.0, 0.0, 0.0])
    v1 = np.array([0.0, 7.5, 0.0])
    cov1 = np.diag([0.01, 0.01, 0.01])
    
    r2 = np.array([7000.0, 0.001, 0.0])
    v2 = np.array([0.0, 0.0, 7.5])
    cov2 = np.diag([0.01, 0.01, 0.01])
    
    hbr = 0.01
    
    # Run a large fixed number of iterations to amortize timing overhead
    iterations = 500_000
    
    # Warm-up
    for _ in range(1000):
        _ = cara_py.compute_2d_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    
    start_time = time.perf_counter()
    for _ in range(iterations):
        _ = cara_py.compute_2d_foster(r1, v1, cov1, r2, v2, cov2, hbr)
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    rate = iterations / duration
    ten_sec_count = rate * 10.0
    
    print(f"Total Iterations: {iterations:,}")
    print(f"Total Duration:   {duration:.4f} s")
    print(f"Throughput:       {rate:,.2f} conj/s")
    print(f"Average Latency:  {1000000/rate:.2f} µs per conjunction")
    print(f"\nEstimated Conjunctions in 10s: {int(ten_sec_count):,}")

if __name__ == "__main__":
    main()
