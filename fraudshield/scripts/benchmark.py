"""
FraudShield Load Test

Locust-based load testing for the scoring API.

Usage:
  1. Start the API: uvicorn src.api:app --workers 4
  2. Run load test: locust -f scripts/benchmark.py --host=http://localhost:8000
  3. Open http://localhost:8089 to configure and run test

Or run headless:
  locust -f scripts/benchmark.py --host=http://localhost:8000 \
    --users 50 --spawn-rate 10 --run-time 60s --headless
"""
import random
from locust import HttpUser, task, between


# Sample transaction templates
MERCHANT_CATEGORIES = [
    "retail", "grocery", "restaurant", "travel", "entertainment",
    "utilities", "healthcare", "gas_station", "online", "other"
]


def generate_transaction():
    """Generate a random transaction payload."""
    return {
        "amount": round(random.lognormvariate(4, 1.2), 2),
        "merchant_category": random.choice(MERCHANT_CATEGORIES),
        "hour": random.randint(0, 23),
        "day_of_week": random.randint(0, 6),
        "is_international": random.random() < 0.1,
        "card_present": random.random() < 0.7,
        "merchant_risk_score": round(random.betavariate(2, 5), 3)
    }


class FraudScoringUser(HttpUser):
    """Simulates a user making fraud scoring requests."""
    
    wait_time = between(0.01, 0.05)  # 10-50ms between requests
    
    @task(10)
    def score_single(self):
        """Score a single transaction (most common operation)."""
        transaction = generate_transaction()
        self.client.post("/score", json=transaction)
    
    @task(1)
    def score_batch(self):
        """Score a batch of transactions."""
        transactions = [generate_transaction() for _ in range(10)]
        self.client.post("/score/batch", json={"transactions": transactions})
    
    @task(1)
    def health_check(self):
        """Check API health."""
        self.client.get("/health")


class HighVolumeUser(HttpUser):
    """High-volume user for stress testing."""
    
    wait_time = between(0.001, 0.01)  # Near-instant requests
    
    @task
    def score_rapid(self):
        """Rapid-fire single transaction scoring."""
        transaction = generate_transaction()
        self.client.post("/score", json=transaction)


# Quick benchmark function for direct execution
def run_quick_benchmark(host: str = "http://localhost:8000", n_requests: int = 1000):
    """
    Run a quick benchmark without Locust UI.
    
    Usage:
        python scripts/benchmark.py
    """
    import time
    import statistics
    import httpx
    
    print(f"Running benchmark against {host}")
    print(f"Requests: {n_requests}")
    print("-" * 40)
    
    latencies = []
    errors = 0
    
    with httpx.Client(base_url=host, timeout=10.0) as client:
        # Warmup
        for _ in range(10):
            client.post("/score", json=generate_transaction())
        
        # Benchmark
        start = time.perf_counter()
        for i in range(n_requests):
            req_start = time.perf_counter()
            try:
                response = client.post("/score", json=generate_transaction())
                if response.status_code == 200:
                    latencies.append((time.perf_counter() - req_start) * 1000)
                else:
                    errors += 1
            except Exception:
                errors += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Completed {i + 1}/{n_requests}")
        
        total_time = time.perf_counter() - start
    
    # Calculate stats
    if latencies:
        latencies.sort()
        print("\n" + "=" * 40)
        print("RESULTS")
        print("=" * 40)
        print(f"Total time:     {total_time:.2f}s")
        print(f"Requests:       {n_requests}")
        print(f"Errors:         {errors}")
        print(f"Throughput:     {n_requests / total_time:.1f} req/s")
        print(f"\nLatency (ms):")
        print(f"  Mean:         {statistics.mean(latencies):.2f}")
        print(f"  Median (p50): {statistics.median(latencies):.2f}")
        print(f"  p95:          {latencies[int(len(latencies) * 0.95)]:.2f}")
        print(f"  p99:          {latencies[int(len(latencies) * 0.99)]:.2f}")
        print(f"  Max:          {max(latencies):.2f}")
    else:
        print("No successful requests!")


if __name__ == "__main__":
    import sys
    
    host = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    n_requests = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    
    run_quick_benchmark(host, n_requests)
