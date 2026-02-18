import time
import concurrent.futures
import numpy as np
import requests
from api.sample_data import generate_sample_sensor_data
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger("LoadTest")

URL = "http://127.0.0.1:8000/predict"
CONCURRENT_REQUESTS = 10
TOTAL_REQUESTS = 100

def send_request(req_id):
    start = time.time()
    # Generate random data size to vary load slightly
    length = np.random.randint(30, 60)
    data = generate_sample_sensor_data(unit_number=req_id, seq_length=length)
    
    try:
        resp = requests.post(URL, json={"data": data}, timeout=5)
        latency = (time.time() - start) * 1000 # ms
        return latency, resp.status_code
    except Exception as e:
        return None, str(e)

def run_load_test():
    logger.info(f"Starting Load Test: {TOTAL_REQUESTS} requests with {CONCURRENT_REQUESTS} concurrency...")
    
    latencies = []
    errors = 0
    
    start_time = time.time()
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENT_REQUESTS) as executor:
        futures = [executor.submit(send_request, i) for i in range(TOTAL_REQUESTS)]
        
        for future in concurrent.futures.as_completed(futures):
            lat, status = future.result()
            if lat is not None and status == 200:
                latencies.append(lat)
            else:
                errors += 1
                
    total_duration = time.time() - start_time
    
    # Report
    latencies = np.array(latencies)
    logger.info(f"\n--- Load Test Report ---")
    logger.info(f"Total Requests: {TOTAL_REQUESTS}")
    logger.info(f"Successful: {len(latencies)}")
    logger.info(f"Failed: {errors}")
    logger.info(f"Total Time: {total_duration:.2f}s")
    logger.info(f"RPS: {TOTAL_REQUESTS / total_duration:.2f}")
    
    if len(latencies) > 0:
        logger.info(f"Avg Latency: {np.mean(latencies):.2f} ms")
        logger.info(f"P95 Latency: {np.percentile(latencies, 95):.2f} ms")
        logger.info(f"P99 Latency: {np.percentile(latencies, 99):.2f} ms")
        logger.info(f"Min/Max: {np.min(latencies):.2f} / {np.max(latencies):.2f} ms")

if __name__ == "__main__":
    # Check if API is alive first
    try:
        requests.get("http://127.0.0.1:8000/health")
        run_load_test()
    except Exception:
        logger.error("API is unreachable. Is uvicorn running?")
