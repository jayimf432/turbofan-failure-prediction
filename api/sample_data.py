import random
import numpy as np
import pandas as pd
from typing import List, Dict, Any
from api.schemas import SensorData

def generate_sample_sensor_data(
    unit_number: int = 1, 
    seq_length: int = 30, 
    failure_scenario: bool = False
) -> List[Dict[str, Any]]:
    """
    Generate synthetic sensor data for testing.
    
    Args:
        unit_number: ID of the engine.
        seq_length: Number of time steps (cycles) to generate.
        failure_scenario: If True, values will drift towards failure thresholds.
        
    Returns:
        List of dictionaries matching SensorData schema.
    """
    data = []
    
    # Baseline values for sensors (approximated from EDA mean)
    baselines = {
        "setting_1": -0.001, "setting_2": 0.001, "setting_3": 100.0,
        "s_1": 518.67, "s_2": 642.00, "s_3": 1585.00, "s_4": 1400.00,
        "s_5": 14.62, "s_6": 21.61, "s_7": 554.00, "s_8": 2388.00,
        "s_9": 9060.00, "s_10": 1.30, "s_11": 47.30, "s_12": 522.00,
        "s_13": 2388.00, "s_14": 8135.00, "s_15": 8.42, "s_16": 0.03,
        "s_17": 392, "s_18": 2388, "s_19": 100.0, "s_20": 39.00,
        "s_21": 23.40
    }
    
    # Drift factors for failure scenario (per cycle)
    drifts = {
        "s_2": 0.05, "s_3": 0.1, "s_4": 0.5, "s_7": -0.05,
        "s_11": 0.02, "s_12": -0.05, "s_15": 0.001, "s_17": 0.1,
        "s_20": -0.01, "s_21": -0.01
    } if failure_scenario else {}
    
    for t in range(1, seq_length + 1):
        row = {
            "unit_number": unit_number,
            "time_in_cycles": t,
            "setting_1": baselines["setting_1"] + random.gauss(0, 0.001),
            "setting_2": baselines["setting_2"] + random.gauss(0, 0.001),
            "setting_3": baselines["setting_3"]
        }
        
        for sensor in [f"s_{i}" for i in range(1, 22)]:
            base = baselines[sensor]
            noise = random.gauss(0, base * 0.005) # 0.5% noise
            drift = drifts.get(sensor, 0) * t
            
            val = base + drift + noise
            row[sensor] = float(val)
            
        data.append(row)
        
    return data

def get_sample_request_payload(unit_id: int = 1, steps: int = 50, failure: bool = False):
    """Return specific payload format for API."""
    raw_data = generate_sample_sensor_data(unit_id, steps, failure)
    return {"data": raw_data}

if __name__ == "__main__":
    # Test generator
    sample = generate_sample_sensor_data(seq_length=5, failure_scenario=True)
    print(f"Generated {len(sample)} rows.")
    print(sample[0])
