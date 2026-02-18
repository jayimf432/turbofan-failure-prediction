import requests
from typing import List, Dict, Any, Optional
import logging
from api.sample_data import generate_sample_sensor_data

logger = logging.getLogger("TurbofanClient")
logging.basicConfig(level=logging.INFO)

class TurbofanClient:
    """
    Python client for Turbofan Failure Prediction API.
    """
    
    def __init__(self, base_url: str = "http://127.0.0.1:8000"):
        self.base_url = base_url.rstrip("/")
        
    def health_check(self) -> Dict[str, Any]:
        """Check API status."""
        try:
            resp = requests.get(f"{self.base_url}/health")
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unreachable", "error": str(e)}
            
    def predict(self, sensor_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Send sensor data for prediction.
        
        Args:
            sensor_data: List of dictionaries matching SensorData schema.
        """
        url = f"{self.base_url}/predict"
        payload = {"data": sensor_data}
        
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.HTTPError as e:
            logger.error(f"Prediction error: {resp.text}")
            raise
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise

    def predict_batch(self, units_data: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Batch prediction for multiple units.
        """
        url = f"{self.base_url}/predict_batch"
        payload = {"units": [{"data": d} for d in units_data]}
        
        try:
            resp = requests.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()["predictions"]
        except Exception as e:
            logger.error(f"Batch prediction error: {e}")
            raise

    def get_feature_importance(self) -> List[Dict[str, float]]:
        """Get global feature importance."""
        try:
            resp = requests.get(f"{self.base_url}/feature_importance")
            resp.raise_for_status()
            return resp.json()["features"]
        except Exception as e:
            logger.error(f"Error fetching importance: {e}")
            return []

if __name__ == "__main__":
    # Example Usage
    client = TurbofanClient()
    
    print("\n--- Health Check ---")
    print(client.health_check())
    
    print("\n--- Single Prediction (Normal) ---")
    dummy_data = generate_sample_sensor_data(unit_number=5, seq_length=30)
    try:
        res = client.predict(dummy_data)
        print(f"Prediction for Unit 5: {res['risk_level']} (Prob: {res['failure_probability']:.2f})")
    except:
        print("Failed.")

    print("\n--- Single Prediction (Failure) ---")
    fail_data = generate_sample_sensor_data(unit_number=99, seq_length=50, failure_scenario=True)
    try:
        res = client.predict(fail_data)
        print(f"Prediction for Unit 99: {res['risk_level']} (Prob: {res['failure_probability']:.2f})")
    except:
        print("Failed.")
