import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Files
TRAIN_FILE = RAW_DATA_DIR / "train_FD001.txt"
TEST_FILE = RAW_DATA_DIR / "test_FD001.txt"
RUL_FILE = RAW_DATA_DIR / "RUL_FD001.txt"

# Data Schema
ID_COL = "unit_number"
TIME_COL = "time_in_cycles"
SETTINGS_COLS = ["setting_1", "setting_2", "setting_3"]
SENSOR_COLS = [f"s_{i}" for i in range(1, 22)]
ALL_COLS = [ID_COL, TIME_COL] + SETTINGS_COLS + SENSOR_COLS

# Preprocessing Parameters
OUTLIER_THRESHOLD = 1.5  # IQR multiplier
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Feature Engineering
# Since CMAPSS is simulation data without real dates, we simulate a start date
SIMULATION_START_DATE = "2024-01-01" 

# Feature Params
FEATURE_WINDOWS = [7, 14, 30]
FEATURE_LAGS = [1, 3, 7, 14]
# Sensors to perform heavy engineering on (high variance/correlation ones)
# Based on EDA, s_2, s_3, s_4, s_7, s_11, s_12, s_14, s_15, s_17, s_20, s_21 show good trends
SENSORS_TO_ENGINEER = [
    "s_2", "s_3", "s_4", "s_7", "s_11", "s_12", "s_14", "s_15", "s_17", "s_20", "s_21"
]

# Feature Selection
FEATURE_SELECTION_PARAMS = {
    "correlation_threshold": 0.95,
    "top_k_features": 50
}
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Model Training
MLFLOW_EXPERIMENT_NAME = "predictive_maintenance"
COST_FALSE_NEGATIVE = 50000
COST_FALSE_POSITIVE = 2000

# Base Model Params (can be overridden by Optuna)
MODEL_PARAMS = {
    "random_forest": {
        "n_estimators": 100,
        "max_depth": 10,
        "random_state": RANDOM_STATE
    },
    "xgboost": {
        "n_estimators": 100,
        "max_depth": 6,
        "learning_rate": 0.1,
        "random_state": RANDOM_STATE
    }
}
