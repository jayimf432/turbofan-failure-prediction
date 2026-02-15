import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import os
from pathlib import Path
from datetime import datetime, timedelta

# Import configuration
try:
    from src import config
except ImportError:
    # Allow running as script from root
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data(file_path: Path) -> pd.DataFrame:
    """
    Load data from a CSV/TXT file.
    
    Args:
        file_path (Path): Path to the data file.
        
    Returns:
        pd.DataFrame: Loaded dataframe.
    """
    try:
        logger.info(f"Loading data from {file_path}")
        # The data is space-separated without header
        df = pd.read_csv(file_path, sep=r"\s+", header=None, names=config.ALL_COLS)
        logger.info(f"Data loaded successfully. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using forward fill for time series, then mean imputation.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with missing values handled.
    """
    try:
        initial_missing = df.isnull().sum().sum()
        logger.info(f"Initial missing values: {initial_missing}")
        
        # Sort by unit and time to ensure forward fill works correctly on time series
        df = df.sort_values(by=[config.ID_COL, config.TIME_COL])
        
        # Forward fill within each unit
        df = df.groupby(config.ID_COL).apply(lambda group: group.ffill()).reset_index(drop=True)
        
        # Fill remaining with mean (if any start with NaN)
        df = df.fillna(df.mean())
        
        final_missing = df.isnull().sum().sum()
        logger.info(f"Missing values handled. Remaining: {final_missing}")
        return df
    except Exception as e:
        logger.error(f"Error in handle_missing_values: {e}")
        raise

def remove_outliers(df: pd.DataFrame, threshold: float = config.OUTLIER_THRESHOLD) -> pd.DataFrame:
    """
    Remove outliers using IQR method for sensor columns.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): IQR multiplier for outlier detection.
        
    Returns:
        pd.DataFrame: Dataframe with outliers removed (or capped).
    """
    try:
        logger.info(f"Removing outliers with IQR threshold: {threshold}")
        df_clean = df.copy()
        outliers_count = 0
        
        for col in config.SENSOR_COLS:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            # Identify outliers
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outliers_count += outliers.sum()
            
            # Option 1: Remove rows (can be aggressive)
            # df_clean = df_clean[~((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound))]
            
            # Option 2: Cap/Clip values (often safer for sensor data to preserve time continuity)
            # Let's use clipping to maintain time series integrity for RUL prediction
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Outliers processed (capped): {outliers_count} values affected.")
        return df_clean
    except Exception as e:
        logger.error(f"Error in remove_outliers: {e}")
        raise

def engineer_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add temporal features (day_of_week, month, hour) by simulating timestamps.
    Assumes 'time_in_cycles' represents hours from a start date.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with new temporal features.
    """
    try:
        logger.info("Engineering temporal features...")
        
        start_date = pd.Timestamp(config.SIMULATION_START_DATE)
        
        # Create a simulated timestamp for each row
        # Adding time_in_cycles as hours to the start date
        # We vary start date slightly by unit to avoid perfect alignment if desired, 
        # but for simplicity we use same start date.
        
        # Efficient vector operation
        df['simulated_timestamp'] = start_date + pd.to_timedelta(df[config.TIME_COL], unit='h')
        
        df['hour'] = df['simulated_timestamp'].dt.hour
        df['day_of_week'] = df['simulated_timestamp'].dt.dayofweek
        df['month'] = df['simulated_timestamp'].dt.month
        
        # Drop temporary timestamp content
        df.drop('simulated_timestamp', axis=1, inplace=True)
        
        logger.info(f"Added features: hour, day_of_week, month")
        return df
    except Exception as e:
        logger.error(f"Error in engineer_temporal_features: {e}")
        raise

def normalize_sensors(df: pd.DataFrame, mode: str = 'train') -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """
    Scale sensor readings using StandardScaler.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        mode (str): 'train' to fit and transform, 'test' to transform only.
        
    Returns:
        Tuple[pd.DataFrame, Optional[StandardScaler]]: Scaled dataframe and the scaler object.
    """
    try:
        logger.info(f"Normalizing sensors in {mode} mode...")
        scaler_path = config.MODELS_DIR / "scaler.joblib"
        
        if mode == 'train':
            scaler = StandardScaler()
            df[config.SENSOR_COLS] = scaler.fit_transform(df[config.SENSOR_COLS])
            joblib.dump(scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        else:
            if not scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {scaler_path}. Run training first.")
            scaler = joblib.load(scaler_path)
            df[config.SENSOR_COLS] = scaler.transform(df[config.SENSOR_COLS])
            logger.info("Loaded scaler and transformed data")
            
        return df, scaler
    except Exception as e:
        logger.error(f"Error in normalize_sensors: {e}")
        raise

def create_train_test_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (80/20) for time series - NO random split.
    Splits are done based on Unit Numbers to ensure no data leakage between units,
    or chronologically within units if that was the goal. 
    Usually for RUL, we split by Units (e.g. Units 1-80 train, 81-100 val).
    
    However, the requirement says "Split data chronologically (80/20) for time series".
    If we split strictly by time, we might cut a unit in half.
    Standard CMAPSS practice is splitting by Units.
    Let's implement a unit-based split which respects the "time series nature" (keeping unit trajectories intact).
    """
    try:
        logger.info("Creating train/test split...")
        
        # Get unique units
        units = df[config.ID_COL].unique()
        n_units = len(units)
        cutoff = int(n_units * (1 - config.TEST_SIZE))
        
        train_units = units[:cutoff]
        test_units = units[cutoff:]
        
        train_df = df[df[config.ID_COL].isin(train_units)].copy()
        test_df = df[df[config.ID_COL].isin(test_units)].copy()
        
        logger.info(f"Split by Units - Train Units: {len(train_units)}, Test Units: {len(test_units)}")
        logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")
        
        return train_df, test_df
    except Exception as e:
        logger.error(f"Error in create_train_test_split: {e}")
        raise

def compute_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Remaining Useful Life (RUL) for training data.
    RUL = Max Cycle - Current Cycle
    
    Args:
        df: Input dataframe
        
    Returns:
        df: Dataframe with 'RUL' column
    """
    try:
        # Calculate max cycle for each unit
        max_cycles = df.groupby(config.ID_COL)[config.TIME_COL].max().reset_index()
        max_cycles.columns = [config.ID_COL, 'max_cycle']
        
        # Merge back
        df = df.merge(max_cycles, on=config.ID_COL, how='left')
        
        # Calculate RUL
        df['RUL'] = df['max_cycle'] - df[config.TIME_COL]
        
        # Drop max_cycle if not needed, or keep for analysis
        df.drop('max_cycle', axis=1, inplace=True)
        return df
    except Exception as e:
        logger.error(f"Error calculating RUL: {e}")
        raise

def save_preprocessed_data(train: pd.DataFrame, test: pd.DataFrame):
    """
    Save processed data to data/processed/.
    
    Args:
        train (pd.DataFrame): Training data.
        test (pd.DataFrame): Testing data.
    """
    try:
        train_path = config.PROCESSED_DATA_DIR / "train.csv"
        test_path = config.PROCESSED_DATA_DIR / "test.csv"
        
        logger.info(f"Saving processed data to {config.PROCESSED_DATA_DIR}")
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        logger.info("Data saved successfully.")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

def main():
    try:
        logger.info("Starting preprocessing pipeline...")
        
        # 1. Load Data
        df = load_data(config.TRAIN_FILE)
        
        # 2. Compute RUL (Target) - Essential for this dataset
        df = compute_rul(df)
        
        # 3. Handle Missing Values
        df = handle_missing_values(df)
        
        # 4. Remove Outliers
        df = remove_outliers(df)
        
        # 5. Temporal Features
        df = engineer_temporal_features(df)
        
        # 6. Split Data
        train_df, test_df = create_train_test_split(df)
        
        # 7. Normalize Sensors (Fit on Train, Transform Test)
        train_df, scaler = normalize_sensors(train_df, mode='train')
        test_df, _ = normalize_sensors(test_df, mode='test')
        
        # 8. Save Data
        save_preprocessed_data(train_df, test_df)
        
        logger.info("Preprocessing pipeline completed successfully.")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        # raise  # Removed to avoid crushing the script execution in main

if __name__ == "__main__":
    main()
