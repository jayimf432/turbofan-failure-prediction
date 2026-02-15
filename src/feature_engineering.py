import pandas as pd
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from typing import List, Dict, Optional
from pathlib import Path

# Import configuration
try:
    from src import config
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOGS_DIR / "feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    """
    Feature Engineering pipeline for predictive maintenance.
    """
    
    def __init__(self):
        self.feature_names: List[str] = []
        self.iso_forest = IsolationForest(contamination=0.05, random_state=config.RANDOM_STATE)
        
    def fit(self, df: pd.DataFrame, y: Optional[pd.Series] = None):
        """
        Fit stateful transformations (Isolation Forest).
        
        Args:
            df (pd.DataFrame): Training data.
            y (pd.Series, optional): Target variable (not used for unsupervised features).
        """
        logger.info("Fitting FeatureEngineer...")
        # Fit Isolation Forest on sensor data
        self.iso_forest.fit(df[config.SENSOR_COLS])
        logger.info("Isolation Forest fitted.")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply all feature transformations.
        
        Args:
            df (pd.DataFrame): Input dataframe.
            
        Returns:
            pd.DataFrame: Dataframe with new features.
        """
        try:
            logger.info("Transforming data...")
            df_eng = df.copy()
            
            # 1. Rolling Features
            df_eng = self._rolling_features(df_eng)
            
            # 2. Lag Features
            df_eng = self._lag_features(df_eng)
            
            # 3. Statistical Features
            df_eng = self._stat_features(df_eng)
            
            # 4. Domain / Trend Features
            df_eng = self._domain_features(df_eng)
            df_eng = self._trend_features(df_eng)
            
            # Handle NaNs created by lags/rolling
            # Drop rows with NaNs at the beginning of each unit's life
            # Or backfill if preserving all rows is critical (usually dropping is safer for training)
            df_eng = df_eng.dropna()
            
            # Save feature names
            self.feature_names = [c for c in df_eng.columns if c not in config.ALL_COLS + ['RUL']]
            logger.info(f"Feature engineering complete. New shape: {df_eng.shape}")
            logger.info(f"Generated {len(self.feature_names)} new features.")
            
            return df_eng
        except Exception as e:
            logger.error(f"Error in transform: {e}")
            raise

    def _rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rolling mean, std, min, max, ewma."""
        logger.info("Generating rolling features...")
        
        # We only apply expensive rolling ops on selected sensors to save time/memory
        sensors = config.SENSORS_TO_ENGINEER
        windows = config.FEATURE_WINDOWS
        
        for w in windows:
            # Group by unit to apply rolling only within same unit
            rolled = df.groupby(config.ID_COL)[sensors].rolling(window=w)
            
            # Mean
            res = rolled.mean().reset_index(level=0, drop=True)
            res.columns = [f"{c}_rolling_mean_{w}" for c in sensors]
            df = pd.concat([df, res], axis=1)
            
            # Std
            res = rolled.std().reset_index(level=0, drop=True)
            res.columns = [f"{c}_rolling_std_{w}" for c in sensors]
            df = pd.concat([df, res], axis=1)
            
            # Min
            res = rolled.min().reset_index(level=0, drop=True)
            res.columns = [f"{c}_rolling_min_{w}" for c in sensors]
            df = pd.concat([df, res], axis=1)
            
            # Max
            res = rolled.max().reset_index(level=0, drop=True)
            res.columns = [f"{c}_rolling_max_{w}" for c in sensors]
            df = pd.concat([df, res], axis=1)
            
            # EWMA (Exponential Weighted Moving Average)
            # Groupby apply is slow, doing it per unit might be slow. 
            # Optimization: since data is sorted by unit, we can just apply ewm and reset at boundaries?
            # Actually proper way:
            ewma = df.groupby(config.ID_COL)[sensors].apply(lambda x: x.ewm(span=w).mean()).reset_index(level=0, drop=True)
            ewma.columns = [f"{c}_ewma_{w}" for c in sensors]
            df = pd.concat([df, ewma], axis=1)
            
        return df

    def _lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate lag features."""
        logger.info("Generating lag features...")
        sensors = config.SENSORS_TO_ENGINEER
        lags = config.FEATURE_LAGS
        
        for lag in lags:
            shifted = df.groupby(config.ID_COL)[sensors].shift(lag)
            shifted.columns = [f"{c}_lag_{lag}" for c in sensors]
            df = pd.concat([df, shifted], axis=1)
            
        return df

    def _stat_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate rate of change, cumulative sum, skew, kurtosis."""
        logger.info("Generating statistical features...")
        sensors = config.SENSORS_TO_ENGINEER
        
        # Rate of Change (Delta)
        # diff(1) is change from previous cycle
        delta = df.groupby(config.ID_COL)[sensors].diff()
        delta.columns = [f"{c}_delta" for c in sensors]
        df = pd.concat([df, delta], axis=1)
        
        # Cumulative Sum
        cumsum = df.groupby(config.ID_COL)[sensors].cumsum()
        cumsum.columns = [f"{c}_cumsum" for c in sensors]
        df = pd.concat([df, cumsum], axis=1)
        
        # Skew & Kurtosis (Rolling 30)
        window = 30
        rolled = df.groupby(config.ID_COL)[sensors].rolling(window=window)
        
        skew = rolled.skew().reset_index(level=0, drop=True)
        skew.columns = [f"{c}_skew_{window}" for c in sensors]
        df = pd.concat([df, skew], axis=1)
        
        kurt = rolled.kurt().reset_index(level=0, drop=True)
        kurt.columns = [f"{c}_kurt_{window}" for c in sensors]
        df = pd.concat([df, kurt], axis=1)
        
        return df

    def _domain_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate domain-specific features."""
        logger.info("Generating domain features...")
        
        # 1. Anomaly Score
        df['anomaly_score'] = self.iso_forest.decision_function(df[config.SENSOR_COLS])
        
        # 2. Key Interactions
        # Temperature (s_2) vs Pressure (s_21 proposed/example)
        # s_2, s_3, s_4, ... check high correlation pairs or physical meaning
        # Creating a few generic interaction terms
        df['temp_vib_interaction'] = df['s_2'] * df['s_11'] # Inlet Temp * Vibration (generic example)
        df['pressure_ratio'] = df['s_2'] / (df['s_21'] + 1e-6)
        
        # 3. Cumulative operating hours (already implicitly time_in_cycles? Yes, but maybe scaled)
        # Just keeping time_in_cycles as a feature
        
        return df

    def _trend_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate slope/trend features."""
        logger.info("Generating trend features...")
        sensors = config.SENSORS_TO_ENGINEER
        window = 14
        
        # Calculating linear slope over window
        # Slope = Cov(x,y) / Var(x)
        # Since x is just 0..window-1, Var(x) is constant.
        # We can implement a simple slope estimation: (mean(last_half) - mean(first_half))
        # Or use polyfit on rolling window (very slow).
        # Optimization: Simple trend = (y_t - y_{t-window}) / window
        
        trend = (df[sensors] - df.groupby(config.ID_COL)[sensors].shift(window)) / window
        trend.columns = [f"{c}_trend_{window}" for c in sensors]
        df = pd.concat([df, trend], axis=1)
        
        return df

    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'RUL') -> pd.DataFrame:
        """Calculate feature importance using Random Forest."""
        logger.info("Calculating feature importance...")
        
        # Use a sample if data is too large
        sample_size = min(10000, len(df))
        df_sample = df.sample(sample_size, random_state=config.RANDOM_STATE)
        
        X = df_sample[self.feature_names]
        if target_col not in df_sample.columns:
             # Try falling back to config provided or RUL
             y = df_sample['RUL']
        else:
             y = df_sample[target_col]
             
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=config.RANDOM_STATE, n_jobs=-1)
        rf.fit(X, y)
        
        importances = pd.DataFrame({
            'feature': X.columns,
            'importance': rf.feature_importances_
        }).sort_values(by='importance', ascending=False)
        
        logger.info("Top 10 features:")
        logger.info(importances.head(10))
        
        return importances

def main():
    try:
        logger.info("Starting Feature Engineering...")
        
        # Load processed data
        train_path = config.PROCESSED_DATA_DIR / "train.csv"
        test_path = config.PROCESSED_DATA_DIR / "test.csv"
        
        if not train_path.exists():
            raise FileNotFoundError("Processed data not found. Run preprocessing first.")
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Initialize and run
        fe = FeatureEngineer()
        
        # Fit on train (Iso Forest)
        fe.fit(train_df)
        
        # Transform
        train_eng = fe.transform(train_df)
        test_eng = fe.transform(test_df)
        
        # Calculate Importance
        fe.get_feature_importance(train_eng)
        
        # Save
        train_save_path = config.PROCESSED_DATA_DIR / "train_engineered.csv"
        test_save_path = config.PROCESSED_DATA_DIR / "test_engineered.csv"
        
        train_eng.to_csv(train_save_path, index=False)
        test_eng.to_csv(test_save_path, index=False)
        
        logger.info(f"Engineered features saved to {config.PROCESSED_DATA_DIR}")
        
    except Exception as e:
        logger.error(f"Feature Engineering failed: {e}")
        # raise

if __name__ == "__main__":
    main()
