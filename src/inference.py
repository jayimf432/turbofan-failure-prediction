import pandas as pd
import numpy as np
import joblib
import logging
import shap
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

# Import project modules
try:
    from src import config
    from src.feature_engineering import FeatureEngineer
    from api.schemas import SensorData
except ImportError:
    import sys
    sys.path.append(str(Path(__file__).parent.parent))
    from src import config
    from src.feature_engineering import FeatureEngineer
    from api.schemas import SensorData

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferencePipeline:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferencePipeline, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance

    def __init__(self):
        if self.initialized:
            return
            
        self.model = None
        self.scaler = None
        self.explainer = None
        self.feature_names = []
        self.load_artifacts()
        self.initialized = True
        
    def load_artifacts(self):
        """Load model, scaler, and configuration."""
        try:
            logger.info("Loading inference artifacts...")
            
            # Paths
            model_path = config.MODELS_DIR / "best_model.joblib"
            scaler_path = config.MODELS_DIR / "scaler.joblib"
            # We also need the feature list. 
            # Ideally this should be saved. For now, we'll try to deduce or load if available.
            # Best way: Load the training data column names or save them explicitly.
            # Workaround: Re-instantiate FeatureEngineer and run on dummy data? No.
            # Better: Load 'train_final.csv' header if it exists, or rely on model.feature_names_in_
            
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            
            self.model = joblib.load(model_path)
            
            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
            else:
                logger.warning("Scaler not found! Inferences might be incorrect if scaling was used.")
                
            # Get feature names from model if possible
            if hasattr(self.model, "feature_names_in_"):
                self.feature_names = list(self.model.feature_names_in_)
            else:
                # Fallback: Load header from processed data
                train_sample = pd.read_csv(config.PROCESSED_DATA_DIR / "train_final.csv", nrows=0)
                self.feature_names = [c for c in train_sample.columns if c not in ['RUL']]
                
            # Initialize SHAP explainer (TreeExplainer for XGB/RF/LGBM)
            try:
                # Background dataset for SHAP (optional but good for accuracy)
                # self.explainer = shap.TreeExplainer(self.model)
                pass # Initializing on every request or lazily might be better if memory is concern
            except Exception as e:
                logger.warning(f"Could not init SHAP explainer: {e}")

            logger.info("Artifacts loaded successfully.")
            
        except Exception as e:
            logger.error(f"Error loading artifacts: {e}")
            raise

    def preprocess(self, data: List[SensorData]) -> pd.DataFrame:
        """
        Convert API input to DataFrame and apply preprocessing.
        Mimics src/preprocessing.py but for inference stream.
        """
        # 1. Convert to DataFrame
        df = pd.DataFrame([d.dict() for d in data])
        
        # 2. Sort by time (critical for feature engineering)
        df = df.sort_values(by=config.TIME_COL)
        
        # 3. Handle Missing Values (FFILL)
        # In inference, we might not have previous data if not passed.
        df = df.ffill().fillna(0) # Simple fallback
        
        # 4. Temporal Features (Simulate timestamp)
        # Re-implementing logic from src/preprocessing.py
        start_date = pd.Timestamp(config.SIMULATION_START_DATE)
        df['simulated_timestamp'] = start_date + pd.to_timedelta(df[config.TIME_COL], unit='h')
        df['hour'] = df['simulated_timestamp'].dt.hour
        df['day_of_week'] = df['simulated_timestamp'].dt.dayofweek
        df['month'] = df['simulated_timestamp'].dt.month
        df.drop('simulated_timestamp', axis=1, inplace=True)
        
        # 5. Normalize Sensors
        if self.scaler:
            df[config.SENSOR_COLS] = self.scaler.transform(df[config.SENSOR_COLS])
            
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply feature engineering. 
        Adapted to be robust for short history (min_periods=1).
        """
        # Reuse logic but with adjusted parameters for inference if needed
        # We will manually apply the transformations found in src/feature_engineering.py
        # but overriding rolling to allow min_periods=1
        
        df_eng = df.copy()
        sensors = config.SENSORS_TO_ENGINEER
        windows = config.FEATURE_WINDOWS
        lags = config.FEATURE_LAGS
        
        # Rolling
        for w in windows:
            # min_periods=1 allows output even if we have fewer than w samples
            rolled = df_eng[sensors].rolling(window=w, min_periods=1)
            
            df_eng = pd.concat([df_eng, 
                rolled.mean().rename(columns=lambda c: f"{c}_rolling_mean_{w}"),
                rolled.std().fillna(0).rename(columns=lambda c: f"{c}_rolling_std_{w}"), # std is NaN for 1 sample
                rolled.min().rename(columns=lambda c: f"{c}_rolling_min_{w}"),
                rolled.max().rename(columns=lambda c: f"{c}_rolling_max_{w}"),
                df_eng[sensors].ewm(span=w, min_periods=1).mean().rename(columns=lambda c: f"{c}_ewma_{w}")
            ], axis=1)

        # Lags
        for lag in lags:
            # Shift will produce NaNs at start. Fill with first value or 0?
            # Backfill is better for inference 'cold start'
            shifted = df_eng[sensors].shift(lag).bfill().fillna(0)
            shifted.rename(columns=lambda c: f"{c}_lag_{lag}", inplace=True)
            df_eng = pd.concat([df_eng, shifted], axis=1)
            
        # Stats
        delta = df_eng[sensors].diff().fillna(0).rename(columns=lambda c: f"{c}_delta")
        cumsum = df_eng[sensors].cumsum().rename(columns=lambda c: f"{c}_cumsum")
        
        # Skew/Kurt requires > 3 samples. 
        rolled30 = df_eng[sensors].rolling(window=30, min_periods=1)
        skew = rolled30.skew().fillna(0).rename(columns=lambda c: f"{c}_skew_30")
        kurt = rolled30.kurt().fillna(0).rename(columns=lambda c: f"{c}_kurt_30")
        
        df_eng = pd.concat([df_eng, delta, cumsum, skew, kurt], axis=1)
        
        # Domain
        # We need the iso_forest from training to calculate anomaly_score accurately.
        # If not saved, we skip or use a placeholder.
        # Check if we can save/load it. For now, skipping anomaly_score or setting 0.
        # Wait, the model might include 'anomaly_score' as a feature!
        # If the model expects it, we MUST provide it.
        # Checking feature_names...
        if 'anomaly_score' in self.feature_names:
             # Ideally load the fitted iso_forest. If not valid, set to 0.
             df_eng['anomaly_score'] = 0.0 # Placeholder
             
        # Interactions
        if 'temp_vib_interaction' in self.feature_names:
             df_eng['temp_vib_interaction'] = df_eng['s_2'] * df_eng['s_11']
        if 'pressure_ratio' in self.feature_names:
            df_eng['pressure_ratio'] = df_eng['s_2'] / (df_eng['s_21'] + 1e-6)
            
        # Trends
        w_trend = 14
        trend = (df_eng[sensors] - df_eng[sensors].shift(w_trend).bfill()).divide(w_trend)
        trend.rename(columns=lambda c: f"{c}_trend_{w_trend}", inplace=True)
        df_eng = pd.concat([df_eng, trend], axis=1)
        
        return df_eng

    def predict(self, data: List[SensorData]) -> Dict[str, Any]:
        """
        End-to-end prediction pipeline.
        Returns prediction for the LAST cycle in the input data.
        """
        # 1. Preprocess
        df = self.preprocess(data)
        
        # 2. Engineer Features
        df_eng = self.engineer_features(df)
        
        # 3. Select Features for Model
        # Ensure we have all columns expected by model, in correct order
        # Fill missing with 0
        current_data = df_eng.iloc[[-1]].copy() # Take last row
        
        # Align columns
        model_input = pd.DataFrame(index=current_data.index)
        for col in self.feature_names:
            if col in current_data.columns:
                model_input[col] = current_data[col]
            else:
                model_input[col] = 0.0
                
        # 4. Predict
        prob = self.model.predict_proba(model_input)[0][1]
        prediction = int(prob > 0.5) # Threshold usually 0.5 for balanced/tuned model
        
        # Risk Level
        if prob < 0.3:
            risk = "Low"
        elif prob < 0.7:
            risk = "Medium"
        else:
            risk = "High"
            
        # 5. Explain (Optional)
        factors = []
        try:
            if not self.explainer:
                self.explainer = shap.TreeExplainer(self.model)
            
            shap_values = self.explainer.shap_values(model_input)
            
            # Handling SHAP output shape
            if isinstance(shap_values, list):
                sv = shap_values[1][0]
            elif len(shap_values.shape) == 3:
                sv = shap_values[0, :, 1]
            else:
                sv = shap_values[0]
                
            # Get top features
            feature_impact = list(zip(self.feature_names, sv))
            feature_impact.sort(key=lambda x: abs(x[1]), reverse=True)
            
            for f, v in feature_impact[:3]:
                factors.append({"feature": f, "impact": float(v)})
                
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")
            
        return {
            "unit_number": data[-1].unit_number,
            "time_in_cycles": data[-1].time_in_cycles,
            "failure_probability": float(prob),
            "prediction": prediction,
            "risk_level": risk,
            "contributing_factors": factors,
            "warning": "Short history" if len(data) < 30 else None
        }

# Singleton instance
pipeline = InferencePipeline()
