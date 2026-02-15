import pandas as pd
import numpy as np
import logging
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List, Optional

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
        logging.FileHandler(config.LOGS_DIR / "interpretation.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelInterpreter:
    """
    Interprets the trained model using SHAP values.
    """
    
    def __init__(self):
        self.model = None
        self.explainer = None
        self.shap_values = None
        self.X_sample = None
        self.output_dir = config.OUTPUTS_DIR / "interpretability"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_resources(self):
        """Load model and test data."""
        logger.info("Loading model and data...")
        
        # Load Model
        model_path = config.MODELS_DIR / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError("Model not found. Run training first.")
        self.model = joblib.load(model_path)
        
        # Load Data
        test_path = config.PROCESSED_DATA_DIR / "test_final.csv"
        df = pd.read_csv(test_path)
        
        # Prepare X (drop target/metadata)
        drop_cols = config.ALL_COLS + ['RUL']
        self.X_test = df.drop(columns=drop_cols, errors='ignore')
        
        # Use a sample for SHAP (it can be slow)
        self.X_sample = self.X_test.sample(n=min(500, len(self.X_test)), random_state=config.RANDOM_STATE)
        logger.info(f"Loaded resources. Output directory: {self.output_dir}")

    def calculate_shap(self):
        """Calculate SHAP values."""
        logger.info("Calculating SHAP values...")
        try:
            # TreeExplainer is best for Tree models (RF, XGB, LGBM)
            # For sklearn RF, it might be slower than XGB/LGBM but usually fine for 500 samples
            self.explainer = shap.TreeExplainer(self.model)
            self.shap_values = self.explainer.shap_values(self.X_sample)
            
            logger.info(f"SHAP values type: {type(self.shap_values)}")
            if isinstance(self.shap_values, list):
                logger.info(f"SHAP values list length: {len(self.shap_values)}")
                logger.info(f"SHAP values[0] shape: {self.shap_values[0].shape}")
            elif hasattr(self.shap_values, 'shape'):
                logger.info(f"SHAP values shape: {self.shap_values.shape}")
            
            # For binary classification, shap_values is a list [class0, class1] or array (N, F, 2)
            # We want class 1 (Failure)
            if isinstance(self.shap_values, list):
                self.shap_values = self.shap_values[1]
            elif len(self.shap_values.shape) == 3 and self.shap_values.shape[2] == 2:
                # Array case (N, F, 2) -> Take Class 1
                self.shap_values = self.shap_values[:, :, 1]
                
            logger.info("SHAP values calculated.")
        except Exception as e:
            logger.error(f"Error calculating SHAP values: {e}")
            # Fallback for KernelExplainer if model not supported by TreeExplainer (e.g. VotingClassifier)
            # But we know it's a tree model here.
            raise

    def plot_global_importance(self):
        """Generate summary plot."""
        logger.info("Generating global importance plot...")
        plt.figure(figsize=(10, 8))
        shap.summary_plot(self.shap_values, self.X_sample, show=False)
        plt.tight_layout()
        plt.savefig(self.output_dir / "shap_summary.png")
        plt.close()

    def plot_local_explanations(self, num_samples: int = 5):
        """Generate waterfall/force plots for specific samples."""
        logger.info(f"Generating local explanations for {num_samples} samples...")
        
        # We need the base value (expected value)
        # For binary list, it's explainer.expected_value[1]
        # Helper to ensure we have 2D shap values (Samples, Features)
        if len(self.shap_values.shape) == 3:
             logger.info(f"Slicing 3D interactions/classes in plot: {self.shap_values.shape}")
             # Assumption: (Samples, Features, Classes) -> Take Class 1 (Index 1)
             # But check if last dim is classes
             if self.shap_values.shape[2] == 2:
                 self.shap_values = self.shap_values[:, :, 1]
        
        base_value = self.explainer.expected_value
        if isinstance(base_value, list) or (isinstance(base_value, np.ndarray) and len(base_value.shape)==1 and len(base_value)>1):
             base_value = base_value[1]
        
        # Generate Waterfall plots
        for i in range(num_samples):
            plt.figure()
            # shap.plots.waterfall requires an Explanation object, simpler to use legacy waterfall or create Explanation
            # Create explanation for single sample
            # shap.plots.waterfall expects an Explanation object for a SINGLE observation
            explanation = shap.Explanation(
                values=self.shap_values[i],
                base_values=base_value,
                data=self.X_sample.iloc[i].values,
                feature_names=self.X_sample.columns
            )
            shap.plots.waterfall(explanation, show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"shap_waterfall_sample_{i}.png")
            plt.close()
            
            # Force plot (HTML)
            # shap.save_html(self.output_dir / f"shap_force_sample_{i}.html", 
            #                shap.force_plot(base_value, self.shap_values[i], self.X_sample.iloc[i]))

    def plot_dependence(self, top_n: int = 5):
        """Generate partial dependence plots for top features."""
        logger.info("Generating dependence plots...")
        
        # Get top features by mean abs SHAP
        vals = np.abs(self.shap_values).mean(0)
        feature_importance = pd.DataFrame(list(zip(self.X_sample.columns, vals)), columns=['col_name','feature_importance_vals'])
        feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
        top_features = feature_importance['col_name'].head(top_n).tolist()
        
        for feature in top_features:
            plt.figure(figsize=(8, 6))
            shap.dependence_plot(feature, self.shap_values, self.X_sample, show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir / f"shap_dependence_{feature}.png")
            plt.close()

    def analyze_interactions(self):
        """Analyze feature interactions (optional, expensive)."""
        logger.info("Calculating SHAP interaction values...")
        try:
            # Only do this for a smaller subset as it's O(N^2)
            small_sample = self.X_sample.iloc[:50] 
            shap_interaction_values = self.explainer.shap_interaction_values(small_sample)
            
            if isinstance(shap_interaction_values, list):
                shap_interaction_values = shap_interaction_values[1]
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_interaction_values, small_sample, show=False)
            plt.tight_layout()
            plt.savefig(self.output_dir / "shap_interactions.png")
            plt.close()
        except Exception as e:
            logger.warning(f"Could not calculate interactions: {e}")

    def explain_single_prediction(self, features: pd.Series) -> Dict[str, Any]:
        """
        Explain a single prediction.
        Desired output: Prob, Top 5 features with values and SHAP contribution.
        """
        # Reshape for model
        X_single = features.to_frame().T
        
        # Prediction
        prob = self.model.predict_proba(X_single)[0][1]
        
        # SHAP
        shap_values = self.explainer.shap_values(X_single)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif len(shap_values.shape) == 3 and shap_values.shape[2] == 2:
             shap_values = shap_values[:, :, 1]
        
        # Organize
        explanation = []
        for feat, val, shap_val in zip(self.X_sample.columns, X_single.iloc[0], shap_values[0]):
            explanation.append({
                "feature": feat,
                "value": val,
                "shap": shap_val
            })
            
        # Sort by absolute SHAP impact
        explanation.sort(key=lambda x: abs(x['shap']), reverse=True)
        
        result = {
            "failure_probability": float(prob),
            "top_drivers": explanation[:5]
        }
        return result

    def run(self):
        """Execute interpretation pipeline."""
        try:
            self.load_resources()
            self.calculate_shap()
            self.plot_global_importance()
            self.plot_local_explanations()
            self.plot_dependence()
            self.analyze_interactions()
            
            # Test explanation function
            logger.info("Testing single prediction explanation...")
            sample = self.X_sample.iloc[0]
            explanation = self.explain_single_prediction(sample)
            logger.info(f"Sample Prediction Explanation: {explanation}")
            
            logger.info("Interpretation complete.")
        except Exception as e:
            logger.error(f"Interpretation failed: {e}")
            raise

if __name__ == "__main__":
    intepreter = ModelInterpreter()
    intepreter.run()
