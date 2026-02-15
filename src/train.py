import pandas as pd
import numpy as np
import logging
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import optuna
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from datetime import datetime

from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score, 
    confusion_matrix, classification_report, average_precision_score,
    roc_curve, precision_recall_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

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
        logging.FileHandler(config.LOGS_DIR / "training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    """
    Pipeline for training, tuning, and evaluating predictive maintenance models.
    """
    
    def __init__(self, experiment_name: str = config.MLFLOW_EXPERIMENT_NAME):
        self.experiment_name = experiment_name
        mlflow.set_experiment(experiment_name)
        self.best_model = None
        self.best_score = -1
        self.best_model_name = ""
        
        # Binary target: 1 if RUL <= 30 (failure imminent), 0 otherwise
        # Just defining a threshold for classification logic
        self.rul_threshold = 30 
        
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Load data and create binary target (Failure within 30 cycles).
        """
        logger.info("Loading data for training...")
        
        # Load final selected features
        train_path = config.PROCESSED_DATA_DIR / "train_final.csv"
        test_path = config.PROCESSED_DATA_DIR / "test_final.csv"
        
        if not train_path.exists():
            raise FileNotFoundError("Train data not found. Run feature selection first.")
            
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        
        # Create Binary Target
        # "Failure" = RUL <= 30
        y_train = (train_df['RUL'] <= self.rul_threshold).astype(int)
        y_test = (test_df['RUL'] <= self.rul_threshold).astype(int)
        
        # Drop non-feature columns
        drop_cols = config.ALL_COLS + ['RUL']
        X_train = train_df.drop(columns=drop_cols, errors='ignore')
        X_test = test_df.drop(columns=drop_cols, errors='ignore')
        
        logger.info(f"Train Shape: {X_train.shape}, Test Shape: {X_test.shape}")
        logger.info(f"Class Balance Train: {y_train.value_counts(normalize=True).to_dict()}")
        
        return X_train, y_train, X_test, y_test

    def train_baseline(self, X_train, y_train, X_test, y_test):
        """Train baseline models."""
        logger.info("Training baseline models...")
        
        models = {
            "LogisticRegression": LogisticRegression(class_weight='balanced', max_iter=1000, random_state=config.RANDOM_STATE),
            "RandomForest": RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=config.RANDOM_STATE)
        }
        
        for name, model in models.items():
            with mlflow.start_run(run_name=f"Baseline_{name}"):
                logger.info(f"Training {name}...")
                model.fit(X_train, y_train)
                
                # Evaluate
                metrics = self.evaluate_model(model, X_test, y_test, name)
                
                # Log to MLflow
                mlflow.log_params(model.get_params())
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(model, name)
                
                # Save as best?
                if metrics['f1'] > self.best_score:
                    self.best_score = metrics['f1']
                    self.best_model = model
                    self.best_model_name = name

    def optimize_xgboost(self, X_train, y_train, X_test, y_test, n_trials=20):
        """Optimize XGBoost using Optuna."""
        logger.info("Optimizing XGBoost...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'gamma': trial.suggest_float('gamma', 0, 5),
                'scale_pos_weight': trial.suggest_float('scale_pos_weight', 1, 10), # Handle imbalance
                'n_jobs': -1,
                'random_state': config.RANDOM_STATE
            }
            
            clf = xgb.XGBClassifier(**params)
            
            # Use TimeSeriesSplit for CV to respect temporal order?
            # Or StratifiedKFold since data is already shuffled/aggregated?
            # Standard CMAPSS is usually split by units, but here we just have rows.
            # Using StratifiedKFold for validatior proxy.
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best XGBoost Params: {study.best_params}")
        
        # Train final model
        with mlflow.start_run(run_name="Best_XGBoost"):
            best_params = study.best_params
            best_params['random_state'] = config.RANDOM_STATE
            model = xgb.XGBClassifier(**best_params)
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model(model, X_test, y_test, "XGBoost_Optuna")
            
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.xgboost.log_model(model, "model")
            
            if metrics['f1'] > self.best_score:
                self.best_score = metrics['f1']
                self.best_model = model
                self.best_model_name = "XGBoost_Optuna"

    def optimize_lightgbm(self, X_train, y_train, X_test, y_test, n_trials=20):
        """Optimize LightGBM using Optuna."""
        logger.info("Optimizing LightGBM...")
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'class_weight': 'balanced',
                'n_jobs': -1,
                'random_state': config.RANDOM_STATE,
                'verbose': -1
            }
            
            clf = lgb.LGBMClassifier(**params)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=config.RANDOM_STATE)
            scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring='f1')
            return scores.mean()

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)
        
        logger.info(f"Best LightGBM Params: {study.best_params}")
        
        # Train final model
        with mlflow.start_run(run_name="Best_LightGBM"):
            best_params = study.best_params
            best_params['random_state'] = config.RANDOM_STATE
            # Ensure class_weight is set if not optimized
            if 'class_weight' not in best_params:
                best_params['class_weight'] = 'balanced'
                
            model = lgb.LGBMClassifier(**best_params)
            model.fit(X_train, y_train)
            
            metrics = self.evaluate_model(model, X_test, y_test, "LightGBM_Optuna")
            
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.lightgbm.log_model(model, "model")
            
            if metrics['f1'] > self.best_score:
                self.best_score = metrics['f1']
                self.best_model = model
                self.best_model_name = "LightGBM_Optuna"

    def evaluate_model(self, model, X_test, y_test, model_name) -> Dict[str, float]:
        """Calculate metrics and generate plots."""
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Business Cost
        total_cost = (fn * config.COST_FALSE_NEGATIVE) + (fp * config.COST_FALSE_POSITIVE)
        
        metrics = {
            "accuracy": model.score(X_test, y_test),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
            "pr_auc": average_precision_score(y_test, y_prob),
            "total_cost": total_cost,
            "false_negatives": fn,
            "false_positives": fp
        }
        
        logger.info(f"Results for {model_name}: F1={metrics['f1']:.4f}, Cost=${total_cost:,.0f}")
        
        # Generate Plots
        self._plot_evaluation(y_test, y_prob, model_name)
        
        return metrics

    def _plot_evaluation(self, y_true, y_prob, model_name):
        """Save ROC and PR curves."""
        output_dir = config.OUTPUTS_DIR / "model_evaluation"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc_score(y_true, y_prob):.2f}")
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title(f'ROC Curve - {model_name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.savefig(output_dir / f"roc_{model_name}.png")
        plt.close()
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(recall, precision, label=f"PR AUC = {average_precision_score(y_true, y_prob):.2f}")
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.legend()
        plt.savefig(output_dir / f"pr_{model_name}.png")
        plt.close()

    def run(self):
        """Execute full pipeline."""
        X_train, y_train, X_test, y_test = self.prepare_data()
        
        # 1. Baseline
        self.train_baseline(X_train, y_train, X_test, y_test)
        
        # 2. Advanced Models with Tuning
        self.optimize_xgboost(X_train, y_train, X_test, y_test, n_trials=10) # Reduced trials for speed
        self.optimize_lightgbm(X_train, y_train, X_test, y_test, n_trials=10)
        
        logger.info(f"Best Model: {self.best_model_name} with F1 Score: {self.best_score:.4f}")
        
        # Save best model to disk
        joblib.dump(self.best_model, config.MODELS_DIR / "best_model.joblib")
        logger.info(f"Best model saved to {config.MODELS_DIR / 'best_model.joblib'}")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.run()
