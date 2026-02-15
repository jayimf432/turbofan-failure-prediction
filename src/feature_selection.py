import pandas as pd
import numpy as np
import logging
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_regression
from typing import List, Dict, Tuple
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
        logging.FileHandler(config.LOGS_DIR / "feature_selection.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def remove_correlated_features(df: pd.DataFrame, threshold: float = config.FEATURE_SELECTION_PARAMS["correlation_threshold"]) -> Tuple[pd.DataFrame, List[str]]:
    """
    Remove features that are highly correlated with each other.
    Keeps the first feature encountered.
    """
    logger.info(f"Removing correlated features with threshold {threshold}...")
    
    # Calculate correlation matrix
    # Only use numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    corr_matrix = numeric_df.corr().abs()
    
    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features with correlation greater than threshold
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    
    logger.info(f"Identified {len(to_drop)} correlated features to remove.")
    
    df_reduced = df.drop(columns=to_drop)
    return df_reduced, to_drop

def calculate_feature_importance(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Calculate feature importance using Random Forest.
    """
    logger.info("Calculating Random Forest feature importance...")
    
    rf = RandomForestRegressor(
        n_estimators=100,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    importances = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values(by='importance', ascending=False)
    
    return importances

def select_top_features(importance_scores: pd.DataFrame, n: int = config.FEATURE_SELECTION_PARAMS["top_k_features"]) -> List[str]:
    """Select top n features."""
    top_features = importance_scores.head(n)['feature'].tolist()
    logger.info(f"Selected top {n} features.")
    return top_features

def plot_feature_analysis(df: pd.DataFrame, importance_scores: pd.DataFrame, selected_features: List[str]):
    """Create and save visualizations."""
    output_dir = config.OUTPUTS_DIR / "feature_selection"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Feature Importance Bar Plot (Top 20)
    plt.figure(figsize=(12, 8))
    sns.barplot(x='importance', y='feature', data=importance_scores.head(20))
    plt.title('Top 20 Features by Random Forest Importance')
    plt.tight_layout()
    plt.savefig(output_dir / "feature_importance.png")
    plt.close()
    
    # 2. Correlation Heatmap of Selected Features
    plt.figure(figsize=(15, 12))
    sns.heatmap(df[selected_features].corr(), cmap='coolwarm', center=0)
    plt.title('Correlation Heatmap (Selected Features)')
    plt.tight_layout()
    plt.savefig(output_dir / "selected_features_correlation.png")
    plt.close()
    
    logger.info(f"Plots saved to {output_dir}")

def main():
    try:
        logger.info("Starting Feature Selection...")
        
        # Load engineered data (using train set for selection)
        train_path = config.PROCESSED_DATA_DIR / "train_engineered.csv"
        
        if not train_path.exists():
            raise FileNotFoundError("Engineered data not found. Run feature engineering first.")
            
        df = pd.read_csv(train_path)
        
        # Separate Target
        target = 'RUL'
        if target not in df.columns:
             raise ValueError("Target column 'RUL' not found.")
             
        X = df.drop(columns=[target] + config.ALL_COLS, errors='ignore')
        y = df[target]
        
        # 1. Remove Correlated Features
        X_uncorr, dropped_corr = remove_correlated_features(X)
        logger.info(f"Features remaining after correlation filter: {X_uncorr.shape[1]}")
        
        # 2. Calculate Importance
        importance_df = calculate_feature_importance(X_uncorr, y)
        
        # Save importance scores
        importance_df.to_csv(config.OUTPUTS_DIR / "feature_importance.csv", index=False)
        
        # 3. Select Top Features
        top_features = select_top_features(importance_df)
        
        # 4. Visualize
        plot_feature_analysis(X_uncorr, importance_df, top_features)
        
        # 5. Save Selected Feature List
        # We need to save this so inference/training knows what to use
        import json
        with open(config.OUTPUTS_DIR / "selected_features.json", "w") as f:
            json.dump(top_features, f, indent=4)
            
        # Create final datasets with selected features + ID columns + Target
        # Keep ID columns for tracking
        base_cols = [c for c in config.ALL_COLS if c in df.columns]
        final_cols = base_cols + top_features + [target]
        
        # Filter Train
        train_selected = df[final_cols]
        train_selected.to_csv(config.PROCESSED_DATA_DIR / "train_final.csv", index=False)
        
        # Filter Test
        test_path_eng = config.PROCESSED_DATA_DIR / "test_engineered.csv"
        if test_path_eng.exists():
            test_df = pd.read_csv(test_path_eng)
            test_selected = test_df[final_cols]
            test_selected.to_csv(config.PROCESSED_DATA_DIR / "test_final.csv", index=False)
        
        logger.info("Feature selection complete. Final datasets saved.")
        
    except Exception as e:
        logger.error(f"Feature Selection failed: {e}")
        # raise

if __name__ == "__main__":
    main()
