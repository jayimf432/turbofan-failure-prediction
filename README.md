# Predictive Maintenance System

An end-to-end Machine Learning pipeline for predicting turbofan engine failures using the NASA CMAPSS dataset.

## Project Structure

-   `src/`: Source code for data processing, training, and interpretation.
    -   `preprocessing.py`: Data cleaning, outlier removal, RUL calculation.
    -   `feature_engineering.py`: Advanced feature generation (rolling, lag, trend).
    -   `feature_selection.py`: Correlation filtering and importance-based selection.
    -   `train.py`: Model training (Logistic Regression, Random Forest, XGBoost, LightGBM) with Optuna tuning.
    -   `interpret_model.py`: SHAP-based model interpretation.
    -   `config.py`: Configuration parameters.
-   `notebooks/`: EDA notebooks.
-   `outputs/`: Generated artifacts (plots, selected features).
-   `models/`: Trained model files.

## Setup

1.  Calculated dependencies: `pip install -r requirements.txt`
2.  Run the pipeline:
    ```bash
    python src/preprocessing.py
    python src/feature_engineering.py
    python src/feature_selection.py
    python src/train.py
    python src/interpret_model.py
    ```

## Model Performance

The best performing model is **Random Forest** with an F1 score of **0.8557**.
Optimized for business cost (False Negative: $50k, False Positive: $2k).

## Interpretation

Model interpretation plots (SHAP summary, waterfall, dependence) are available in `outputs/interpretability/`.
