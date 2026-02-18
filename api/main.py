import logging
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
from typing import List

from api.schemas import (
    PredictionRequest, PredictionResponse, 
    BatchPredictionRequest, BatchPredictionResponse,
    HealthResponse, FeatureImportanceResponse
)
from src.inference import pipeline
from src import config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("api")

app = FastAPI(
    title="Turbofan Failure Prediction API",
    description="Predictive Maintenance API for NASA Turbofan Engines",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Starting up API...")
    # Ensure pipeline is loaded
    if not pipeline.initialized:
        pipeline.load_artifacts()

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint returning system status and model info.
    """
    return {
        "status": "healthy",
        "model_loaded": pipeline.initialized,
        "model_version": "v1.0", # Could be dynamic from mlflow
        "timestamp": datetime.now()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Predict failure probability for a single unit based on historical sensor data.
    """
    try:
        result = pipeline.predict(request.data)
        return result
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple units.
    """
    predictions = []
    try:
        for unit_req in request.units:
            pred = pipeline.predict(unit_req.data)
            predictions.append(pred)
        return {"predictions": predictions}
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/feature_importance", response_model=FeatureImportanceResponse)
async def get_feature_importance():
    """
    Return global feature importance from the model.
    """
    try:
        if hasattr(pipeline.model, "feature_importances_"):
            importances = pipeline.model.feature_importances_
            features = pipeline.feature_names
            
            # Zip and sort
            feat_imp = [
                {"feature": f, "importance": float(i)} 
                for f, i in zip(features, importances)
            ]
            feat_imp.sort(key=lambda x: x["importance"], reverse=True)
            
            return {"features": feat_imp[:20]} # Top 20
        else:
             raise HTTPException(status_code=400, detail="Model does not support feature importance")
    except Exception as e:
        logger.error(f"Feature importance failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """
    Trigger model retraining in the background. (Placeholder)
    """
    background_tasks.add_task(dummy_retrain_task)
    return {"message": "Retraining started in background"}

def dummy_retrain_task():
    import time
    logger.info("Retraining started...")
    time.sleep(5)
    logger.info("Retraining completed.")
