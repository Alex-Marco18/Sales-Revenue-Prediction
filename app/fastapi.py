from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import ValidationError
from app.schema import PredictRequest, PredictResponse
from app.model_utils import load_model, build_input_dataframe
import numpy as np
import pandas as pd

# Initialize FastAPI app
app = FastAPI(title="Sales Prediction API")

# Allow CORS for local testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["*"],
)

# Load model and metadata on startup
@app.on_event("startup")
def startup_event():
    global PIPELINE, META, TARGET_TRANSFORM
    PIPELINE, META = load_model()
    TARGET_TRANSFORM = META.get("target_transform", None)
    if TARGET_TRANSFORM not in [None, "log1p"]:
        raise ValueError(f"Unsupported target_transform: {TARGET_TRANSFORM}")

# Health check route
@app.get("/health")
def health_check():
    return {"status": "ok"}

# ---------------------------------
# Single prediction endpoint
# ---------------------------------
@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, compute_lags: bool = False):
    """Make a prediction for a single input"""
    try:
        payload_dict = payload.dict()
    except ValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Build input DataFrame
    X = build_input_dataframe(payload_dict, META, compute_lags=compute_lags)

    # Validate columns
    missing_cols = [c for c in (META["num_cols"] + META["cat_cols"]) if c not in X.columns]
    if missing_cols:
        raise HTTPException(status_code=400, detail=f"Missing required columns: {missing_cols}")

    # Predict
    try:
        pred_raw = PIPELINE.predict(X)[0]
        if TARGET_TRANSFORM == "log1p":
            pred_sales = float(np.expm1(pred_raw))
        else:
            pred_sales = float(pred_raw)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    return {
        "prediction_sales": pred_sales,
        "prediction_raw": float(pred_raw),
        "model_version": META.get("model_version", "unknown")
    }

# ---------------------------------
# Batch prediction endpoint
# ---------------------------------
@app.post("/predict_batch")
def predict_batch(payloads: list[PredictRequest], compute_lags: bool = False):
    """Make predictions for a batch of inputs"""
    try:
        # Convert all payloads to dicts
        results = [p.dict() for p in payloads]

        # Build DataFrames for each
        dfs = [build_input_dataframe(r, META, compute_lags=compute_lags) for r in results]
        X = pd.concat(dfs, ignore_index=True)

        # Predict
        preds_raw = PIPELINE.predict(X)
        preds = np.expm1(preds_raw) if TARGET_TRANSFORM == "log1p" else preds_raw

        return {"predictions": [float(x) for x in preds]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")