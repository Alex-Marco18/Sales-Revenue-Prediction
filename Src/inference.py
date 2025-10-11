from joblib import load
import json
import numpy as np
import os
import pandas as pd


# 1. Set up paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "pipeline_final.joblib")
metadata_path = os.path.join(BASE_DIR, "models", "metadata.json")
pipeline_path = model_path  # for clarity
# metadata_path already defined





# 2. Load pipeline and metadata
print("Loading model pipeline...")
pipeline = load(pipeline_path)
print("Model loaded successfully!")

with open(metadata_path, "r") as f:
    metadata = json.load(f)

print("Loaded metadata from:", metadata_path)
print("Trained at:", metadata["trained_at"])


# 3. Create sample input 
# Replace these values with real examples or values from your dataset
sample_data = pd.DataFrame([{
    "Customers": 850,                       # average daily customers
    "CompetitionDistance": 1200.0,          # meters to nearest competitor
    "CompetitionOpenSinceMonths": 36,       # competitor has been open for 3 years
    "Promo": 1,                             # store currently running a promotion
    "Promo2": 1,                            # second-level promo active
    "Promo2SinceWeeks": 8,                  # promo2 running for 8 weeks
    "Year": 2015,
    "Month": 7,                             # July (summer season)
    "Day": 10,                              # specific day
    "DayOfWeek": 5,                         # Friday
    "WeekOfYear": 28,
    "lag_7": 7350,                          # sales 7 days ago
    "lag_14": 7200,                         # sales 14 days ago
    "lag_28": 7100,                         # sales 28 days ago
    "roll_mean_7": 7450,                    # 7-day rolling mean
    "roll_mean_28": 7300,                   # 28-day rolling mean
    "roll_std_7": 150.0,                    # short-term variation
    "is_month_start": 0,
    "is_month_end": 0,
    "StoreType": "b",                       # e.g. larger suburban store
    "Assortment": "extra",                  # carries more product variety
    "StateHoliday": "0",                    # no holiday
    "SchoolHoliday": 1                      # schools on break → more customers
}])
print("Sample input data:")
print(sample_data)

# 4. Run prediction
print("Making prediction...")
pred_log = pipeline.predict(sample_data)
pred = np.expm1(pred_log)  # Inverse of log1p

print(f"Predicted Sales: {pred[0]:,.2f}")


# 5. Save prediction

prediction_result = {"predicted_value": float(pred[0])}
output_path = os.path.join(BASE_DIR, "prediction.json")

with open(output_path, "w") as f:
    json.dump(prediction_result, f, indent=2)

print("Prediction saved to:", output_path)