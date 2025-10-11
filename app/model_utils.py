# corrected_module.py
import os
import json
from joblib import load
import pandas as pd
import numpy as np
from datetime import datetime

# ----------------------
# Paths
# ----------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
PIPELINE_PATH = os.path.join(BASE_DIR, "..", "Src", "models", "pipeline_final.joblib")
METADATA_PATH = os.path.join(BASE_DIR, "..", "Src", "models", "metadata.json")
RECENT_SALES_CSV = os.path.join(BASE_DIR, "..", "data", "recent_sales.csv")




# ----------------------
# Model loader
# ----------------------
def load_model(pipeline_path=PIPELINE_PATH, metadata_path=METADATA_PATH):
    """
    Load pipeline and metadata. Return (pipeline, metadata).
    """
    print("Loading model pipeline...")
    pipeline = load(pipeline_path)
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    print("Model and metadata loaded.")
    return pipeline, metadata


# ----------------------
# Date parts helper
# ----------------------
def compute_date_parts(date: pd.Timestamp):
    """
    Return a dict with Year, Month, Day, DayOfWeek (1..7), WeekOfYear.
    Accepts pandas.Timestamp or anything pd.to_datetime understands.
    """
    if pd.isna(date):
        return {}
    ts = pd.to_datetime(date)
    # dayofweek: Monday=0 -> convert to 1..7
    day_of_week = int(ts.dayofweek) + 1
    # week: robust extraction (works across pandas versions)
    try:
        week_of_year = int(ts.isocalendar().week)
    except Exception:
        # fallback for older versions
        week_of_year = int(ts.isocalendar()[1])
    return {
        "Year": int(ts.year),
        "Month": int(ts.month),
        "Day": int(ts.day),
        "DayOfWeek": day_of_week,
        "WeekOfYear": week_of_year,
    }


# ----------------------
# Lag & rolling features helper
# ----------------------
def compute_lags_from_recent_sales(
    store_id: int,
    target_date,
    recent_df: pd.DataFrame | None = None,
    recent_sales_csv: str = RECENT_SALES_CSV,
    lookback_days: tuple = (7, 14, 28),
):
    """
    Compute lag_N and roll_mean/std features for a store using recent sales.
    - recent_df: optional DataFrame (if you preloaded recent_sales.csv to avoid re-reading).
    - Returns dict like {"lag_7": ..., "lag_14": ..., "roll_mean_7": ..., ...}
    """
    # Load CSV only if recent_df not provided
    if recent_df is None:
        try:
            recent_df = pd.read_csv(recent_sales_csv, parse_dates=["Date"])
        except FileNotFoundError:
            # no recent sales data available -> cannot compute lags
            return {}

    # normalize the date column (drop time-of-day differences)
    recent_df = recent_df.copy()
    recent_df["Date"] = pd.to_datetime(recent_df["Date"]).dt.normalize()
    target_ts = pd.to_datetime(target_date).normalize()

    # filter for store
    df_store = recent_df[recent_df["Store"] == store_id].copy()
    if df_store.empty:
        return {}

    # keep only history strictly before target_date
    hist = df_store[df_store["Date"] < target_ts].sort_values("Date")
    if hist.empty:
        return {}

    out: dict = {}

    # compute simple lags (exact date matches)
    for lag in lookback_days:
        ref_date = target_ts - pd.Timedelta(days=lag)
        row = hist[hist["Date"] == ref_date]
        out[f"lag_{lag}"] = float(row["Sales"].iloc[0]) if not row.empty else np.nan

    # create daily series from the earliest date to target-1 to ensure contiguous index
    start_range = max(hist["Date"].min(), target_ts - pd.Timedelta(days=60))  # safe window
    full_index = pd.date_range(start=start_range, end=target_ts - pd.Timedelta(days=1), freq="D")
    daily_sales = (
        hist.set_index("Date")["Sales"].resample("D").sum().reindex(full_index, fill_value=0)
    )

    # 7-day window (last 7 days before target_date)
    end_7 = target_ts - pd.Timedelta(days=1)
    start_7 = end_7 - pd.Timedelta(days=6)
    window_7 = daily_sales.loc[start_7:end_7]
    out["roll_mean_7"] = float(window_7.mean()) if len(window_7) > 0 else np.nan
    out["roll_std_7"] = float(window_7.std(ddof=0)) if len(window_7) > 0 else np.nan

    # 28-day window
    start_28 = target_ts - pd.Timedelta(days=28)
    window_28 = daily_sales.loc[start_28:end_7]
    out["roll_mean_28"] = float(window_28.mean()) if len(window_28) > 0 else np.nan

    return out


# ----------------------
# Build input DataFrame from payload
# ----------------------
def build_input_dataframe(payload: dict, meta: dict, compute_lags: bool = False, recent_df: pd.DataFrame | None = None):
    """
    Build a one-row DataFrame ready for pipeline.predict.
    - payload: dict (e.g. from request.json())
    - meta: metadata dict (must contain "num_cols" and "cat_cols")
    - compute_lags: if True, will attempt to compute lag features using recent sales
    - recent_df: optional preloaded recent sales DataFrame to speed repeated calls
    """
    num_cols = list(meta.get("num_cols", []))
    cat_cols = list(meta.get("cat_cols", []))
    required_cols = num_cols + cat_cols

    # parse date (if provided)
    date_val = payload.get("Date", None)
    date_ts = None
    if date_val is not None:
        try:
            date_ts = pd.to_datetime(date_val)
        except Exception:
            date_ts = None

    # create row dict with placeholders (use None or np.nan)
    row = {}
    for c in required_cols:
        row[c] = payload.get(c, np.nan)

    # If date provided, compute and overwrite date parts
    if date_ts is not None:
        row.update(compute_date_parts(date_ts))

    # compute missing lags if requested and if we have a Store id
    missing_lags = any(pd.isna(row.get(f"lag_{lag}", np.nan)) for lag in (7, 14, 28))
    if missing_lags and compute_lags and ("Store" in payload) and (date_ts is not None):
        try:
            store_id = int(payload["Store"])
        except Exception:
            store_id = None
        if store_id is not None:
            lags = compute_lags_from_recent_sales(store_id, date_ts, recent_df=recent_df)
            for k, v in lags.items():
                # only fill if key exists and is missing
                if k in row and (pd.isna(row[k]) or row[k] is None):
                    row[k] = v

    # Build DataFrame and ensure column order
    df = pd.DataFrame([row])
    # If some required columns are missing from row, add them with NaN
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[required_cols]  # reorder / select only required columns

    # Cast numeric columns to numeric types
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Optional: cast categorical columns to category (or leave as-is)
    for c in cat_cols:
        # keep NaNs — astype("category") can handle them
        df[c] = df[c].astype("category")

    return df