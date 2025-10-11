
## 📋 Project Overview

This project is an end-to-end Machine Learning system designed to predict a company’s daily sales revenue based on various store, promotion, and holiday factors.

It demonstrates the full lifecycle of an ML project — from data collection, feature engineering, and model training, to deployment using a FastAPI web service.



## 🎯 Problem Statement

Retail companies often struggle to accurately forecast sales revenue due to fluctuating factors such as promotions, competition distance, holidays, and store type.

The goal of this project is to:

> Predict daily sales revenue for different stores, using historical data and business features — helping the company plan inventory, marketing, and staffing efficiently.




## 📊 Dataset

The dataset used comes from Kaggle’s Rossmann Store Sales dataset.

**Key columns include:** 

* Store — Unique ID for each store

* Date — Date of the observation

* Sales — Target variable (store revenue)

* Customers — Number of customers that day

* Promo — Whether a store is running a promotion

* StateHoliday — Indicates a state holiday

* SchoolHoliday — Indicates if school was closed

* StoreType, Assortment, PromoInterval, CompetitionDistance — Store and competition-related attributes



🧩 **Project Objectives**

1. Clean and preprocess historical store sales data

2. Engineer useful features for model learning

3. Train and evaluate regression models to predict sales

4. Build a reusable scikit-learn pipeline

5. Serve predictions via a FastAPI web service



## ⚙️ Tech Stack & Tools

Category Tools Used

Programming Language	Python 3.10+
Libraries for ML	pandas, numpy, scikit-learn, lightgbm
Model Deployment	FastAPI, Uvicorn
Model Serialization	joblib
IDE/Environment	VS Code, Jupyter Notebook
Dataset Source	Kaggle (Rossmann Store Sales)



## 🧠 Machine Learning Approach

1️⃣ **Data Preprocessing & Feature Engineering**

Merged store data with sales data

Parsed date fields (Year, Month, WeekOfYear, etc.)

Created new temporal features such as:

is_month_start, is_month_end

CompetitionOpenSinceMonths, Promo2SinceWeeks

Handled missing values using imputers

Encoded categorical variables using One-Hot Encoding


2️⃣ **Model Building**

Used LightGBM Regressor for its efficiency with tabular data

Built a full Pipeline using ColumnTransformer and Pipeline in scikit-learn:

Numeric features → SimpleImputer + StandardScaler

Categorical features → SimpleImputer + OneHotEncoder


3️⃣ **Model Evaluation**

Used Root Mean Squared Error (RMSE) as evaluation metric

Applied log1p() transformation on the target (Sales) to stabilize variance

Achieved strong validation performance on unseen data


4️⃣ **Model Saving**

Trained on the full dataset

Saved the entire pipeline as lgbm_pipeline_final.joblib using joblib.dump()


🌐 **API Deployment with FastAPI**

### Overview

The trained model was deployed locally using FastAPI — a modern, fast web framework for building APIs.

Steps:

1. Loaded saved model pipeline inside app.py

2. Defined SalesInput schema using Pydantic for input validation

3. Created a /predict endpoint to receive JSON requests and return predictions

4. Ran the API using uvicorn



Example JSON Input:

{
  "Store": 1,
  "DayOfWeek": 4,
  "Promo": 1,
  "StateHoliday": "0",
  "SchoolHoliday": 0,
  "StoreType": "a",
  "Assortment": "a",
  "CompetitionDistance": 200.0,
  "Promo2": 1,
  "PromoInterval": "Jan,Apr,Jul,Oct"
}

Example JSON Output:

{
  "Predicted_Sales": 5345.76
}

## 🧩 Project Structure

📁 Sales-Revenue-Projects
│
├── app/
│   ├── main.py              # FastAPI entry point
│   ├── schema.py            # Pydantic models for request/response
│   ├── utils.py             # Helper utilities for prediction and preprocessing
│
├── models/
│   ├── pipeline_final.joblib    # Trained ML pipeline
│   └── metadata.json            # Model details and configuration
│
├── training/
│   ├── feature_engineering.py   # Feature creation and transformation logic
│   ├── train_model.py           # Training and evaluation script
│
├── data/
│   └── rossmann_store_sales.csv # Raw dataset (if included)
│
├── requirements.txt
└── README.md



## 🧱 Challenges, Constraints & Solutions

Building this project wasn’t smooth sailing — here’s what I faced and how each was solved:
1. Missing & Inconsistent Data
Issue: Columns like CompetitionOpenSinceYear and Promo2SinceWeek had missing or mismatched values.
Solution: Carefully imputed missing fields with default or derived values and used datetime arithmetic to generate reliable competition and promo timelines.

2. Feature Leakage from Lag Variables
Issue: Rolling means and lag values risked including future data when grouped incorrectly.
Solution: Used groupby().shift() and ensured lag generation happens after sorting by store and date to prevent leakage.

3. Overfitting During Model Training
Issue: LightGBM overfitted quickly on smaller samples.
Solution:
 * Introduced early stopping
 * Used subsample and colsample parameters
 * Tuned hyperparameters (learning rate, depth, leaves)

4. Pipeline Integration Errors
Issue: ColumnTransformer caused schema mismatch when deployed.
Solution: Ensured categorical columns were cast to string and that feature order matched training during inference.

5. API Import Errors & Model Mismatch
Issue: ImportError: cannot import name 'PredictResponse' and incorrect prediction outputs during first deployment.
Solution: Fixed Pydantic model definitions and ensured metadata was correctly linked to the model path in utils.py.

6. Log-Transformation Misinterpretation
Issue: Predictions were in log scale due to log1p transformation.
Solution: Applied expm1() during inference to get back true sales predictions.

7. Continuous Loading During API Startup
Issue: Uvicorn app kept “loading” indefinitely.
Solution: Rebuilt project structure to follow standard FastAPI patterns and ensured relative imports (app.schema) were correct.



🔍 **Results & Insights**

The model achieved a low RMSE on validation data, showing good predictive performance.

Feature importance analysis showed that:

Promo, CompetitionDistance, and Month had the strongest impact on sales.

The API successfully returns real-time sales predictions based on user input.



🚀 **Future Improvements**

Add cross-validation and hyperparameter tuning (Optuna)

Add model monitoring (using MLflow)

Deploy API to Render or Railway for public access

Build a Streamlit frontend for easier user interaction



👤 **Author**

Developed by: Alex Marco
📧 Email: [marcoalex201804@gmail.com]
💼 LinkedIn: [www.linkedin.com/in/alex-marco1820]
📺 YouTube: [http://bit.ly/KodIQ]
🐙 GitHub: [https://github.com/Alex-Marco18]


“Real-world machine learning is not about building models — it’s about building systems that work reliably in production.” 

