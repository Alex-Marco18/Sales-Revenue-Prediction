from pydantic import BaseModel
from typing import Optional

class PredictRequest(BaseModel):
    Store: int
    Date: str
    Customers: int
    CompetitionDistance: float
    Promo: int
    Promo2: int
    lag_7: float
    lag_14: float
    lag_28: float
    roll_mean_7: float
    roll_mean_28: float
    roll_std_7: float
    StoreType: str
    Assortment: str
    StateHoliday: str
    SchoolHoliday: int
    CompetitionOpenSinceMonths: int
    Year: int

class PredictResponse(BaseModel):
    prediction_sales: float
    prediction_raw: float
    model_version: Optional[str] = "unknown"
