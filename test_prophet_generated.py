import pandas as pd
import numpy as np
from prophet import Prophet

df = pd.read_csv("AirPassengers.csv")


import pandas as pd
import numpy as np
from prophet import Prophet

def run(df: pd.DataFrame) -> dict:
    if 'Month' not in df.columns or 'Passengers' not in df.columns:
        raise ValueError("Required columns 'Month' and 'Passengers' are missing from the DataFrame.")
    
    df_prophet = df.rename(columns={'MONTH': 'ds', 'Passengers': 'y'})
    df_prophet['ds'] = pd.to_datetime(df_prophet['ds'], errors='coerce')
    df_prophet['y'] = pd.to_numeric(df_prophet['y'], errors='coerce')
    df_prophet = df_prophet.dropna(subset=['ds', 'y'])

    if df_prophet.empty:
        return {"forecast_head": [], "columns_used": {}, "n_rows_used": 0, "notes": "No valid data available for forecasting."}

    model = Prophet()
    
    model.fit(df_prophet)

    future = model.make_future_dataframe(periods=30, freq='M')
    
    forecast = model.predict(future)
    
    return {
        "forecast_head": forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head(10).to_dict(orient='records'),
        "columns_used": {'ds': 'Month', 'y': 'Passengers'},
        "n_rows_used": len(df_prophet),
        "notes": "Forecast generated successfully."
    }

result = run(df)
print(result)