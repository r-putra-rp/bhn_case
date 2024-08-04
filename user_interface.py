import os
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Union
from statsmodels.api import OLS, add_constant
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from core import Core

FILENAME = "data/IDRQ.csv"
MODEL_FILEPATH = "models"
PRECISION = 6

# load models
filenames = [f for f in os.listdir(MODEL_FILEPATH) if f.endswith(".pkl")]


def load_models(filename: str) -> Union[OLS, RandomForestRegressor, StandardScaler]:
    with open(filename, "rb") as f:
        model = joblib.load(f)
    return model


models: Dict[str, Union[OLS, RandomForestRegressor, StandardScaler]] = {}
for filename in filenames:
    filename = MODEL_FILEPATH + "/" + filename
    model = load_models(filename=filename)
    filename = filename.split("/")[1]
    models[filename] = model

scaler = models["scaler.pkl"]
models.pop("scaler.pkl")

print("Starting prediction engine")

year: str = input("\nInput latest year in 2023, 2024, etc format:\n")
try:
    year = int(year)
except:
    print(f"Input is not integer, please check! (user input: {year})")
    exit()


quarter: str = input("\nInput latest quarter in month (3, 6, 9, 12)\n")
try:
    quarter = int(quarter)
except:
    print(f"Input is not integer, please check! (user input: {quarter})")
    exit()

if quarter not in [3, 6, 9, 12]:
    print(
        f"Quarter must be one of (3, 6, 9, 12), please check! (user input: {quarter})"
    )
    exit()

usdtw: str = input("\nInput latest USDTW\n")
try:
    usdtw = round(float(usdtw), PRECISION)
except:
    print(f"USDTW is not a number, please check! (user input: {usdtw})")
    exit()

ca: str = input("\nInput CA\n")
try:
    ca = round(float(ca), PRECISION)
except:
    print(f"CA is not a number, please check! (user input: {ca})")
    exit()

print("\nSubmitted parameters are")
print(f"Quarter {quarter}")
print(f"USDTW   {usdtw:_}")
print(f"CA      {ca:_}")
opt: str = input("Is this correct? y/n\n")
if opt.lower() != "y":
    print(f"Input is not correct, please check! (user input: {opt})")
    exit()

# loading df
core = Core("interface")
df: pd.DataFrame = core.dataloader(FILENAME)["Complete Data"]

# adding to df
df_new = pd.DataFrame(
    {
        "Month-Year": [f"{year}-{quarter}-1"],
        "USDTW": [usdtw],
        "CA": [ca],
        "IDR": [np.nan],
    }
)
df_new["Month-Year"] = pd.to_datetime(df_new["Month-Year"], format="%Y-%m-%d")
df = pd.concat([df, df_new], ignore_index=True)

# transform data
df: pd.DataFrame = core.dataloader_w_feature_eng(
    save_scaler=False, df_dict={"Complete Data": df}
)["Complete Data"]

print("Latest database is ")
print(df.tail())
print(df[["Month-Year", "USDTW", "CA", "IDR"]].tail())
print("\n")
opt: str = input("Is this correct? y/n\n")
if opt.lower() != "y":
    print(f"Database is not correct, please check! (user input: {opt})")
    exit()


# predicting ols best rmse
for name, model in models.items():
    metric = name.split("_")[1]
    name = name.split("_")[0]
    name_verbose = (
        "Ordinary Least Square Regressor"
        if name == "ols"
        else "Random Forest Regressor"
    )
    metric_verbose = "R-squared" if metric == "r2" else "Root Mean Squared Error"
    print(f"\nPredicting using {name_verbose} with best {metric_verbose} metric")

    subset_df = df.iloc[[-1]]

    if name == "ols":
        X = add_constant(subset_df[["USDTW", "CA"]], has_constant="add")
        predictions = model.predict(X)
        predictions = round(float(predictions.iloc[0]), 2)
        print(f"Prediction result is {predictions:_}")
    else:
        X = subset_df[
            [
                "USDTW",
                "CA",
                "month_scaled",
                "pct_change_usdtw_scaled",
                "pct_change_ca_scaled",
            ]
        ]
        predictions = model.predict(X)
        predictions = round(float(predictions[0]), 2)
        print(f"Prediction result is {predictions:_}")
