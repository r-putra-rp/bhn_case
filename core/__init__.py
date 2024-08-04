import sys
import joblib
import pandas as pd
from typing import Dict, Union
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from statsmodels.api import OLS

sys.path.append("..")

from utils.logger import get_logger

logger = get_logger()

FILENAME = "../data/IDRQ.csv"
MODEL_FILEPATH = "../models"
PRECISION = 6


class Core:
    def __init__(self, name: str) -> None:
        self.name: str = name

    def has_variability(self, series: pd.Series) -> bool:
        if series.std() == 0:
            logger.warning(
                f"IDR at {self.name} has no variability, skipping this subset."
            )
            return False
        else:
            return True
    
    def dataloader_w_feature_eng(
        self,
        filename: str = FILENAME,
        save_scaler: bool = True,
        df_dict: Dict[str, pd.DataFrame] = None,
        scaler: StandardScaler = None
    ) -> Dict[str, pd.DataFrame]:

        if df_dict is None:
            df_dict = self.dataloader(filename)

        for name, df in df_dict.items():
            logger.debug(f"Feature engineering for {name}")

            # target for regressor
            df["pct_change_idr"] = df["USDTW"].pct_change()

            # stationarize main features
            df["month"] = df["Month-Year"].dt.month
            df["month_scaled"] = df["month"] / 12
            df["month_scaled"] = df["month_scaled"].round(2)
            df["pct_change_usdtw"] = df["USDTW"].pct_change()
            df["pct_change_ca"] = df["CA"].pct_change()
            
            
            # scaling data
            scaled_df = df[["USDTW", "CA", "pct_change_usdtw", "pct_change_ca"]].copy()
            scaled_colnames = ["usdtw_scaled", "ca_scaled", "pct_change_usdtw_scaled", "pct_change_ca_scaled"]

            if scaler is None:
                scaler = StandardScaler()
                scaler = scaler.fit(scaled_df)

            scaled_df = scaler.transform(scaled_df)
            scaled_df = pd.DataFrame(scaled_df, columns=scaled_colnames)
            for col in scaled_colnames:
                df[col] = scaled_df[col]
            
            del scaled_df

            # logger.debug(f"Dataframe head \n {df.head()}")
            # logger.debug(f"Dataframe tail \n {df.tail()}")
            # logger.debug(f"Dataframe info \n {df.info()}")
            # logger.debug(f"Dataframe desc \n {df.describe()}")
            
            df_dict[name] = df

            if name == "Complete Data" and save_scaler:
                self.save_models(name="scaler", model=scaler)
        return df_dict

    @staticmethod
    def round(num: float, precision: int = PRECISION) -> float:
        return round(num, precision)

    @staticmethod
    def save_models(name: str, model: Union[OLS, RandomForestRegressor, StandardScaler], filepath: str = MODEL_FILEPATH):
        filename = f"{filepath}/{name}.pkl"
        with open(filename, "wb") as f:
            joblib.dump(model, f)

    @staticmethod
    def dataloader(filename: str = FILENAME) -> Dict[str, pd.DataFrame]:
        # load the dataframe from csv file
        logger.debug(f"Loading data from {filename}")
        df = pd.read_csv(filename)

        # parse unnecessary CSV formatting
        df["CA"] = df["CA"].replace({",": ""}, regex=True).astype(float)
        df["IDR"] = df["IDR"].replace({",": ""}, regex=True).astype(float)

        # parse triwulan to month and year
        df["Month-Year"] = pd.to_datetime(df["Triwulan"], format="%b-%y")

        df = df[["Month-Year", "USDTW", "CA", "IDR"]]
        # logger.debug(f"Dataframe head \n {df.head()}")
        # logger.debug(f"Dataframe tail \n {df.tail()}")
        # logger.debug(f"Dataframe info \n {df.info()}")
        # logger.debug(f"Dataframe desc \n {df.describe()}")

        # slice the dataframe into three time slice
        time_slice_1 = df[df["Month-Year"] >= pd.Timestamp("1986-03-01")].reset_index(
            level=0, drop=True
        )  # all data starting from March 1986.
        time_slice_2 = df[df["Month-Year"] >= pd.Timestamp("2014-03-01")].reset_index(
            level=0, drop=True
        )  # the last 10 years up to March 2024
        time_slice_3 = df[df["Month-Year"] >= pd.Timestamp("2000-03-01")].reset_index(
            level=0, drop=True
        )  # 2ince March 2000.

        # personal analysis, decade data
        time_slice_4 = df[
            (df["Month-Year"] >= pd.Timestamp("1990-03-01"))
            & (df["Month-Year"] < pd.Timestamp("2000-03-01"))
        ].reset_index(level=0, drop=True)
        time_slice_5 = df[
            (df["Month-Year"] >= pd.Timestamp("2000-03-01"))
            & (df["Month-Year"] < pd.Timestamp("2010-03-01"))
        ].reset_index(level=0, drop=True)
        time_slice_6 = df[
            (df["Month-Year"] >= pd.Timestamp("2010-03-01"))
            & (df["Month-Year"] < pd.Timestamp("2020-03-01"))
        ].reset_index(level=0, drop=True)

        return {
            "Complete Data": df,
            "1986-2024 Data": time_slice_1,
            "2014-2024 Data": time_slice_2,
            "2000-2024 Data": time_slice_3,
            "1990-2000 Data": time_slice_4,
            "2000-2010 Data": time_slice_5,
            "2010-2020 Data": time_slice_6,
        }
