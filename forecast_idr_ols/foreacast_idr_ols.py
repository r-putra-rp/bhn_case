import sys
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from sklearn.metrics import mean_squared_error
from typing import Dict, Optional

sys.path.append("..")

from core import Core
from utils.logger import get_logger

logger = get_logger()

SUBSET_WINDOW_MIN = 4 * 5  # Starting subset for recursion to avoid overfitting
SAVE_FILES = ["ols_r2_complete_data", "ols_rmse_complete_data"]


class ForecastIdrOls(Core):
    def __init__(self) -> None:
        super().__init__(name="OLS")

    def main(self) -> None:
        logger.debug(f"Starting IDR forecasting using {self.name}")

        df_dict = self.dataloader_w_feature_eng()

        for name, df in df_dict.items():
            models = self.analysis(name, df)

            for name, model in models.items():
                if name in SAVE_FILES:
                    self.save_models(name=name, model=model)

    def analysis(
        self,
        name: str, 
        df: pd.DataFrame
    ) -> Dict[str, Optional[OLS]]:
        best_r2: float = -float("inf")
        best_window_r2: int = 0
        best_end_date_r2: pd.Timestamp = pd.Timestamp("1900-01-01")
        best_model_r2: Optional[OLS] = None
        betas_r2 = []

        best_rmse: float = float("inf")
        best_window_rmse: int = np.inf
        best_end_date_rmse: pd.Timestamp = pd.Timestamp("1900-01-01")
        best_model_rmse: Optional[OLS] = None
        betas_rmse = []

        for window in range(SUBSET_WINDOW_MIN, len(df) + 1):
            # get the subset of data for the current window, moving backward
            subset_df = df.iloc[-window:]

            # prepare data for regression
            X = add_constant(subset_df[["USDTW", "CA"]])
            y = subset_df["IDR"]

            if not self.has_variability(y):
                continue

            # fit the OLS model
            model = OLS(y, X).fit()
            r2 = model.rsquared
            predictions = model.predict(X)
            betas = model.params

            rmse = np.sqrt(mean_squared_error(y, predictions))

            # update best model based on the chosen metric
            if r2 > best_r2:
                best_r2 = self.round(r2)
                best_r2_acc_rmse = self.round(rmse)
                best_window_r2 = window
                best_end_date_r2 = subset_df.iloc[0]["Month-Year"]
                best_model_r2 = model
                betas_r2 = betas

            if rmse < best_rmse:
                best_rmse = self.round(rmse)
                best_rmse_acc_r2 = self.round(r2)
                best_window_rmse = window
                best_end_date_rmse = subset_df.iloc[0]["Month-Year"]
                best_model_rmse = model
                betas_rmse = betas

        logger.debug("\n\n")
        logger.debug(f" ==================== {name} ==================== ")

        logger.debug(
            f"Best OLS R2: {best_r2} with RMSE {best_r2_acc_rmse} window {best_window_r2} quarters or at {best_end_date_r2}"
        )

        beta_strings = [f"{coef:.4f}" for coef in betas_r2]
        equation = f"IDR = {beta_strings[0]} + {beta_strings[1]} * USDTW + {beta_strings[2]} * CA"
        logger.debug(f"Model R2 Betas: {equation}")

        logger.debug("")

        logger.debug(
            f"Best OLS RMSE: {best_rmse} with R2 {best_rmse_acc_r2} window {best_window_rmse} quarters or at {best_end_date_rmse}"
        )
        beta_strings = [f"{coef:.4f}" for coef in betas_rmse]
        equation = f"IDR = {beta_strings[0]} + {beta_strings[1]} * USDTW + {beta_strings[2]} * CA"
        logger.debug(f"Model RMSE Betas: {equation}")

        logger.debug(f" ==================== {name} ==================== ")
        logger.debug("\n\n")

        name = name.replace("-", "_").replace(" ", "_").lower()

        return {
            f"ols_r2_{name}": best_model_r2,
            f"ols_rmse_{name}": best_model_rmse,
        }


if __name__ == "__main__":
    forecast = ForecastIdrOls()
    forecast.main()
