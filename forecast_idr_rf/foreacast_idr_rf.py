import sys
import pandas as pd
import numpy as np
from typing import Dict, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

sys.path.append("..")

from core import Core
from utils.logger import get_logger

logger = get_logger()

N_ESTIMATORS = 200
SUBSET_WINDOW_MIN = 4 * 5  # Starting subset for recursion to avoid overfitting
SAVE_FILES = ["rf_r2_complete_data", "rf_rmse_complete_data"]


class ForecastIdrRf(Core):
    def __init__(self) -> None:
        super().__init__(name="RandomForest")

    def main(self) -> None:
        logger.debug(f"Starting IDR forecasting using {self.name}")

        # df_dict = self.dataloader()
        df_dict = self.dataloader_w_feature_eng()

        for name, df in df_dict.items():
            models = self.analysis(name, df)

            for name, model in models.items():
                if name in SAVE_FILES:
                    self.save_models(name=name, model=model)

    def analysis(
        self, name: str, df: pd.DataFrame
    ) -> Dict[str, Optional[RandomForestRegressor]]:
        #for method in ["direct", "percent"]:
        for method in ['direct']:
            best_r2: float = -float("inf")
            best_window_r2: int = 0
            best_endate_r2: pd.Timestamp = pd.Timestamp("1900-01-01")
            best_model_r2: Optional[RandomForestRegressor] = None
            betas_r2 = []

            best_rmse: float = float("inf")
            best_window_rmse: int = np.inf
            best_endate_rmse: pd.Timestamp = pd.Timestamp("1900-01-01")
            best_model_rmse: Optional[RandomForestRegressor] = None
            betas_rmse = []

            for window in range(SUBSET_WINDOW_MIN, len(df) + 1):
                # get the subset of data for the current window, moving backward
                df_copy = df.copy()
                df_copy["idr_prev"] = df_copy["IDR"].shift(1)
                df_copy = df_copy.dropna()
                subset_df = df_copy.iloc[-window:]
                subset_df = subset_df.dropna()

                # prepare data for regression
                X = subset_df[
                    [
                        "USDTW",
                        "CA",
                        "month_scaled",
                        "pct_change_usdtw_scaled",
                        "pct_change_ca_scaled",
                    ]
                ]

                if method == "direct":
                    y = subset_df["IDR"]
                else:
                    y = subset_df["pct_change_idr"]

                if not self.has_variability(y):
                    continue

                # fit the Random Forest model
                model = RandomForestRegressor(
                    n_estimators=N_ESTIMATORS,
                    # verbose=True,
                    n_jobs=-1,
                )
                model.fit(X, y)
                predictions = model.predict(X)
                betas = model.feature_importances_

                if method == "direct":
                    r2 = model.score(X, y)
                    rmse = np.sqrt(mean_squared_error(y, predictions))
                else:
                    predictions_transformed = subset_df["idr_prev"] * (1 + predictions)
                    actual_idr = subset_df["IDR"]
                    r2 = r2_score(actual_idr, predictions_transformed)
                    rmse = np.sqrt(
                        mean_squared_error(actual_idr, predictions_transformed)
                    )

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

                del df_copy, subset_df

            logger.debug("\n\n")
            logger.debug(f" ==================== {name} ==================== ")

            logger.debug(
                f"Best RF using {method} method R2: {best_r2} with RMSE {best_r2_acc_rmse} window {best_window_r2} quarters or at {best_end_date_r2}"
            )
            importance_strings = [f"{imp:.4f}" for imp in betas_rmse]
            equation = f"Feature Importances: USDTW = {importance_strings[0]}, CA = {importance_strings[1]}"
            logger.debug(f"Model RMSE Feature Importances: {equation}")

            logger.debug("")

            logger.debug(
                f"Best RF using {method} method RMSE: {best_rmse} with R2 {best_rmse_acc_r2} window {best_window_rmse} quarters or at {best_end_date_rmse}"
            )
            importance_strings = [f"{imp:.4f}" for imp in betas_rmse]
            equation = f"Feature Importances: USDTW = {importance_strings[0]}, CA = {importance_strings[1]}"
            logger.debug(f"Model RMSE Feature Importances: {equation}")

            logger.debug(f" ==================== {name} ==================== ")
            logger.debug("\n\n")

        name = name.replace("-", "_").replace(" ", "_").lower()

        return {
            f"rf_r2_{name}": best_model_r2,
            f"rf_rmse_{name}": best_model_rmse,
        }


if __name__ == "__main__":
    forecast = ForecastIdrRf()
    forecast.main()
