import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

sys.path.append("..")

from core import Core
from utils.logger import get_logger

logger = get_logger()


FILE_TO_ANALYZE = ["Complete Data"]
FEATURES = [
    "USDTW",
    "CA",
    "usdtw_scaled",
    "ca_scaled",
    "pct_change_usdtw",
    "pct_change_ca",
    "pct_change_usdtw_scaled",
    "pct_change_ca_scaled",
]
HIST_BINS = 25
ROLLING_WINDOW = 4


class Analysis(Core):
    def __init__(self) -> None:
        super().__init__(name="RandomForest")

    def main(self) -> None:
        df_dict = self.dataloader_w_feature_eng()

        for name in FILE_TO_ANALYZE:
            df: pd.DataFrame = df_dict[name]
            logger.debug(f"Dataframe head \n {df.head()}")
            logger.debug(f"Dataframe tail \n {df.tail().to_clipboard(index=False)}")
            logger.debug(f"Dataframe info \n {df.info()}")
            logger.debug(f"Dataframe desc \n {df.describe()}")
            # self.plot_hist(df)
            # self.plot_rolling_stats(df)

    @staticmethod
    def plot_rolling_stats(df: pd.DataFrame) -> None:
        for feat in FEATURES:
            plt.figure(figsize=(12, 8))
            rolling_mean = df[feat].rolling(window=ROLLING_WINDOW).mean()
            rolling_std = df[feat].rolling(window=ROLLING_WINDOW).std()

            # Plot rolling mean
            plt.plot(df.index, rolling_mean, label=f"{feat} Rolling Mean", color="blue")

            # Plot rolling standard deviation as whiskers
            plt.fill_between(
                df.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                color="blue",
                alpha=0.2,
                label=f"{feat} ±1 Std. Dev.",
            )

            plt.title(f"{feat} - Rolling Mean and Standard Deviation")
            plt.xlabel("Date")
            plt.ylabel("Value")
            plt.legend(loc="best")
            plt.grid(True)
            plt.show()

    @staticmethod
    def plot_hist(df: pd.DataFrame):
        for feat in FEATURES:
            plt.figure(figsize=(10, 6))
            plt.hist(df[feat].dropna(), bins=HIST_BINS, edgecolor="black")
            plt.title(f"Histogram of {feat}")
            plt.xlabel(feat)
            plt.ylabel("Frequency")
            plt.grid(axis="y", linestyle="--", alpha=0.7)
            plt.tight_layout()
            plt.show()
            logger.debug(f"Plotted histogram for {feat}")


if __name__ == "__main__":
    analysis = Analysis()
    analysis.main()
