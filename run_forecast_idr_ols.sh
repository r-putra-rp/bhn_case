#!/bin/bash

# Navigate to the forecast_idr_ols directory and run the forecast
cd "$(dirname "$0")/forecast_idr_ols"

# Run the forecast_idr_ols module
python -m forecast_idr_ols.forecast_idr_ols