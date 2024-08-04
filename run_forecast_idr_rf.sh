#!/bin/bash

# Navigate to the forecast_idr_rf directory and run the forecast
cd "$(dirname "$0")/forecast_idr_rf"

# Run the forecast_idr_rf module
python -m forecast_idr_rf.forecast_idr_rf