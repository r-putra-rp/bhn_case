# Forecast IDR Project

## Short Description

This project involves forecasting the IDR (Indonesian Rupiah) exchange rate using machine learning models. It includes implementations for both OLS (Ordinary Least Squares) regression and Random Forest regression. The code covers data preprocessing, feature engineering, model training, and evaluation. 

## Requirements
- Python 3.10
- Packages as listed in the `requirements.txt`
## Project Structure

- `bahana/`
  - `forecast_idr_ols/`
    - `Dockerfile`
    - `__init__.py`
    - `forecast_idr_ols.py`
  - `forecast_idr_rf/`
    - `Dockerfile`
    - `__init__.py`
    - `forecast_idr_rf.py`
  - `analysis/`
    - `__init__.py`
    - `analysis.py`
  - `core/`
    - `__init__.py`
  - `utils/`
    - `__init__.py`
    - `config.py`
    - `logger.py`
  - `requirements.txt`
  - `docker-compose.yml`
  - `run_forecast_idr_ols.sh`
  - `run_forecast_idr_rf.sh`

## How to Run Using Shell Scripts

1. **Run the OLS Forecasting Model**

   Navigate to the root directory of the project and run the following shell script to execute the OLS forecasting model:

   ```bash
   ./run_forecast_idr_ols.sh
2. **Run the Random Forest Forecasting Model**

   Navigate to the root directory of the project and run the following shell script to execute the Random Forest Forecasting model:

   ```bash
   ./run_forecast_idr_rf.sh

## How to Run Using Shell Scripts

1. **Build and Run Docker Compose**

   Navigate to the root directory of the project and use Docker Compose to build and run the containers for both forecasting models:

   ```bash
   docker-compose up --build
   ```

   This command will build the Docker images as defined in the docker-compose.yml file and start the containers. Each container corresponds to a different model and will execute the respective forecasting code.

