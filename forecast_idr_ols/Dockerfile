# Use an official Python runtime as a parent image
FROM python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container at /app
COPY ../requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /app/forecast_idr_ols
COPY ../core /app/core
COPY ../utils /app/utils
COPY . /app/forecast_idr_ols

# Make the run script executable
RUN chmod +x /app/forecast_idr_ols/run_forecast_idr_ols.sh

# Run the application
CMD ["sh", "/app/forecast_idr_ols/run_forecast_idr_ols.sh"]
