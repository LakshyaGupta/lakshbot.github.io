Analysis Application

This is a Flask-based web application designed to perform data analysis and prediction using various economic datasets. The application integrates machine learning techniques (LSTM) to forecast trends and prices for selected datasets.

Features

Merges and preprocesses multiple datasets.

Performs feature scaling and LSTM-based time series predictions.

Provides a REST API for data analysis.

Outputs predictions to a CSV file.

Requirements

Ensure you have the following dependencies installed:

Python 3.9+

Flask

pandas

numpy

matplotlib

scikit-learn

tensorflow

gunicorn

Install these dependencies via the requirements.txt file:

pip install -r requirements.txt

Files

analysis_app.py: The main Flask application file.

requirements.txt: List of dependencies required to run the application.

Usage

Running the Application Locally

Clone the repository and navigate to the project directory.

Run the Flask application:

python analysis_app.py

The application will start on http://127.0.0.1:8080/.

API Endpoints

GET /

Returns a simple welcome message.

POST /analyze

Performs data analysis and prediction.

Input: Form data with the key datasets. Use "all" to include all datasets, or provide a comma-separated list of dataset names (e.g., SP500, DJIA).

Output: A CSV file (predicted_data.csv) with predictions.

Example Request

Using curl to analyze all datasets:

curl -X POST -F "datasets=all" http://127.0.0.1:8080/analyze

Deployment

Google Cloud Platform

Create an app.yaml file for deployment:

runtime: python39
entrypoint: gunicorn -w 2 -b :$PORT analysis_app:app

Deploy the application:

gcloud app deploy

Access the hosted application using the provided URL.

Datasets

The application expects the following CSV files in the project directory:

SP500 (2).csv

FEDFUNDS (2).csv

DJIA (2).csv

NASDAQCOM (2).csv

NASDAQ100 (1).csv

GDPC1.csv

UNRATE.csv

DCOILWTICO.csv

DGS10.csv

M2V.csv

WM2NS.csv

Output

Predictions are saved in predicted_data.csv.

A plot is displayed showing actual and predicted values, as well as future forecasts.

Troubleshooting

Ensure all dataset files are in the correct format and present in the project directory.

If you encounter memory errors, increase the available heap size.

For deployment issues, verify the configuration in app.yaml and the dependencies in requirements.txt.
