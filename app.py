from flask import Flask, request, jsonify
import pandas as pd
from your_analysis_script import perform_analysis  # Replace with actual analysis logic

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    datasets = {}

    for file in files:
        df = pd.read_csv(file)
        datasets[file.filename] = df

    available_datasets = list(datasets.keys())
    return jsonify({'datasets': available_datasets})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    datasets = data['datasets']
    time_step = data['timeStep']
    prediction_weeks = data['predictionWeeks']

    results = perform_analysis(datasets, time_step, prediction_weeks)
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True)
