from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from src.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file uploaded
        if 'file' in request.files and request.files['file'].filename != '':
            file = request.files['file']
            df = pd.read_csv(file)
        else:
            # Manual input from form
            # Get all sensor columns from raw.csv header except 'Unnamed: 0' and 'Good/Bad'
            all_columns = pd.read_csv('artifact/raw.csv', nrows=0).columns.tolist()
            sensor_columns = [col for col in all_columns if col not in ['Unnamed: 0', 'Good/Bad']]

            # Initialize input data with zeros for all sensors
            input_data = {col: [0.0] for col in sensor_columns}
            # Add 'Unnamed: 0' column with default value 0
            input_data['Unnamed: 0'] = [0]

            # Replace first 5 sensors with user input
            feature_names = [col for col in sensor_columns[:5]]
            for feature in feature_names:
                val = request.form.get(feature)
                if val is None or val == '':
                    return render_template('index.html', error=f"Missing value for {feature}")
                if feature not in input_data:
                    return render_template('index.html', error=f"Unexpected feature: {feature}")
                input_data[feature] = [float(val)]

            df = pd.DataFrame(input_data)

        pipeline = PredictPipeline()
        preds = pipeline.predict(df)

        results = ["Good Wafer" if p == 1 else "Bad Wafer" for p in preds]
        if not results:
            return render_template('index.html', error="No predictions were generated.")
        return render_template('index.html', predictions=results)
    except Exception as e:
        return render_template('index.html', error=f"Prediction error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)