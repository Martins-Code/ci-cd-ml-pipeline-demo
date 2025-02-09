from flask import Flask, request, render_template, jsonify
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

# Route for Homepage
@app.route('/')
def index():
    """Render the index page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict_datapoint():
    """
    Handle prediction requests. 
    - On `GET` request: Render the prediction form (`home.html`).
    - On `POST` request: Process input data, make a prediction, and return the result.
    """
    if request.method == 'GET':
        return render_template('home.html')
    
    try:
        # Collect form input and create a `CustomData` instance
        student_data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),   
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=int(request.form.get('reading_score')),
            writing_score=int(request.form.get('writing_score'))
        )

        # Convert to DataFrame for model compatibility
        input_data = student_data.to_dataframe()
        
        # Load and run the prediction pipeline
        predictor = PredictPipeline()
        predicted_score = predictor.predict(input_data)

        # Render the result on the home page
        return render_template('home.html', prediction=round(predicted_score[0], 2))

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)
