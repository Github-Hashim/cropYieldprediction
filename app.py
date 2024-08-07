from src.cropYieldprediction.logger import logging
from src.cropYieldprediction.exception import CustomException
import sys
from src.cropYieldprediction.components.data_ingestion import DataIngestion
from src.cropYieldprediction.components.data_transformation import DataTransformation
from src.cropYieldprediction.components.model_trainer import ModelTrainer
from flask import Flask, request, render_template
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('artifacts/model.pkl')  # Replace 'model.pkl' with your model's filename

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            rain_fall = float(request.form['rain_fall'])
            fertilizer = float(request.form['fertilizer'])
            temperature = float(request.form['temperature'])
            nitrogen = float(request.form['nitrogen'])
            phosphorus = float(request.form['phosphorus'])
            potassium = float(request.form['potassium'])
            
            # Create a DataFrame for the input
            input_data = pd.DataFrame([[rain_fall, fertilizer, temperature, nitrogen, phosphorus, potassium]],
                                      columns=['Rain Fall (mm)', 'Fertilizer', 'Temperature', 'Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)'])
            
            # Make prediction
            prediction = model.predict(input_data)[0]

            return render_template('index.html', prediction=prediction)
        except Exception as e:
            return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)
