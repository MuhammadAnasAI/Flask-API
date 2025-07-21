# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('diabetese_model.pkl', 'rb'))

# Home route that renders the form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route that handles form submissions
@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        age = int(request.form['Age'])
        bloodpressure = int(request.form['BloodPressure'])  # 0 for male, 1 for female
        
        
        # Prepare the data for prediction
        features = np.array([[age, bloodpressure]])
        
        # Make a prediction
        prediction = model.predict(features)
        # Output the result
        result = f'The value of Pregnancies is: {prediction[0]}'
        
        return render_template('index.html', prediction_text=result)
    
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
