from flask import Flask, request, render_template
import joblib
import numpy as np

# Initialize the Flask application
app = Flask(__name__)

# Load the trained model, scaler, and columns
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')

@app.route('/')
def home():
    # Render the home page (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the data from the POST request (from our form)
    # The order of features must be the same as in 'model_columns.joblib'
    
    # Create a list to hold the input data in the correct order
    input_data = []
    for col in model_columns:
        # We use .get() to get the form value by its name
        # The type is converted to float
        value = request.form.get(col, type=float)
        input_data.append(value)

    # Convert the list to a NumPy array for the model
    final_features = [np.array(input_data)]
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(final_features)
    
    # Make a prediction
    prediction = model.predict(scaled_features)
    
    # Get the probability of churn
    prediction_proba = model.predict_proba(scaled_features)

    # Determine the output message
    if prediction[0] == 1:
        output = "This customer is likely to CHURN."
        churn_prob = prediction_proba[0][1] * 100
        result_message = f"{output} (Probability: {churn_prob:.2f}%)"
    else:
        output = "This customer is likely to STAY."
        churn_prob = prediction_proba[0][1] * 100
        result_message = f"{output} (Churn Probability: {churn_prob:.2f}%)"

    # Render the same page but with the prediction result
    return render_template('index.html', prediction_text=result_message)

if __name__ == "__main__":
    # Run the app
    app.run(debug=True)