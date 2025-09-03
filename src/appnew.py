from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle

# Load the saved model and scaler
model = pickle.load(open('model.pkl', 'rb'))
ms = pickle.load(open('minmaxscaler.pkl', 'rb'))

# Initialize the Flask App
app = Flask(__name__)

@app.route('/')
def index():
    # Renders the main page of the web application
    # --- FIX: Changed to your filename ---
    return render_template("UserInterface.html")

@app.route("/predict", methods=['POST'])
def predict():
    # Get values from the form and convert them to float
    N = float(request.form['Nitrogen'])
    P = float(request.form['Phosporus'])
    K = float(request.form['Potassium'])
    temp = float(request.form['Temperature'])
    humidity = float(request.form['Humidity'])
    ph = float(request.form['Ph'])
    rainfall = float(request.form['Rainfall'])

    # Create a list and then a DataFrame
    feature_list = [N, P, K, temp, humidity, ph, rainfall]
    feature_names = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
    single_pred_df = pd.DataFrame([feature_list], columns=feature_names)

    # Scale the features
    scaled_features = ms.transform(single_pred_df)

    # Make the prediction
    prediction = model.predict(scaled_features)

    # Map the prediction to a crop name
    crop_dict = {
        1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut",
        6: "Papaya", 7: "Orange", 8: "Apple", 9: "Muskmelon",
        10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
        14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
        17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas",
        20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
    }

    if prediction[0] in crop_dict:
        crop = crop_dict[prediction[0]]
        result = "{} is the best crop to be cultivated right there".format(crop)
    else:
        result = "Sorry, we could not determine the best crop for the provided data."

    # Render the page with the result
    # --- FIX: Changed to your filename ---
    return render_template('UserInterface.html', result=result)

# Run the Flask App
if __name__ == "__main__":
    app.run(debug=True)