import pickle
import numpy as np
import os

# --- Load the trained model and scaler ---
# Ensure the script can find the files even when run from the same folder
try:
    model = pickle.load(open('model.pkl', 'rb'))
    scaler = pickle.load(open('minmaxscaler.pkl', 'rb'))
except FileNotFoundError:
    print("Error: 'model.pkl' or 'minmaxscaler.pkl' not found.")
    print("Please run 'train_model.py' first to generate these files.")
    exit()

# --- Define the crop dictionary ---
CROP_DICT = {
    1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya",
    7: "Orange", 8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes",
    12: "Mango", 13: "Banana", 14: "Pomegranate", 15: "Lentil", 16: "Blackgram",
    17: "Mungbean", 18: "Mothbeans", 19: "Pigeonpeas", 20: "Kidneybeans",
    21: "Chickpea", 22: "Coffee"
}

def get_user_input():
    """Gets and validates the 7 feature inputs from the user."""
    features = []
    feature_names = [
        "Nitrogen (N) content (e.g., 90)",
        "Phosphorus (P) content (e.g., 42)",
        "Potassium (K) content (e.g., 43)",
        "Temperature in °C (e.g., 20.8)",
        "Relative Humidity in % (e.g., 82.0)",
        "Soil pH value (e.g., 6.5)",
        "Rainfall in mm (e.g., 202.9)"
    ]
    
    print("\nEnter the following details for your farmland:")
    for name in feature_names:
        while True:
            try:
                value = float(input(f"- {name}: "))
                features.append(value)
                break
            except ValueError:
                print("  Invalid input. Please enter a number.")
    return np.array(features)

def main():
    """Main function to run the prediction CLI."""
    print("--- Crop Recommendation System (Command-Line Interface) ---")
    
    # Get user input
    user_features = get_user_input()
    
    # Reshape for a single prediction
    final_features = user_features.reshape(1, -1)
    
    # Scale the features
    scaled_features = scaler.transform(final_features)
    
    # Make a prediction
    prediction = model.predict(scaled_features)
    
    # Get the crop name
    result_crop = CROP_DICT.get(prediction[0], "an unknown crop")
    
    # Display the result
    print("\n-------------------------------------------")
    print(f"✅ Recommended Crop: {result_crop.upper()}")
    print("-------------------------------------------")

if __name__ == '__main__':
    main()
