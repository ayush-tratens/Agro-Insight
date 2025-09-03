import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pickle

print("Script started...")

# 1. Load Data
try:
    crop = pd.read_csv("Crop_recommendation.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Error: Crop_recommendation.csv not found. Make sure it's in the same folder.")
    exit()

# 2. Preprocess Data
crop_dict = {
    'rice': 1, 'maize': 2, 'jute': 3, 'cotton': 4, 'coconut': 5, 'papaya': 6,
    'orange': 7, 'apple': 8, 'muskmelon': 9, 'watermelon': 10, 'grapes': 11,
    'mango': 12, 'banana': 13, 'pomegranate': 14, 'lentil': 15, 'blackgram': 16,
    'mungbean': 17, 'mothbeans': 18, 'pigeonpeas': 19, 'kidneybeans': 20,
    'chickpea': 21, 'coffee': 22
}
crop['crop_num'] = crop['label'].map(crop_dict)
crop.drop(['label'], axis=1, inplace=True)
print("Data preprocessing complete.")

# 3. Split Data
X = crop.drop(['crop_num'], axis=1)
y = crop['crop_num']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Train-test split complete.")

# 4. Scale Features
ms = MinMaxScaler()
X_train_scaled = ms.fit_transform(X_train)
# We save the scaler that has been fit on the training data
print("Feature scaling complete.")

# 5. Train Model
rfc = RandomForestClassifier(random_state=42)
rfc.fit(X_train_scaled, y_train)
print("Model training complete.")

# 6. Save Model and Scaler
pickle.dump(rfc, open('model.pkl', 'wb'))
pickle.dump(ms, open('minmaxscaler.pkl', 'wb'))
print("✅ Model and scaler have been saved successfully as 'model.pkl' and 'minmaxscaler.pkl'")