from flask_cors import CORS
CORS()

from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Initialize Flask app
app = Flask(__name__)

# Load and preprocess data
data = pd.read_csv('D:/xss/Data_66_featurs.csv')
X = data.drop(columns=['Label'])  
y = data['Label']  

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)

# Print accuracy and classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Function to preprocess user input
def preprocess_user_input(user_input):
    features = np.zeros(X.shape[1])  # Ensure it matches the feature size

    # Example preprocessing logic (adjust based on actual features):
    features[0] = len(user_input)  # `url_length`
    features[1] = sum(1 for char in user_input if char in "!@#$%^&*()_+")  # `url_special_characters`
    features[2] = user_input.lower().count("<script>")  # `url_tag_script`
    features[3] = user_input.lower().count("<iframe>")  # `url_tag_iframe`
    features[-1] = len(user_input) * 5  # Example for `html_length`

    return features

# Route for XSS vulnerability prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.json
        user_input = data.get('user_input', '')

        # Preprocess user input
        user_input_features = preprocess_user_input(user_input)

        # Convert to a DataFrame with column names matching the training data
        user_input_df = pd.DataFrame([user_input_features], columns=X.columns)

        # Scale the features
        user_input_scaled = scaler.transform(user_input_df)

        # Predict using the trained model
        prediction = model.predict(user_input_scaled)
        result = "Yes" if prediction[0] == 1 else "No"

        # Return the result as JSON
        return jsonify({"prediction": result})

    except Exception as e:
        return jsonify({"error": str(e)})

# Route for health check
@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "API is running"})

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True)
