import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

data = pd.read_csv('D:/xss/Data_66_featurs.csv')
X = data.drop(columns=['Label'])  
y = data['Label']  
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)  
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
def preprocess_user_input(user_input):
   
    features = np.zeros(67) 

 
    features[0] = len(user_input)  # `url_length`
    features[1] = sum(1 for char in user_input if char in "!@#$%^&*()_+")  # `url_special_characters`
    features[2] = user_input.lower().count("<script>")  # `url_tag_script`
    features[3] = user_input.lower().count("<iframe>")  # `url_tag_iframe`
    features[-1] = len(user_input) * 5  # hawa logic for `html_length` 

  
    return features


def predict_user_input(user_input):
    # Preprocess user input
    user_input_features = preprocess_user_input(user_input)

    # Convert to a DataFrame with column names matching the training data
    user_input_df = pd.DataFrame([user_input_features], columns=X.columns)

    # Scale the features
    user_input_scaled = scaler.transform(user_input_df)

    # train_model_use(predition)
    prediction = model.predict(user_input_scaled)
    return "Yes" if prediction[0] == 1 else "No"


while True:
   
    user_input = input("Enter a text/URL to check for XSS vulnerability (or type 'exit' to quit): ")
    
   
    if user_input.lower() == 'exit':
        print("Exiting...")
        break
    

    result = predict_user_input(user_input)
    print(f"Prediction: {result}")
