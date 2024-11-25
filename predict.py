from flask import Flask, request, jsonify
import pandas as pd
import joblib

# Load the trained model
MODEL_PATH = "best_model.pkl"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict delays given input features.
    The API expects a JSON payload with feature values.
    Example JSON format:
    {
        "Unnamed: 0": 0, 
        "Month": 1,
        "DayOfWeek": 4,
        "DepTime": 1829,
        "CRSDepTime": 1755,
        "ArrTime": 1959,
        "CRSArrTime": 1925,
        "UniqueCarrier": "WN",
        "TailNum": "N464WN",
        "ActualElapsedTime": 9.48683298050514,
        "Origin": "IND",
        "Dest": "BWI",
        "TaxiIn": 1.73205080756888,
        "TaxiOut": 3.16227766016838
    }
    """
    # Parse input JSON
    data = request.get_json()

    # Convert JSON into DataFrame
    df = pd.DataFrame([data])

    # Make predictions
    predictions = model.predict(df)
    result = {"prediction": predictions.tolist()}
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
