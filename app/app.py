from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "fraud_model.pkl")

# Load model pipeline
model = joblib.load(MODEL_PATH)

# Feature column order expected by preprocessor
FEATURE_NAMES = [
    "Time", "V1", "V2", "V3", "V4", "V5", "V6", "V7", "V8", "V9", "V10",
    "V11", "V12", "V13", "V14", "V15", "V16", "V17", "V18", "V19", "V20",
    "V21", "V22", "V23", "V24", "V25", "V26", "V27", "V28", "Amount"
]

# Sample transactions from real creditcard.csv dataset
SAMPLE_TRANSACTIONS = {
    "legitimate": {
        "Time": 0.0, "V1": -1.359807, "V2": -0.072781, "V3": 2.536347, "V4": 1.378155,
        "V5": -0.338321, "V6": 0.462388, "V7": 0.239599, "V8": 0.098698, "V9": 0.363787,
        "V10": 0.090794, "V11": -0.551600, "V12": -0.617801, "V13": -0.991390, "V14": -0.311169,
        "V15": 1.468177, "V16": -0.470401, "V17": 0.207971, "V18": 0.025791, "V19": 0.403993,
        "V20": 0.251412, "V21": -0.018307, "V22": 0.277838, "V23": -0.110474, "V24": 0.066928,
        "V25": 0.128539, "V26": -0.189115, "V27": 0.133558, "V28": -0.021053, "Amount": 149.62
    },
    "fraudulent": {
        "Time": 406.0, "V1": -2.312227, "V2": 1.951992, "V3": -1.609851, "V4": 3.997906,
        "V5": -0.522188, "V6": -1.426545, "V7": -2.537387, "V8": 1.391657, "V9": -2.770089,
        "V10": -2.772272, "V11": 3.202033, "V12": -2.899907, "V13": -0.595222, "V14": -4.289254,
        "V15": 0.389724, "V16": -1.140747, "V17": -2.830056, "V18": -0.016822, "V19": 0.416956,
        "V20": 0.126911, "V21": 0.517232, "V22": -0.035049, "V23": -0.465211, "V24": 0.320198,
        "V25": 0.044519, "V26": 0.177840, "V27": 0.261145, "V28": -0.143276, "Amount": 0.00
    }
}

@app.route("/")
def home():
    return render_template("index.html", feature_names=FEATURE_NAMES, form_data=SAMPLE_TRANSACTIONS["legitimate"])

@app.route("/api/sample/<sample_type>")
def get_sample(sample_type):
    if sample_type in SAMPLE_TRANSACTIONS:
        return jsonify({"status": "success", "data": SAMPLE_TRANSACTIONS[sample_type]})
    return jsonify({"status": "error", "message": "Sample type not found"}), 404

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {}
        for feature in FEATURE_NAMES:
            val = request.form.get(feature, 0.0)
            data[feature] = float(val) if val != "" else 0.0

        # Create single-row DataFrame for pipeline input
        input_df = pd.DataFrame([data], columns=FEATURE_NAMES)

        # Make prediction and compute probability
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0] if hasattr(model, "predict_proba") else [0.5, 0.5]
        
        fraud_prob = round(float(probabilities[1]) * 100, 2)
        legit_prob = round(float(probabilities[0]) * 100, 2)

        is_fraud = bool(prediction == 1)

        result_payload = {
            "is_fraud": is_fraud,
            "result_title": "Fraudulent Transaction Detected 🚨" if is_fraud else "Legitimate Transaction Verified ✅",
            "fraud_probability": fraud_prob,
            "legit_probability": legit_prob,
            "risk_level": "HIGH RISK" if is_fraud else "LOW RISK"
        }

        return render_template(
            "index.html",
            feature_names=FEATURE_NAMES,
            form_data=data,
            result=result_payload
        )

    except Exception as e:
        error_payload = {
            "is_fraud": False,
            "result_title": f"Prediction Error: {str(e)}",
            "fraud_probability": 0.0,
            "legit_probability": 0.0,
            "risk_level": "ERROR"
        }
        return render_template(
            "index.html",
            feature_names=FEATURE_NAMES,
            form_data=request.form.to_dict(),
            result=error_payload
        )

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)