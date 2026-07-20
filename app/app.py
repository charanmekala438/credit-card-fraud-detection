from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "..",
    "model",
    "fraud_model.pkl"
)

model = joblib.load(MODEL_PATH)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    try:

        values = []

        for i in range(1, 31):
            values.append(
                float(
                    request.form[f"feature{i}"]
                )
            )

        features = np.array(values).reshape(1, -1)

        prediction = model.predict(features)

        if prediction[0] == 1:
            result = "Fraudulent Transaction 🚨"
        else:
            result = "Legitimate Transaction ✅"

        return render_template(
            "index.html",
            prediction_text=result
        )

    except Exception as e:

        return render_template(
            "index.html",
            prediction_text=str(e)
        )

if __name__ == "__main__":
    app.run(debug=True)