from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "superkart_model.joblib"
MODEL_FEATURES = [
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    "Product_Id_char",
    "Store_Age_Years",
    "Product_Type_Category",
]
NUMERIC_FEATURES = [
    "Product_Weight",
    "Product_Allocated_Area",
    "Product_MRP",
    "Store_Age_Years",
]

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
model = joblib.load(MODEL_PATH)


def validate_records(records):
    if not records:
        raise ValueError("At least one input record is required.")

    frame = pd.DataFrame(records)
    missing = [column for column in MODEL_FEATURES if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    frame = frame[MODEL_FEATURES].copy()
    for column in NUMERIC_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="raise")

    if not np.isfinite(frame[NUMERIC_FEATURES].to_numpy()).all():
        raise ValueError("Numeric values must be finite.")
    if (frame["Product_Weight"] <= 0).any():
        raise ValueError("Product_Weight must be greater than zero.")
    if ((frame["Product_Allocated_Area"] < 0) | (frame["Product_Allocated_Area"] > 1)).any():
        raise ValueError("Product_Allocated_Area must be between 0 and 1.")
    if (frame["Product_MRP"] <= 0).any():
        raise ValueError("Product_MRP must be greater than zero.")
    if (frame["Store_Age_Years"] < 0).any():
        raise ValueError("Store_Age_Years cannot be negative.")
    return frame


@app.get("/")
def home():
    return jsonify({
        "service": "SuperKart Sales Forecasting API",
        "status": "ready",
        "endpoints": ["/health", "/v1/predict", "/v1/predictbatch"],
    })


@app.get("/health")
def health():
    return jsonify({"status": "healthy", "model_loaded": True})


@app.post("/v1/predict")
def predict():
    try:
        payload = request.get_json(silent=False)
        if not isinstance(payload, dict):
            raise ValueError("The JSON body must contain one input object.")
        frame = validate_records([payload])
        prediction = float(model.predict(frame)[0])
        return jsonify({"prediction": round(prediction, 2)})
    except (ValueError, TypeError, KeyError) as error:
        return jsonify({"error": str(error)}), 400


@app.post("/v1/predictbatch")
def predict_batch():
    try:
        if "file" not in request.files:
            raise ValueError("Upload a CSV file using the form field named 'file'.")
        uploaded = request.files["file"]
        if not uploaded.filename.lower().endswith(".csv"):
            raise ValueError("Only CSV files are supported.")
        batch_frame = pd.read_csv(uploaded)
        validated = validate_records(batch_frame.to_dict(orient="records"))
        predictions = model.predict(validated)
        return jsonify({
            "count": int(len(predictions)),
            "predictions": [round(float(value), 2) for value in predictions],
        })
    except (ValueError, TypeError, KeyError, pd.errors.ParserError) as error:
        return jsonify({"error": str(error)}), 400


@app.errorhandler(413)
def file_too_large(_error):
    return jsonify({"error": "Uploaded file exceeds the 5 MB limit."}), 413


if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port, debug=False)
