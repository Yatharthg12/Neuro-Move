import os
import pickle
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "rep_quality_model.pkl"
)

model = None

# Load model only if it exists
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    print("AI model loaded.")
else:
    print("AI model not found. Running without AI predictions.")


def predict_quality(feature_vector):
    if model is None:
        return "AI not trained"

    prob = model.predict_proba([feature_vector])[0][1]

    if prob > 0.7:
        return "Correct"
    elif prob < 0.3:
        return "Incorrect"
    else:
        return "Uncertain"
