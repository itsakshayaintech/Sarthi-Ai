import joblib
import pandas as pd
import numpy as np


MODEL_PATH = "models/fusion_model.pkl"


# Load model
package = joblib.load(MODEL_PATH)

model = package["model"]
features = package["features"]


def predict_fusion(
    indicator_svi=None,
    text_svi=None,
    voice_svi=None,
    context_svi=None
):

    values = {
        "indicator_svi": indicator_svi,
        "text_svi": text_svi,
        "voice_svi": voice_svi,
        "context_svi": context_svi
    }

    row = {}

    for feature in features:

        value = values.get(feature)

        if value is None:
            value = 0.0

        row[feature] = value

    X = pd.DataFrame(
        [row],
        columns=features
    )

    prediction = model.predict(X)[0]

    prediction = float(
        np.clip(prediction, 0, 100)
    )

    return round(prediction, 2)
