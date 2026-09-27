import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

# -------------------------------------------------

# CONFIGURATION

# -------------------------------------------------

DATA_DIR = "."
MODEL_DIR = "models"

os.makedirs(MODEL_DIR, exist_ok=True)

def evaluate_model(name, model, X_test, y_test):
    """
    Evaluates a regression model safely.
    """

    predictions = model.predict(X_test)

    predictions = np.clip(predictions, 0, 100)

    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n" + "=" * 50)
    print(f"{name} MODEL RESULTS")
    print("=" * 50)
    print(f"MAE: {mae:.2f}")
    print(f"R2 Score: {r2:.2f}")

    return model
 

# -------------------------------------------------
# 1. SVI INDICATOR MODEL
# -------------------------------------------------

def train_indicator_model():
    print("\nTraining SVI Indicator Model...")

    file_path = os.path.join(
        DATA_DIR,
        "nhaa_svi_training_examples.csv"
    )

    df = pd.read_csv(file_path)

    feature_columns = [
        "fear",
        "threat",
        "anxiety",
        "trauma",
        "social_isolation",
        "displacement",
        "legal_stress",
        "safety_concern"
    ]

    X = df[feature_columns]
    y = df["svi_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=42,
        max_depth=10
    )

    model.fit(X_train, y_train)

    evaluate_model(
        "SVI INDICATOR",
        model,
        X_test,
        y_test
    )

    joblib.dump(
        {
            "model": model,
            "features": feature_columns
        },
        os.path.join(
            MODEL_DIR,
            "indicator_model.pkl"
        )
    )

    print("Saved: models/indicator_model.pkl")
 

# -------------------------------------------------

# 2. MULTILINGUAL TEXT MODEL

# -------------------------------------------------

def train_text_model():

 
    print("\nTraining Text Assessment Model...")

file_path = os.path.join(
    DATA_DIR,
    "nhaa_synthetic_multilingual_text.csv"
)

df = pd.read_csv(file_path)

X = df["text"].fillna("")
y = df["svi_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    (
        "tfidf",
        TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2)
        )
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            max_depth=15
        )
    )
])

model.fit(X_train, y_train)

evaluate_model(
    "TEXT",
    model,
    X_test,
    y_test
)

joblib.dump(
    model,
    os.path.join(
        MODEL_DIR,
        "text_model.pkl"
    )
)

print("Saved: models/text_model.pkl")
 

# -------------------------------------------------

# 3. VOICE FEATURE MODEL

# -------------------------------------------------

def train_voice_model():

 
    print("\nTraining Voice Feature Model...")

file_path = os.path.join(
    DATA_DIR,
    "nhaa_synthetic_voice_features.csv"
)

df = pd.read_csv(file_path)

numeric_features = [
    "speech_rate_words_per_sec",
    "avg_pause_seconds",
    "pitch_variability_hz",
    "normalized_energy",
    "hesitation_ratio",
    "speech_duration_sec"
]

X = df[numeric_features]
y = df["svi_score"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    ),
    (
        "scaler",
        StandardScaler()
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=12
        )
    )
])

model.fit(X_train, y_train)

evaluate_model(
    "VOICE",
    model,
    X_test,
    y_test
)

joblib.dump(
    {
        "model": model,
        "features": numeric_features
    },
    os.path.join(
        MODEL_DIR,
        "voice_model.pkl"
    )
)

print("Saved: models/voice_model.pkl")
 

# -------------------------------------------------

# 4. CASE CONTEXT MODEL

# -------------------------------------------------

def train_context_model():

 
    print("\nTraining Case Context Model...")

file_path = os.path.join(
    DATA_DIR,
    "nhaa_synthetic_case_context.csv"
)

df = pd.read_csv(file_path)

feature_columns = [
    "state_ut",
    "victim_group",
    "reported_context",
    "immediate_safety_concern",
    "social_isolation_indicator",
    "displacement_indicator",
    "legal_delay_indicator"
]

X = df[feature_columns]
y = df["svi_score"]

categorical_features = [
    "state_ut",
    "victim_group",
    "reported_context"
]

numeric_features = [
    "immediate_safety_concern",
    "social_isolation_indicator",
    "displacement_indicator",
    "legal_delay_indicator"
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            categorical_features
        ),
        (
            "numeric",
            Pipeline([
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    )
                )
            ]),
            numeric_features
        )
    ]
)

model = Pipeline([
    (
        "preprocessor",
        preprocessor
    ),
    (
        "regressor",
        RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            max_depth=12
        )
    )
])

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

model.fit(X_train, y_train)

evaluate_model(
    "CASE CONTEXT",
    model,
    X_test,
    y_test
)

joblib.dump(
    {
        "model": model,
        "features": feature_columns
    },
    os.path.join(
        MODEL_DIR,
        "context_model.pkl"
    )
)

print("Saved: models/context_model.pkl")
 

# -------------------------------------------------
# MAIN TRAINING
# -------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("SARTHI-AI MODEL TRAINING STARTED")
    print("=" * 60)

    train_indicator_model()
    train_text_model()
    train_voice_model()
    train_context_model()

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)