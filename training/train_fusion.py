import os
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# PATHS
# --------------------------------------------------

DATA_PATH = "data/fusion_data.csv"
MODEL_PATH = "models/fusion_model.pkl"


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

print("\nFusion Dataset")
print("----------------")
print("Shape:", df.shape)
print(df.head())


# --------------------------------------------------
# FEATURES
# --------------------------------------------------

features = [
    "indicator_svi",
    "text_svi",
    "voice_svi",
    "context_svi"
]

target = "final_svi"


# Check columns
missing_columns = [
    column for column in features + [target]
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in fusion_data.csv: {missing_columns}"
    )


# --------------------------------------------------
# DATA
# --------------------------------------------------

X = df[features]
y = df[target]


# --------------------------------------------------
# TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


# --------------------------------------------------
# MODEL
# --------------------------------------------------

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=10,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)


# --------------------------------------------------
# TRAIN
# --------------------------------------------------

model.fit(X_train, y_train)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = mean_squared_error(
    y_test,
    predictions
) ** 0.5

r2 = r2_score(
    y_test,
    predictions
)


print("\nFusion Model Performance")
print("-------------------------")
print("MAE :", round(mae, 3))
print("RMSE:", round(rmse, 3))
print("R²  :", round(r2, 3))


# --------------------------------------------------
# FEATURE IMPORTANCE
# --------------------------------------------------

print("\nFeature Importance")
print("------------------")

for feature, importance in zip(
    features,
    model.feature_importances_
):
    print(
        f"{feature}: {importance:.4f}"
    )


# --------------------------------------------------
# SAVE
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_package = {
    "model": model,
    "features": features
}

joblib.dump(
    model_package,
    MODEL_PATH
)

print("\nFusion model saved to:")
print(MODEL_PATH)
