"""Package the final delay classifier as a reusable artefact.

Reads outputs/part_c_final_config.csv, rebuilds the stratified split, fits the
final Pipeline on the training rows, and saves with joblib:
    models/delay_model.pkl        - the fitted Pipeline (selector + forest)
    models/delay_threshold.csv    - one column: threshold
    models/feature_order.csv       - one column: feature (the 17 FEATURES in order)
    models/impute_values.csv       - columns column, median (the three fill medians)
Then verifies a reload reproduces the in-memory probabilities.
"""
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from src.shipment_prep import clean_shipments, FEATURES

MODEL_PATH = "data/modelling_table.csv"
RAW_PATH = "data/raw_shipments.csv"


def main():
    # read the chosen configuration from Part C
    config = pd.read_csv("outputs/part_c_final_config.csv").iloc[0]
    n_estimators = int(config["n_estimators"])
    max_depth = None if str(config["max_depth"]) == "unlimited" else int(config["max_depth"])
    min_samples_leaf = int(config["min_samples_leaf"])
    threshold = float(config["threshold"])

    # rebuild Part C's stratified split with the same settings
    df = pd.read_csv(MODEL_PATH)
    X = df[FEATURES].copy()
    y = df["is_delayed"].copy()
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # fit the final Pipeline on the training rows
    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, n_estimators=n_estimators,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf)),
    ]).fit(X_train, y_train)

    # the three medians from clean_shipments (fill values for gaps)
    raw = pd.read_csv(RAW_PATH)
    _, medians = clean_shipments(raw)

    # save artefacts
    joblib.dump(final_model, "models/delay_model.pkl")
    pd.DataFrame({"threshold": [threshold]}).to_csv("models/delay_threshold.csv", index=False)
    pd.DataFrame({"feature": FEATURES}).to_csv("models/feature_order.csv", index=False)
    pd.DataFrame({
        "column": ["distance_km", "parcel_weight_kg", "declared_value_inr"],
        "median": [round(medians["distance_km"], 4), round(medians["parcel_weight_kg"], 4),
                   round(medians["declared_value_inr"], 4)],
    }).to_csv("models/impute_values.csv", index=False)

    # reload check: predict_proba for the 180 test rows, in-memory vs reloaded
    reloaded = joblib.load("models/delay_model.pkl")
    proba_inmem = final_model.predict_proba(X_test)[:, 1]
    proba_reloaded = reloaded.predict_proba(X_test)[:, 1]
    reload_diff = float(np.max(np.abs(proba_inmem - proba_reloaded)))

    pd.DataFrame([
        ("test_rows_scored", int(len(X_test))),
        ("reload_max_abs_probability_diff", round(reload_diff, 10)),
        ("threshold", threshold),
    ], columns=["metric", "value"]).to_csv("outputs/part_d_reload_check.csv", index=False)

    print("Package model complete.")
    print(f"  test_rows_scored: {len(X_test)}")
    print(f"  reload_max_abs_probability_diff: {reload_diff}")
    print(f"  threshold: {threshold}")


if __name__ == "__main__":
    main()
