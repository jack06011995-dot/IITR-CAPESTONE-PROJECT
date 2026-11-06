"""Scoring script: turn a batch of new parcels into an action list.

Loads the packaged model and the hub segments, cleans and featurises the new
parcels using the SAME shared functions used in training (src/shipment_prep.py),
computes a delay probability, flags parcels at/above the stored threshold, joins
the hub segment, and assigns an action. Running twice produces identical output.
"""
import numpy as np
import pandas as pd
import joblib

from src.shipment_prep import clean_shipments, build_features


def main():
    # load inputs and artefacts
    new = pd.read_csv("data/new_shipments.csv")
    hubs = pd.read_csv("data/hubs.csv")
    model = joblib.load("models/delay_model.pkl")
    threshold = float(pd.read_csv("models/delay_threshold.csv")["threshold"].iloc[0])
    feature_order = pd.read_csv("models/feature_order.csv")["feature"].tolist()

    # stored fill medians (the three from clean_shipments)
    iv = pd.read_csv("models/impute_values.csv")
    medians = dict(zip(iv["column"], iv["median"]))

    # clean + featurise using the shared functions (no duplicate cleaning logic)
    clean, _ = clean_shipments(new, medians=medians)
    feats = build_features(clean, hubs)

    # arrange columns in the saved feature order
    X = feats[feature_order]

    # delay probability (positive class), rounded to 4 decimals
    proba = model.predict_proba(X)[:, 1]
    delay_probability = np.round(proba, 4)

    # flag at/above the stored threshold
    flagged = (delay_probability >= threshold).astype(int)

    # join the hub segment
    segments = pd.read_csv("outputs/hub_segments.csv")[["hub_id", "segment"]]
    out = feats[["shipment_id", "hub_id"]].copy()
    out = out.merge(segments, on="hub_id", how="left")
    out["delay_probability"] = delay_probability
    out["flagged"] = flagged

    # assign action
    def action_of(row):
        if row["flagged"] == 1 and row["segment"] == "Escalate":
            return "Escalate to hub manager"
        elif row["flagged"] == 1:
            return "Proactive customer call"
        else:
            return "No action"
    out["action"] = out.apply(action_of, axis=1)

    out = out[["shipment_id", "hub_id", "segment", "delay_probability", "flagged", "action"]]
    out.to_csv("outputs/new_shipment_actions.csv", index=False)

    print("Scoring complete.")
    print(f"  new parcels: {len(out)}")
    print(f"  flagged: {int(out['flagged'].sum())}")
    print(f"  escalate-to-hub-manager: {int((out['action']=='Escalate to hub manager').sum())}")
    print(f"  proactive calls: {int((out['action']=='Proactive customer call').sum())}")
    print(f"  no action: {int((out['action']=='No action').sum())}")


if __name__ == "__main__":
    main()
