"""Part A - Data Foundation and Delay Evidence.

Reads data/raw_shipments.csv and data/hubs.csv, cleans the raw export, builds
features, writes the modelling table, and produces the delay-evidence outputs:
    data/modelling_table.csv
    outputs/part_a_checks.csv
    outputs/part_a_delay_by_group.csv
    outputs/part_a_correlations.csv
"""
import pandas as pd
import numpy as np

from src.shipment_prep import (
    clean_shipments, build_features, FEATURES, PROMISED_HOURS, NOISE_COLUMNS,
)

RAW_PATH = "data/raw_shipments.csv"
HUBS_PATH = "data/hubs.csv"
MODELLING_PATH = "data/modelling_table.csv"


def round4(x):
    """Round a scalar to 4 decimals; leave integers as integers."""
    if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
        return int(x)
    return round(float(x), 4)


def main():
    raw = pd.read_csv(RAW_PATH)
    hubs = pd.read_csv(HUBS_PATH)

    # ---- "before" state: after removing duplicates, before any filling ----
    raw_rows = len(raw)
    dedup = raw.drop_duplicates()
    rows_after_dedup = len(dedup)
    duplicate_rows_removed = raw_rows - rows_after_dedup

    missing_distance_km_before = int(dedup["distance_km"].isna().sum())
    missing_parcel_weight_before = int(dedup["parcel_weight_kg"].isna().sum())  # NaN only, not sentinels
    sentinel_weight_rows = int((dedup["parcel_weight_kg"] > 100).sum())
    missing_declared_value_before = int(dedup["declared_value"].isna().sum())
    missing_payment_mode_before = int(dedup["payment_mode"].isna().sum())
    service_type_spellings_before = int(dedup["service_type"].nunique())
    service_type_spellings_after = int(dedup["service_type"].astype(str).str.strip().str.title().nunique())

    # ---- clean + build features ----
    clean, medians = clean_shipments(raw)
    modelling = build_features(clean, hubs)

    # target: is_delayed = 1 when delivery_hours > promised_hours
    modelling["is_delayed"] = (modelling["delivery_hours"] > modelling["promised_hours"]).astype(int)

    # write the modelling table (data/, full precision, index=False)
    modelling.to_csv(MODELLING_PATH, index=False)

    rows_after_hub_merge = len(modelling)
    nulls_in_features_and_target = int(modelling[FEATURES + ["delivery_hours", "is_delayed"]].isna().sum().sum())
    delayed_count = int(modelling["is_delayed"].sum())
    delay_rate = round4(delayed_count / rows_after_hub_merge)

    # ---- delivery_hours quartiles + 1.5*IQR outlier rule (unrounded, round on write) ----
    q1 = modelling["delivery_hours"].quantile(0.25)
    q3 = modelling["delivery_hours"].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    hours_outlier_count = int(((modelling["delivery_hours"] < lower_bound) |
                               (modelling["delivery_hours"] > upper_bound)).sum())

    # ---- declared-value metrics on the median-filled column of the modelling table ----
    declared_value_mean_inr = round4(modelling["declared_value_inr"].mean())
    declared_value_median_inr = round4(modelling["declared_value_inr"].median())
    log_declared_value_mean = round4(modelling["log_declared_value"].mean())
    log_declared_value_median = round4(modelling["log_declared_value"].median())

    # ---- part_a_checks.csv ----
    checks = [
        ("raw_rows", int(raw_rows)),
        ("duplicate_rows_removed", int(duplicate_rows_removed)),
        ("rows_after_dedup", int(rows_after_dedup)),
        ("missing_distance_km_before", missing_distance_km_before),
        ("missing_parcel_weight_before", missing_parcel_weight_before),
        ("sentinel_weight_rows", sentinel_weight_rows),
        ("missing_declared_value_before", missing_declared_value_before),
        ("missing_payment_mode_before", missing_payment_mode_before),
        ("service_type_spellings_before", service_type_spellings_before),
        ("service_type_spellings_after", service_type_spellings_after),
        ("median_distance_km", round4(medians["distance_km"])),
        ("median_parcel_weight_kg", round4(medians["parcel_weight_kg"])),
        ("median_declared_value_inr", round4(medians["declared_value_inr"])),
        ("rows_after_hub_merge", int(rows_after_hub_merge)),
        ("nulls_in_features_and_target", nulls_in_features_and_target),
        ("delayed_count", delayed_count),
        ("delay_rate", delay_rate),
        ("hours_q1", round4(q1)),
        ("hours_q3", round4(q3)),
        ("hours_lower_bound", round4(lower_bound)),
        ("hours_upper_bound", round4(upper_bound)),
        ("hours_outlier_count", hours_outlier_count),
        ("declared_value_mean_inr", declared_value_mean_inr),
        ("declared_value_median_inr", declared_value_median_inr),
        ("log_declared_value_mean", log_declared_value_mean),
        ("log_declared_value_median", log_declared_value_median),
    ]
    pd.DataFrame(checks, columns=["metric", "value"]).to_csv("outputs/part_a_checks.csv", index=False)

    # ---- part_a_delay_by_group.csv ----
    rows = []

    # service_type (alphabetical: Economy, Express, Standard)
    for grp in sorted(modelling["service_type"].unique()):
        sub = modelling[modelling["service_type"] == grp]
        rows.append(("service_type", grp, len(sub), int(sub["is_delayed"].sum()),
                     round4(sub["is_delayed"].mean()), round4(sub["delivery_hours"].mean())))

    # hub_tier (ascending 1, 2, 3)
    for grp in sorted(modelling["hub_tier"].unique()):
        sub = modelling[modelling["hub_tier"] == grp]
        rows.append(("hub_tier", str(grp), len(sub), int(sub["is_delayed"].sum()),
                     round4(sub["is_delayed"].mean()), round4(sub["delivery_hours"].mean())))

    # distance_band (0-200, 201-400, 401-600, 601+ ; right edge inclusive)
    modelling["distance_band"] = pd.cut(
        modelling["distance_km"],
        bins=[0, 200, 400, 600, 2000],
        labels=["0-200", "201-400", "401-600", "601+"],
        right=True, include_lowest=True,
    )
    for grp in ["0-200", "201-400", "401-600", "601+"]:
        sub = modelling[modelling["distance_band"] == grp]
        rows.append(("distance_band", grp, len(sub), int(sub["is_delayed"].sum()),
                     round4(sub["is_delayed"].mean()), round4(sub["delivery_hours"].mean())))

    pd.DataFrame(rows, columns=["dimension", "group", "shipments", "delayed",
                                "delay_rate", "mean_hours"]).to_csv(
        "outputs/part_a_delay_by_group.csv", index=False)

    # ---- part_a_correlations.csv ----
    num_cols = modelling.select_dtypes(include="number").columns.tolist()
    corr = modelling[num_cols].corr().round(4)
    corr.to_csv("outputs/part_a_correlations.csv", index=True)

    print("Part A complete.")
    print(f"  modelling_table rows: {len(modelling)}")
    print(f"  delayed: {delayed_count} ({delay_rate})")
    print(f"  medians: {medians}")


if __name__ == "__main__":
    main()
