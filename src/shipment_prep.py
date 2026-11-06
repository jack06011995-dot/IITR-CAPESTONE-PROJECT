"""Shared constants, cleaning and feature building for the Delay Intelligence pipeline.

Two functions live here so the same preprocessing is used in training (Part A/B/C/D)
and at scoring time (predict_delay.py):
    clean_shipments(raw, medians=None) -> (clean_df, medians)
    build_features(clean, hubs)        -> features_df
"""
import numpy as np
import pandas as pd

# Promise windows (hours) by service type.
PROMISED_HOURS = {"Express": 36, "Standard": 60, "Economy": 96}

# The 17 model inputs, pinned in exactly this order.
FEATURES = [
    "promised_hours", "distance_km", "parcel_weight_kg", "log_declared_value", "n_stops",
    "rain_flag", "sale_event_flag", "is_cod", "is_weekend", "dispatch_hour",
    "hub_tier", "hub_daily_capacity",
    "label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no",
]

# Bookkeeping columns that carry no real signal about delay.
NOISE_COLUMNS = ["label_print_batch", "scanner_build_no", "tape_roll_no", "sticker_serial_mod", "shelf_slot_no"]


def clean_shipments(raw, medians=None):
    """Clean a raw shipments DataFrame.

    Returns (clean_df, medians). Steps applied in this exact order:
      1. drop exact duplicate rows
      2. service_type: strip + title case (-> Express, Standard, Economy)
      3. payment_mode: missing -> "Unknown"
      4. declared_value: strip rupee sign/commas -> declared_value_inr (numeric), drop text col
      5. dispatch_time: convert to datetime
      6. parcel_weight_kg: values > 100 are scanner-fault sentinels -> missing
      7. fill missing distance_km / parcel_weight_kg / declared_value_inr with column medians
    If medians is None the three medians are computed from the non-missing values
    left after steps 1-6; otherwise the supplied dict is used. The returned medians
    dict has exactly the keys distance_km, parcel_weight_kg, declared_value_inr.
    """
    df = raw.copy()

    # 1. drop exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. service_type: strip leading/trailing spaces and title case
    df["service_type"] = df["service_type"].astype(str).str.strip().str.title()

    # 3. payment_mode: missing -> "Unknown"
    df["payment_mode"] = df["payment_mode"].fillna("Unknown")

    # 4. declared_value: remove rupee sign and thousands commas -> declared_value_inr
    s = (df["declared_value"]
         .astype(str)
         .str.replace("\u20b9", "", regex=False)
         .str.replace(",", "", regex=False))
    df["declared_value_inr"] = pd.to_numeric(s, errors="coerce")
    df = df.drop(columns=["declared_value"])

    # 5. dispatch_time: convert to datetime
    df["dispatch_time"] = pd.to_datetime(df["dispatch_time"])

    # 6. parcel_weight_kg: scanner-fault sentinel (> 100 kg) -> missing
    df.loc[df["parcel_weight_kg"] > 100, "parcel_weight_kg"] = np.nan

    # 7. fill missing distance_km / parcel_weight_kg / declared_value_inr with medians
    fill_cols = ["distance_km", "parcel_weight_kg", "declared_value_inr"]
    if medians is None:
        medians = {c: float(df[c].median()) for c in fill_cols}
    for c in fill_cols:
        df[c] = df[c].fillna(medians[c])

    return df, medians


def build_features(clean, hubs):
    """Attach hub columns (left merge on hub_id) and add derived feature columns.

    Does not use delivery_hours, so it also works on parcels that have not been
    delivered yet. Returns the enriched DataFrame.
    """
    df = clean.copy()

    # left merge on hub_id -> hub_city, region, hub_tier, hub_daily_capacity
    df = df.merge(hubs, on="hub_id", how="left")

    # promised_hours from PROMISED_HOURS (integer)
    df["promised_hours"] = df["service_type"].map(PROMISED_HOURS).astype(int)

    # is_cod: 1 when payment_mode is COD, else 0 (Unknown -> 0)
    df["is_cod"] = (df["payment_mode"] == "COD").astype(int)

    # is_weekend: 1 when dispatch day is Saturday or Sunday
    df["is_weekend"] = (df["dispatch_time"].dt.dayofweek >= 5).astype(int)

    # dispatch_hour: hour of dispatch timestamp as integer
    df["dispatch_hour"] = df["dispatch_time"].dt.hour.astype(int)

    # log_declared_value: natural log of declared_value_inr
    df["log_declared_value"] = np.log(df["declared_value_inr"])

    return df
