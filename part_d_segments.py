"""Part D - Hub Segments, Packaging and Operations Playbook.

Clusters the 24 hubs into three operating segments, finds hubs whose tier label
is misleading, and writes the segment/mismatch tables used by the scoring script.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

MODEL_PATH = "data/modelling_table.csv"
HUBS_PATH = "data/hubs.csv"

PROFILE_COLS = ["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]
TIER_TO_SEGMENT = {1: "Standard", 2: "Watch", 3: "Escalate"}


def separation_ratio(scaled, labels):
    """Mean across-cluster pair distance / mean within-cluster pair distance."""
    n = len(scaled)
    across, within = [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(scaled[i] - scaled[j]))
            if labels[i] == labels[j]:
                within.append(d)
            else:
                across.append(d)
    across_mean = np.mean(across) if across else 0.0
    within_mean = np.mean(within) if within else 0.0
    return round(across_mean / within_mean, 4) if within_mean else 0.0


def cluster_sizes_str(labels):
    sizes = sorted(pd.Series(labels).value_counts().tolist(), reverse=True)
    return "-".join(str(s) for s in sizes)


def main():
    modelling = pd.read_csv(MODEL_PATH)
    hubs = pd.read_csv(HUBS_PATH)

    # 1. hub profile (group by hub_id, sorted ascending)
    profile = modelling.groupby("hub_id").agg(
        shipments=("hub_id", "count"),
        mean_delivery_hours=("delivery_hours", "mean"),
        delay_rate=("is_delayed", "mean"),
        mean_distance_km=("distance_km", "mean"),
        mean_n_stops=("n_stops", "mean"),
    ).reset_index()
    profile = profile.sort_values("hub_id").reset_index(drop=True)
    profile[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]] = (
        profile[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]].round(4)
    )
    profile.to_csv("outputs/hub_profile.csv", index=False)

    # 2. scale + K-Means (hubs in ascending hub_id order)
    X = profile[PROFILE_COLS].values
    scaled = StandardScaler().fit_transform(X)

    # 3. separation evidence for K in 2,3,4,5
    sep_rows = []
    k3_labels = None
    for k in [2, 3, 4, 5]:
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(scaled)
        labels = km.labels_
        sep_rows.append([k, separation_ratio(scaled, labels), cluster_sizes_str(labels)])
        if k == 3:
            k3_labels = labels
    pd.DataFrame(sep_rows, columns=["K", "across_over_within", "cluster_sizes"]).to_csv(
        "outputs/part_d_separation.csv", index=False)

    # 4. segment names: order 3 clusters by mean delay_rate
    profile["cluster"] = k3_labels
    cluster_delay = profile.groupby("cluster")["delay_rate"].mean().sort_values()
    cluster_to_segment = {c: name for c, name in zip(
        cluster_delay.index, ["Standard", "Watch", "Escalate"])}
    profile["segment"] = profile["cluster"].map(cluster_to_segment)

    # hub_segments.csv
    segments = profile[["hub_id", "segment"]].merge(hubs[["hub_id", "hub_city", "hub_tier"]],
                                                     on="hub_id", how="left")
    segments = segments[["hub_id", "segment", "hub_city", "hub_tier"]]
    segments.to_csv("outputs/hub_segments.csv", index=False)

    # segment_profile.csv (means of hub-level values)
    seg_prof = profile.groupby("segment").agg(
        hubs=("hub_id", "count"),
        mean_delivery_hours=("mean_delivery_hours", "mean"),
        delay_rate=("delay_rate", "mean"),
        mean_distance_km=("mean_distance_km", "mean"),
        mean_n_stops=("mean_n_stops", "mean"),
    ).reset_index()
    # order Standard, Watch, Escalate
    seg_order = ["Standard", "Watch", "Escalate"]
    seg_prof["__o"] = seg_prof["segment"].map({n: i for i, n in enumerate(seg_order)})
    seg_prof = seg_prof.sort_values("__o").drop(columns="__o")
    seg_prof[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]] = (
        seg_prof[["mean_delivery_hours", "delay_rate", "mean_distance_km", "mean_n_stops"]].round(4)
    )
    seg_prof.to_csv("outputs/segment_profile.csv", index=False)

    # tier_vs_segment_mismatch.csv
    segments["tier_segment"] = segments["hub_tier"].map(TIER_TO_SEGMENT)
    mismatch = segments[segments["segment"] != segments["tier_segment"]][
        ["hub_id", "segment", "hub_city", "hub_tier"]]
    mismatch.to_csv("outputs/tier_vs_segment_mismatch.csv", index=False)

    print("Part D complete.")
    print("  separation:"); 
    print(pd.read_csv("outputs/part_d_separation.csv").to_string(index=False))
    print("  segment profile:"); print(seg_prof.to_string(index=False))
    print("  mismatches:"); print(mismatch.to_string(index=False))


if __name__ == "__main__":
    main()
