"""Part C - Delay Classifier and Alert Threshold.

Reads data/modelling_table.csv, builds a delay classifier (target is_delayed),
removes the bookkeeping noise columns via SelectKBest, tunes a random forest,
and chooses an alert probability threshold matching the operations rule.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif

from src.shipment_prep import FEATURES, NOISE_COLUMNS

MODEL_PATH = "data/modelling_table.csv"


def cm_metrics(y_true, y_pred):
    """Return TN, FP, FN, TP, accuracy, precision, recall, f1 (rounded to 4)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    total = tn + fp + fn + tp
    acc = (tp + tn) / total if total else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tn, fp, fn, tp, round(acc, 4), round(prec, 4), round(rec, 4), round(f1, 4)


def fmt_params(p):
    """Format a params dict as a compact string for a CSV cell."""
    return ", ".join(f"{k}={v}" for k, v in p.items())


def main():
    df = pd.read_csv(MODEL_PATH)
    X = df[FEATURES].copy()
    y = df["is_delayed"].copy()

    # 1. stratified split: 840/180/180, delayed 195/42/41
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)
    assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)
    print(f"  delayed train/val/test: {int(y_train.sum())}/{int(y_val.sum())}/{int(y_test.sum())}")

    # 2. blind baseline + three classifiers (default 0.5 rule) on validation rows
    rows = []

    # blind: predict "not delayed" for every parcel
    tn, fp, fn, tp, a, p, r, f1 = cm_metrics(y_val, np.zeros(len(y_val), dtype=int))
    rows.append(["blind_all_on_time", tn, fp, fn, tp, a, p, r, f1])

    # logistic regression
    lr = Pipeline([("scaler", StandardScaler()), ("model", LogisticRegression())]).fit(X_train, y_train)
    tn, fp, fn, tp, a, p, r, f1 = cm_metrics(y_val, lr.predict(X_val))
    rows.append(["logistic_regression", tn, fp, fn, tp, a, p, r, f1])

    # decision tree: choose depth by highest validation F1 (ties -> smaller d)
    best_d, best_f1 = None, -1.0
    for d in [2, 3, 4, 5, 6, 8, 10]:
        m = DecisionTreeClassifier(criterion="gini", max_depth=d, random_state=42).fit(X_train, y_train)
        _, _, _, _, _, _, _, f1 = cm_metrics(y_val, m.predict(X_val))
        if f1 > best_f1 + 1e-12:
            best_f1 = f1
            best_d = d
    dt = DecisionTreeClassifier(criterion="gini", max_depth=best_d, random_state=42).fit(X_train, y_train)
    tn, fp, fn, tp, a, p, r, f1 = cm_metrics(y_val, dt.predict(X_val))
    rows.append([f"decision_tree_depth_{best_d}", tn, fp, fn, tp, a, p, r, f1])

    # random forest default
    rf_def = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    tn, fp, fn, tp, a, p, r, f1 = cm_metrics(y_val, rf_def.predict(X_val))
    rows.append(["random_forest_default", tn, fp, fn, tp, a, p, r, f1])

    pd.DataFrame(rows, columns=["model", "TN", "FP", "FN", "TP", "accuracy",
                                 "precision", "recall", "f1"]).to_csv(
        "outputs/part_c_models.csv", index=False)

    # 3. feature signal: forest importance + SelectKBest F-scores
    forest17 = RandomForestClassifier(n_estimators=100, random_state=42).fit(X_train, y_train)
    skb_all = SelectKBest(f_classif, k="all").fit(X_train, y_train)

    f_scores = skb_all.scores_
    order = np.argsort(-f_scores)  # descending; stable for ties
    f_rank = np.empty(len(FEATURES), dtype=int)
    for rank, idx in enumerate(order, start=1):
        f_rank[idx] = rank

    feature_scores = pd.DataFrame({
        "feature": FEATURES,
        "f_score": np.round(f_scores, 4),
        "f_rank": f_rank,
        "rf_importance": np.round(forest17.feature_importances_, 4),
        "is_noise": [c in NOISE_COLUMNS for c in FEATURES],
    })
    feature_scores.to_csv("outputs/part_c_feature_scores.csv", index=False)

    # noise columns' combined share of forest importance
    noise_share = feature_scores[feature_scores["is_noise"]]["rf_importance"].sum()
    print(f"  noise importance share: {round(noise_share, 4)}")

    # selectk: k in 5,7,10,12,17 -> validation F1
    selectk_rows = []
    for k in [5, 7, 10, 12, 17]:
        pipe = Pipeline([
            ("selector", SelectKBest(f_classif, k=k)),
            ("model", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]).fit(X_train, y_train)
        _, _, _, _, _, _, _, f1 = cm_metrics(y_val, pipe.predict(X_val))
        selectk_rows.append([k, f1])
    pd.DataFrame(selectk_rows, columns=["k", "val_f1"]).to_csv(
        "outputs/part_c_selectk.csv", index=False)

    # 4. tuning on the 7 selected columns
    selector = SelectKBest(f_classif, k=7).fit(X_train, y_train)
    X_train_sel = selector.transform(X_train)
    X_val_sel = selector.transform(X_val)

    param_grid = {
        "n_estimators": [50, 100, 150],
        "max_depth": [4, 6, 8],
        "min_samples_leaf": [1, 5, 10],
    }

    grid = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5).fit(X_train_sel, y_train)
    rand = RandomizedSearchCV(RandomForestClassifier(random_state=42), param_grid,
                              n_iter=9, cv=5, random_state=42).fit(X_train_sel, y_train)

    def val_f1_of(est):
        _, _, _, _, _, _, _, f1 = cm_metrics(y_val, est.predict(X_val_sel))
        return f1

    grid_val_f1 = val_f1_of(grid.best_estimator_)
    rand_val_f1 = val_f1_of(rand.best_estimator_)

    # default forest on the same 7 columns
    default_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    default_forest = RandomForestClassifier(random_state=42, n_estimators=100,
                                             max_depth=None, min_samples_leaf=1).fit(X_train_sel, y_train)
    default_val_f1 = val_f1_of(default_forest)

    tuning_rows = [
        ["grid_search", 27, 5, 27 * 5, fmt_params(grid.best_params_),
         round(grid.best_score_, 4), grid_val_f1],
        ["randomized_search", 9, 5, 9 * 5, fmt_params(rand.best_params_),
         round(rand.best_score_, 4), rand_val_f1],
        ["default_forest_same_7_features", 1, 0, 1, "defaults", "", default_val_f1],
    ]
    pd.DataFrame(tuning_rows, columns=["search", "candidates", "cv_folds", "total_fits",
                                       "best_params", "best_cv_accuracy", "val_f1"]).to_csv(
        "outputs/part_c_tuning.csv", index=False)

    # 5. choose final configuration (highest val_f1; ties -> default, grid, randomized)
    candidates = [
        ("default_forest_same_7_features", default_val_f1, default_params),
        ("grid_search", grid_val_f1, grid.best_params_),
        ("randomized_search", rand_val_f1, rand.best_params_),
    ]
    chosen = max(candidates, key=lambda c: c[1])[0]

    if chosen == "default_forest_same_7_features":
        chosen_params = {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1}
    elif chosen == "grid_search":
        chosen_params = grid.best_params_
    else:
        chosen_params = rand.best_params_

    final_model = Pipeline([
        ("selector", SelectKBest(f_classif, k=7)),
        ("model", RandomForestClassifier(random_state=42, **chosen_params)),
    ]).fit(X_train, y_train)

    # 6. alert threshold sweep
    val_proba = final_model.predict_proba(X_val)[:, 1]
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    thr_rows = []
    for t in thresholds:
        pred = (val_proba >= t).astype(int)
        tn, fp, fn, tp, a, p, r, f1 = cm_metrics(y_val, pred)
        thr_rows.append([t, tn, fp, fn, tp, a, p, r, f1])
    pd.DataFrame(thr_rows, columns=["threshold", "TN", "FP", "FN", "TP", "accuracy",
                                    "precision", "recall", "f1"]).to_csv(
        "outputs/part_c_thresholds.csv", index=False)

    # operations rule: highest t whose validation recall >= 0.80; else 0.1
    # row layout: [threshold, TN, FP, FN, TP, accuracy, precision, recall, f1]
    chosen_threshold = 0.1
    for t, row in zip(thresholds, thr_rows):
        if row[7] >= 0.80:
            chosen_threshold = t
    print(f"  chosen: {chosen}, threshold {chosen_threshold}")

    # 7. one test evaluation
    test_proba = final_model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= chosen_threshold).astype(int)
    tn, fp, fn, tp, a, p, r, f1 = cm_metrics(y_test, test_pred)
    pd.DataFrame([[chosen_threshold, tn, fp, fn, tp, a, p, r, f1]],
                 columns=["threshold", "TN", "FP", "FN", "TP", "accuracy",
                          "precision", "recall", "f1"]).to_csv(
        "outputs/part_c_final_test.csv", index=False)

    max_depth_str = "unlimited" if chosen_params["max_depth"] is None else chosen_params["max_depth"]
    pd.DataFrame([{
        "selector_k": 7,
        "n_estimators": chosen_params["n_estimators"],
        "max_depth": max_depth_str,
        "min_samples_leaf": chosen_params["min_samples_leaf"],
        "threshold": chosen_threshold,
        "chosen_from": chosen,
    }]).to_csv("outputs/part_c_final_config.csv", index=False)

    print("Part C complete.")
    print("  tuning val_f1: grid", grid_val_f1, "rand", rand_val_f1, "default", default_val_f1)
    print("  final test: prec", p, "rec", r, "f1", f1)


if __name__ == "__main__":
    main()
