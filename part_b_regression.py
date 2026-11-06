"""Part B - Delivery-Hours Regression.

Builds a point estimate of delivery_hours from data/modelling_table.csv, compares
four candidate regression models, and picks a final model. All outputs in outputs/.
Inputs X are the 17 FEATURES in pinned order; target y is delivery_hours.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.shipment_prep import FEATURES

MODEL_PATH = "data/modelling_table.csv"


def reg_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return [round(mae, 4), round(mse, 4), round(rmse, 4), round(r2, 4)]


def main():
    df = pd.read_csv(MODEL_PATH)
    X = df[FEATURES].copy()
    y = df["delivery_hours"].copy()

    # 1. split: 840 / 180 / 180
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    assert (len(X_train), len(X_val), len(X_test)) == (840, 180, 180)

    # 2. scaling without leakage: fit scaler on training rows only
    scaler = StandardScaler().fit(X_train)
    X_train_sc = scaler.transform(X_train)
    X_test_sc = scaler.transform(X_test)
    train_scaled_max_abs_mean = float(np.abs(X_train_sc.mean(axis=0)).max())
    test_scaled_max_abs_mean = float(np.abs(X_test_sc.mean(axis=0)).max())

    pd.DataFrame([
        ("train_rows", len(X_train)),
        ("val_rows", len(X_val)),
        ("test_rows", len(X_test)),
        ("train_scaled_max_abs_mean", round(train_scaled_max_abs_mean, 4)),
        ("test_scaled_max_abs_mean", round(test_scaled_max_abs_mean, 4)),
    ], columns=["metric", "value"]).to_csv("outputs/part_b_checks.csv", index=False)

    # 3. baseline: predict mean of y_train for every validation row
    baseline_pred = np.full(len(y_val), y_train.mean())
    baseline_metrics = ["mean_baseline", "val"] + reg_metrics(y_val, baseline_pred)

    # 4. tree-depth diagnostic
    depths = [1, 2, 3, 4, 6, 8, 12, None]
    tree_rows = []
    best_depth = None
    best_val_rmse = np.inf
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, random_state=42).fit(X_train, y_train)
        tr_rmse = float(np.sqrt(mean_squared_error(y_train, m.predict(X_train))))
        v_rmse = float(np.sqrt(mean_squared_error(y_val, m.predict(X_val))))
        tree_rows.append([("unlimited" if d is None else d), round(tr_rmse, 4), round(v_rmse, 4)])
        # lowest val_rmse, ties -> smaller depth
        if v_rmse < best_val_rmse - 1e-12:
            best_val_rmse = v_rmse
            best_depth = d
    pd.DataFrame(tree_rows, columns=["max_depth", "train_rmse", "val_rmse"]).to_csv(
        "outputs/part_b_tree_depth.csv", index=False)

    # 5. four candidate models, scored on validation rows
    models = {}

    lr = Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]).fit(X_train, y_train)
    models["linear_regression"] = lr

    dt = DecisionTreeRegressor(max_depth=best_depth, random_state=42).fit(X_train, y_train)
    models["decision_tree"] = dt

    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    models["random_forest"] = rf

    gb = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42).fit(X_train, y_train)
    models["gradient_boosting"] = gb

    metrics_rows = [baseline_metrics]
    val_rmse_by_model = {}
    for name, m in models.items():
        preds = m.predict(X_val)
        row = [name, "val"] + reg_metrics(y_val, preds)
        metrics_rows.append(row)
        val_rmse_by_model[name] = float(np.sqrt(mean_squared_error(y_val, preds)))

    # 6. learning-rate experiment
    lr_rows = []
    for rate in [0.01, 0.1, 1.0]:
        m = GradientBoostingRegressor(n_estimators=100, learning_rate=rate, random_state=42).fit(X_train, y_train)
        preds = m.predict(X_val)
        v_rmse = float(np.sqrt(mean_squared_error(y_val, preds)))
        v_r2 = r2_score(y_val, preds)
        lr_rows.append([rate, round(v_rmse, 4), round(v_r2, 4)])
    pd.DataFrame(lr_rows, columns=["learning_rate", "val_rmse", "val_r2"]).to_csv(
        "outputs/part_b_gb_learning_rate.csv", index=False)

    # 7. cross-validated comparison (two best by val RMSE) + final model
    ranked = sorted(val_rmse_by_model.items(), key=lambda kv: kv[1])
    two_best = [name for name, _ in ranked[:2]]

    X_trainval = pd.concat([X_train, X_val])
    y_trainval = pd.concat([y_train, y_val])
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    cv_rows = []
    for name in two_best:
        scores = cross_val_score(models[name], X_trainval, y_trainval, cv=kf)
        cv_rows.append([name, round(scores.mean(), 4), round(scores.std(), 4)])
    pd.DataFrame(cv_rows, columns=["model", "mean_r2", "std_r2"]).to_csv("outputs/part_b_cv.csv", index=False)

    # final model: candidate with lowest validation RMSE; score once on test
    final_name = ranked[0][0]
    final_pred = models[final_name].predict(X_test)
    metrics_rows.append([final_name, "test"] + reg_metrics(y_test, final_pred))

    pd.DataFrame(metrics_rows, columns=["model", "split", "MAE", "MSE", "RMSE", "R2"]).to_csv(
        "outputs/part_b_metrics.csv", index=False)

    print("Part B complete.")
    print(f"  split sizes: {len(X_train)}/{len(X_val)}/{len(X_test)}")
    print(f"  train_scaled_max_abs_mean: {train_scaled_max_abs_mean:.2e}")
    print(f"  test_scaled_max_abs_mean: {test_scaled_max_abs_mean:.4f}")
    print(f"  best tree depth: {best_depth} (val_rmse {best_val_rmse:.4f})")
    print(f"  val RMSE ranking: {ranked}")
    print(f"  final model: {final_name}")
    for r in metrics_rows:
        print("   ", r)


if __name__ == "__main__":
    main()
