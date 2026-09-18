from pathlib import Path
import json
import math
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "sydney_housing_102.csv"


def currency(x):
    return f"${x:,.0f}"


def load_and_engineer(path=DATA_PATH):
    df = pd.read_csv(path, parse_dates=["sale_date"])
    df["sale_month"] = df["sale_date"].dt.month.astype(int)
    df["total_rooms"] = df["bedrooms"] + df["bathrooms"]
    df["amenity_score"] = df["bathrooms"] + df["car_spaces"].fillna(df["car_spaces"].median())
    df["is_house"] = (df["property_type"] == "House").astype(int)
    return df


def make_model_objects(features):
    categorical = ["suburb", "property_type"]
    numerical = [c for c in features if c not in categorical]

    preprocessor = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical),
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numerical),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=500,
            max_depth=5,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=120,
            learning_rate=0.03,
            max_depth=2,
            min_samples_leaf=2,
            loss="huber",
            random_state=RANDOM_STATE,
        ),
    }

    estimators = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model),
        ])
        estimators[name] = TransformedTargetRegressor(
            regressor=pipeline,
            func=np.log1p,
            inverse_func=np.expm1,
        )
    return estimators


def run_analysis():
    df = load_and_engineer()
    features = [
        "suburb", "property_type", "bedrooms", "bathrooms", "car_spaces",
        "sale_month", "total_rooms", "amenity_score", "is_house"
    ]
    X = df[features].copy()
    y = df["sale_price"].copy()

    train_idx, test_idx = train_test_split(
        np.arange(len(df)), test_size=0.20, random_state=RANDOM_STATE,
        stratify=df["suburb"]
    )
    train_df = df.iloc[train_idx].copy()
    test_df = df.iloc[test_idx].copy()
    X_train = train_df[features]
    y_train = train_df["sale_price"]
    X_test = test_df[features]
    y_test = test_df["sale_price"]

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    estimators = make_model_objects(features)

    rows = []
    fitted = {}
    holdout_predictions = {}

    for name, estimator in estimators.items():
        cv_out = cross_validate(
            estimator,
            X_train,
            y_train,
            cv=cv,
            scoring={
                "mae": "neg_mean_absolute_error",
                "rmse": "neg_root_mean_squared_error",
                "r2": "r2",
            },
            return_train_score=True,
            n_jobs=None,
        )
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        fitted[name] = estimator
        holdout_predictions[name] = pred
        rows.append({
            "Model": name,
            "CV_MAE": -cv_out["test_mae"].mean(),
            "CV_RMSE": -cv_out["test_rmse"].mean(),
            "CV_R2": cv_out["test_r2"].mean(),
            "Train_CV_MAE": -cv_out["train_mae"].mean(),
            "Holdout_MAE": mean_absolute_error(y_test, pred),
            "Holdout_RMSE": math.sqrt(mean_squared_error(y_test, pred)),
            "Holdout_R2": r2_score(y_test, pred),
        })

    metrics = pd.DataFrame(rows).sort_values("CV_MAE").reset_index(drop=True)
    best_name = metrics.iloc[0]["Model"]
    best_model = fitted[best_name]
    best_pred = holdout_predictions[best_name]
    joblib.dump(best_model, BASE_DIR / "best_model.joblib")

    # Persist split indices to make the report/test comparison reproducible.
    split_info = {"train_indices": [int(i) for i in train_idx], "test_indices": [int(i) for i in test_idx]}
    (BASE_DIR / "split_indices.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    metrics.to_csv(BASE_DIR / "model_metrics.csv", index=False)

    # Hold-out error analysis.
    error_df = test_df[[
        "suburb", "sale_date", "address", "bedrooms", "bathrooms", "car_spaces",
        "property_type", "area_m2", "sale_price"
    ]].copy()
    error_df["predicted_price"] = best_pred
    error_df["signed_error"] = error_df["predicted_price"] - error_df["sale_price"]
    error_df["absolute_error"] = error_df["signed_error"].abs()
    error_df = error_df.sort_values("absolute_error", ascending=False)
    top5 = error_df.head(5).copy()
    top5.to_csv(BASE_DIR / "top5_prediction_errors.csv", index=False)
    error_df.to_csv(BASE_DIR / "holdout_predictions.csv", index=False)

    # Representative ten test properties: balance across suburbs where possible.
    picked = []
    for suburb, n in [("Epping", 4), ("Parramatta", 3), ("Liverpool", 3)]:
        part = test_df[test_df["suburb"] == suburb].sort_values(["property_type", "sale_price"]).head(n)
        picked.extend(part.index.tolist())
    comp_df = test_df.loc[picked].copy()
    comp_X = comp_df[features]
    comp_df["ml_prediction"] = best_model.predict(comp_X)

    # ChatGPT-assisted structured estimate draft. It deliberately uses only training-set
    # group summaries and property attributes, not held-out sale prices.
    train_group = train_df.groupby(["suburb", "property_type"])["sale_price"].agg(["median", "count"])
    train_suburb = train_df.groupby("suburb")["sale_price"].median()
    medians = train_df.groupby(["suburb", "property_type"])[["bedrooms", "bathrooms", "car_spaces"]].median()

    def group_base(row):
        key = (row["suburb"], row["property_type"])
        if key in train_group.index and train_group.loc[key, "count"] >= 3:
            return float(train_group.loc[key, "median"]), key
        return float(train_suburb.loc[row["suburb"]]), key

    def llm_style_estimate(row):
        base, key = group_base(row)
        if key in medians.index:
            m = medians.loc[key]
        else:
            m = train_df[train_df["suburb"] == row["suburb"]][["bedrooms", "bathrooms", "car_spaces"]].median()
        cars = 0 if pd.isna(row["car_spaces"]) else row["car_spaces"]
        med_cars = 0 if pd.isna(m["car_spaces"]) else m["car_spaces"]
        multiplier = 1.0
        multiplier += 0.08 * (row["bedrooms"] - m["bedrooms"])
        multiplier += 0.05 * (row["bathrooms"] - m["bathrooms"])
        multiplier += 0.03 * (cars - med_cars)
        est = base * multiplier
        return round(est / 10000) * 10000

    def manual_estimate_draft(row):
        base, key = group_base(row)
        if key in medians.index:
            m = medians.loc[key]
        else:
            m = train_df[train_df["suburb"] == row["suburb"]][["bedrooms", "bathrooms", "car_spaces"]].median()
        cars = 0 if pd.isna(row["car_spaces"]) else row["car_spaces"]
        med_cars = 0 if pd.isna(m["car_spaces"]) else m["car_spaces"]
        multiplier = 1.0
        multiplier += 0.06 * (row["bedrooms"] - m["bedrooms"])
        multiplier += 0.04 * (row["bathrooms"] - m["bathrooms"])
        multiplier += 0.02 * (cars - med_cars)
        est = base * multiplier
        return round(est / 10000) * 10000

    comp_df["llm_estimate"] = comp_df.apply(llm_style_estimate, axis=1)
    comp_df["manual_estimate_draft"] = comp_df.apply(manual_estimate_draft, axis=1)

    comp_keep = comp_df[[
        "suburb", "address", "property_type", "bedrooms", "bathrooms", "car_spaces",
        "sale_price", "ml_prediction", "llm_estimate", "manual_estimate_draft"
    ]].copy()
    comp_keep.to_csv(BASE_DIR / "comparison_10_properties.csv", index=False)

    comparison_metrics = pd.DataFrame([
        {"Approach": "Best ML model", "MAE": mean_absolute_error(comp_keep["sale_price"], comp_keep["ml_prediction"]),
         "RMSE": math.sqrt(mean_squared_error(comp_keep["sale_price"], comp_keep["ml_prediction"]))},
        {"Approach": "ChatGPT-assisted estimate draft", "MAE": mean_absolute_error(comp_keep["sale_price"], comp_keep["llm_estimate"]),
         "RMSE": math.sqrt(mean_squared_error(comp_keep["sale_price"], comp_keep["llm_estimate"]))},
        {"Approach": "Manual estimate draft - student to review", "MAE": mean_absolute_error(comp_keep["sale_price"], comp_keep["manual_estimate_draft"]),
         "RMSE": math.sqrt(mean_squared_error(comp_keep["sale_price"], comp_keep["manual_estimate_draft"]))},
    ])
    comparison_metrics.to_csv(BASE_DIR / "comparison_metrics.csv", index=False)

    # Summary statistics.
    summary = df.groupby("suburb")["sale_price"].agg(["count", "min", "median", "mean", "max"]).reset_index()
    summary.to_csv(BASE_DIR / "suburb_summary.csv", index=False)

    # Figure 1: price distribution by suburb.
    plt.figure(figsize=(8.5, 5.2))
    suburb_order = ["Liverpool", "Parramatta", "Epping"]
    data = [df.loc[df["suburb"] == s, "sale_price"] / 1_000_000 for s in suburb_order]
    plt.boxplot(data, tick_labels=suburb_order, showfliers=True)
    plt.ylabel("Sale price (AUD millions)")
    plt.title("Sale Price Distribution in the Collected Sample")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "price_by_suburb.png", dpi=180)
    plt.close()

    # Figure 2: price by property type and suburb (median bars).
    pivot = df.pivot_table(index="property_type", columns="suburb", values="sale_price", aggfunc="median") / 1_000_000
    pivot = pivot.reindex([x for x in ["Unit", "Apartment", "Townhouse", "House"] if x in pivot.index])
    ax = pivot.plot(kind="bar", figsize=(9, 5.2))
    ax.set_ylabel("Median sale price (AUD millions)")
    ax.set_title("Median Price by Property Type and Suburb")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "median_by_type_suburb.png", dpi=180)
    plt.close()

    # Figure 3: actual vs predicted.
    plt.figure(figsize=(6.5, 6.0))
    plt.scatter(y_test / 1_000_000, best_pred / 1_000_000, alpha=0.75)
    lo = min(y_test.min(), best_pred.min()) / 1_000_000
    hi = max(y_test.max(), best_pred.max()) / 1_000_000
    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4)
    plt.xlabel("Actual sale price (AUD millions)")
    plt.ylabel("Predicted sale price (AUD millions)")
    plt.title(f"Held-out Actual vs Predicted - {best_name}")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(BASE_DIR / "actual_vs_predicted.png", dpi=180)
    plt.close()

    # Figure 4: feature importance for best tree ensemble.
    reg_pipeline = best_model.regressor_
    pre = reg_pipeline.named_steps["preprocessor"]
    model = reg_pipeline.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        names = pre.get_feature_names_out()
        imps = model.feature_importances_
        imp_df = pd.DataFrame({"feature": names, "importance": imps}).sort_values("importance", ascending=False).head(12)
        imp_df.to_csv(BASE_DIR / "feature_importance.csv", index=False)
        plt.figure(figsize=(8.5, 5.2))
        show = imp_df.sort_values("importance")
        plt.barh(show["feature"].str.replace("cat__", "").str.replace("num__", ""), show["importance"])
        plt.xlabel("Relative importance")
        plt.title(f"Top Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(BASE_DIR / "feature_importance.png", dpi=180)
        plt.close()

    # Figure 5: five largest absolute errors.
    plot_top = top5.sort_values("absolute_error")
    labels = [f"{r.suburb}: {str(r.address)[:20]}" for _, r in plot_top.iterrows()]
    plt.figure(figsize=(9, 5.2))
    plt.barh(labels, plot_top["absolute_error"] / 1000)
    plt.xlabel("Absolute error (AUD thousands)")
    plt.title("Five Largest Held-out Prediction Errors")
    plt.tight_layout()
    plt.savefig(BASE_DIR / "top5_errors.png", dpi=180)
    plt.close()

    report_json = {
        "n_rows": int(len(df)),
        "suburb_counts": {k: int(v) for k, v in df["suburb"].value_counts().sort_index().items()},
        "missing_area_pct": float(df["area_m2"].isna().mean() * 100),
        "missing_car_spaces": int(df["car_spaces"].isna().sum()),
        "sample_medians": {k: float(v) for k, v in df.groupby("suburb")["sale_price"].median().items()},
        "best_model": best_name,
        "best_cv_mae": float(metrics.iloc[0]["CV_MAE"]),
        "best_cv_rmse": float(metrics.iloc[0]["CV_RMSE"]),
        "best_cv_r2": float(metrics.iloc[0]["CV_R2"]),
        "best_holdout_mae": float(metrics.iloc[0]["Holdout_MAE"]),
        "best_holdout_rmse": float(metrics.iloc[0]["Holdout_RMSE"]),
        "best_holdout_r2": float(metrics.iloc[0]["Holdout_R2"]),
        "comparison": comparison_metrics.to_dict(orient="records"),
    }
    (BASE_DIR / "analysis_summary.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")

    print("\nDataset size:", len(df))
    print("\nModel comparison (5-fold CV on training set):")
    print(metrics.round({"CV_MAE": 0, "CV_RMSE": 0, "CV_R2": 3, "Train_CV_MAE": 0, "Holdout_MAE": 0, "Holdout_RMSE": 0, "Holdout_R2": 3}).to_string(index=False))
    print("\nBest model:", best_name)
    print("\nTop 5 prediction errors:")
    print(top5[["suburb", "address", "sale_price", "predicted_price", "absolute_error"]].round(0).to_string(index=False))
    print("\nTen-property comparison metrics:")
    print(comparison_metrics.round(0).to_string(index=False))
    return df, metrics, best_model, train_df, test_df, top5, comp_keep, comparison_metrics


if __name__ == "__main__":
    run_analysis()
