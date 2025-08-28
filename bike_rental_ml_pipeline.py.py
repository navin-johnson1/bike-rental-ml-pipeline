#Importing the necessary libraries
import os
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from typing import Tuple, Dict

from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#Warnings to keep the output clean
warnings.filterwarnings("ignore")

#Assigning variables to the Files
DATA_RAW = "FloridaBikeRentals.csv"
DATA_PROCESSED = "bike_rental_features.csv"
BEST_MODEL_FILE = "best_poly_model.pkl"
OUTPUT_DIR = "Unit 3 Capstone Outputs"

#Plot defaults
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

#Utility helpers
def ensure_out_dir():
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

#Loading data
def load_raw_data() -> pd.DataFrame:
    try:
        df = pd.read_csv(DATA_RAW, encoding="utf-8")
    except UnicodeDecodeError:
        print("UTF-8 failed; retrying with latin1...")
        df = pd.read_csv(DATA_RAW, encoding="latin1")
    print(f"Loaded {DATA_RAW} | shape={df.shape}")
    print("Columns:", list(df.columns))
    return df

#Clean and engineer
def clean_and_engineer(df: pd.DataFrame) -> Tuple[pd.DataFrame, str, list, list]:
    df = df.copy()

    #Parse Date
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        #Drop rows where date couldn't be parsed
        df = df.dropna(subset=["Date"])
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["dayofweek"] = df["Date"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    #Map human-friendly Yes/No columns to 0/1 if present
    if "Holiday" in df.columns:
        df["Holiday"] = df["Holiday"].astype(str).str.strip()
        df["Holiday"] = df["Holiday"].map({"Holiday": 1, "No Holiday": 0}).fillna(0).astype(int)

    if "Functioning Day" in df.columns:
        df["Functioning Day"] = df["Functioning Day"].astype(str).str.strip()
        df["Functioning Day"] = df["Functioning Day"].map({"Yes": 1, "No": 0}).fillna(1).astype(int)

    #Normalize column names
    def normalize(col: str) -> str:
        col = col.replace("°", "")
        col = col.replace("(", "").replace(")", "").replace("/", "_").replace("%", "")
        col = col.replace(".", "_")
        col = col.replace("-", "_")
        col = col.strip().lower().replace(" ", "_")
        return col

    df.columns = [normalize(c) for c in df.columns]

    #Target column
    #After normalization
    if "rented_bike_count" not in df.columns:
        raise ValueError("Expected target column 'Rented Bike Count' (normalized to 'rented_bike_count') not found.")

    target = "rented_bike_count"

    #Identify categorical & numeric features
    cat_features = []
    if "seasons" in df.columns:
        cat_features.append("seasons")

    drop_cols = ["date", target] if "date" in df.columns else [target]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    #Split into numeric vs categorical
    numeric_features = []
    for c in feature_cols:
        if c in cat_features:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_features.append(c)
        else:
            #if any non-numeric slips in (unlikely beyond 'seasons'), treat as categorical
            if c not in cat_features:
                cat_features.append(c)

    #Handle missing values: numeric -> median, categorical -> mode
    for c in numeric_features:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_features:
        if df[c].isnull().any():
            df[c] = df[c].fillna(df[c].mode()[0])

    return df, target, numeric_features, cat_features

#Saving Encoded dataset
def save_encoded(df: pd.DataFrame, target: str, cat_features: list):
    #Shallow copy to avoid side effects
    tmp = df.copy()
    #One-hot encode categorical features
    tmp = pd.get_dummies(tmp, columns=cat_features, drop_first=True)
    #Save
    tmp.to_csv(DATA_PROCESSED, index=False)
    print(f"\nSaved encoded dataset to: {DATA_PROCESSED}")

#Saving Correlation Heatmap
def correlation_heatmap(df: pd.DataFrame):
    ensure_out_dir()
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    plt.figure(figsize=(12, 9))
    sns.heatmap(corr, cmap="coolwarm", annot=False)
    plt.title("Correlation Heatmap (Numeric Features)")
    fp = os.path.join(OUTPUT_DIR, "correlation_heatmap.png")
    plt.tight_layout()
    plt.savefig(fp)
    plt.close()
    print(f"\nSaved heatmap to {fp}")

#Building Preprocessor
def build_preprocessor(numeric_features: list, cat_features: list) -> ColumnTransformer:
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="drop",
    )
    return preprocessor

#Evaluating
def evaluate(model, X_test, y_test) -> Dict[str, float]:
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return {"MAE": mae, "MSE": mse, "R2": r2}

#Cross Validating
def cross_validate(model, X_train, y_train, scoring="neg_mean_squared_error"):
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
    if scoring == "neg_mean_squared_error":
        print(f"CV MSE: mean={-scores.mean():.4f}, std={scores.std():.4f}")
    else:
        print(f"CV {scoring}: mean={scores.mean():.4f}, std={scores.std():.4f}")

#Task 1
def task1_feature_engineering() -> Tuple[pd.DataFrame, str, list, list]:
    df_raw = load_raw_data()
    df_eng, target, num_feats, cat_feats = clean_and_engineer(df_raw)
    #Saving encoded dataset
    save_encoded(df_eng, target, cat_feats)
    #EDA heatmap
    correlation_heatmap(df_eng)
    return df_eng, target, num_feats, cat_feats

#Task 2
def task2_model_building(df_eng, target, num_feats, cat_feats):
    X = df_eng[num_feats + cat_feats]
    y = df_eng[target]

    #Train/test split BEFORE any scaling/encoding to avoid leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor(num_feats, cat_feats)

    models_and_params = {
        "Linear": (LinearRegression(), {}),
        "Ridge": (Ridge(), {"model__alpha": [0.01, 0.1, 1, 10, 100]}),
        "Lasso": (Lasso(max_iter=10000), {"model__alpha": [0.001, 0.01, 0.1, 1, 10]}),
        "ElasticNet": (
            ElasticNet(max_iter=10000),
            {"model__alpha": [0.001, 0.01, 0.1, 1], "model__l1_ratio": [0.2, 0.5, 0.8]},
        ),
    }

    fitted_models = {}
    test_metrics = {}

    print("\nTask 2: Model building & tuning (no polynomial)")
    for name, (estimator, param_grid) in models_and_params.items():
        print(f"\n{name}")
        pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
        grid = GridSearchCV(
            pipe,
            param_grid=param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
        )
        grid.fit(X_train, y_train)
        fitted_models[name] = grid.best_estimator_
        print("Best params:", grid.best_params_ if grid.best_params_ else "{}")
        print("Best CV MSE (neg):", grid.best_score_)

        # Cross-validation on the train split (reporting)
        cross_validate(grid.best_estimator_, X_train, y_train)

        # Test-set evaluation
        test_metrics[name] = evaluate(grid.best_estimator_, X_test, y_test)
        print("Test metrics:", test_metrics[name])

    return fitted_models, test_metrics, X_train, X_test, y_train, y_test, preprocessor

#Task 3
def task3_polynomial_models(X_train, X_test, y_train, y_test, preprocessor, num_feats, cat_feats):
    print("\nTask 3: Polynomial features models")

    # Preprocessor augmented with PolynomialFeatures for numeric part
    num_poly = Pipeline(
        steps=[
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("scaler", StandardScaler()),
        ]
    )
    cat_enc = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    poly_preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_poly, num_feats),
            ("cat", cat_enc, cat_feats),
        ],
        remainder="drop",
    )

    models = {
        "Linear_poly": LinearRegression(),
        "Ridge_poly": Ridge(),
        "Lasso_poly": Lasso(max_iter=20000),
        "ElasticNet_poly": ElasticNet(max_iter=20000),
    }
    
    fitted_poly = {}
    poly_metrics = {}

    for name, est in models.items():
        print(f"\n{name}")
        pipe = Pipeline([("preprocessor", poly_preprocessor), ("model", est)])
        pipe.fit(X_train, y_train)
        fitted_poly[name] = pipe

        #Cross-validation on the train split (polynomial)
        cross_validate(pipe, X_train, y_train)

        #Test-set evaluation
        poly_metrics[name] = evaluate(pipe, X_test, y_test)
        print("Test metrics:", poly_metrics[name])

    #Choose best polynomial model by R squared on test
    best_poly_name = max(poly_metrics.keys(), key=lambda k: poly_metrics[k]["R2"])
    best_poly_model = fitted_poly[best_poly_name]
    print(f"\nBest polynomial model by R squared: {best_poly_name} | {poly_metrics[best_poly_name]}")

    #Save best polynomial model
    ensure_out_dir()
    joblib.dump(best_poly_model, os.path.join(OUTPUT_DIR, BEST_MODEL_FILE))
    print(f"\nSaved best polynomial model to {os.path.join(OUTPUT_DIR, BEST_MODEL_FILE)}")

    return fitted_poly, poly_metrics, best_poly_name, best_poly_model

#Task 4
def task4_compare_and_validate(test_metrics_linear, test_metrics_poly):
    print("\nTask 4: Comparison (Linear vs Polynomial models)")
    print("\nLinear models (test):")
    for name, m in test_metrics_linear.items():
        print(f"{name}: {m}")

    print("\nPolynomial models (test):")
    for name, m in test_metrics_poly.items():
        print(f"{name}: {m}")

#Task 5
def task5_report(
    num_feats, cat_feats, test_metrics_linear, test_metrics_poly, best_poly_name
):
    ensure_out_dir()
    report_path = os.path.join(OUTPUT_DIR, "report.txt")
    lines = []
    lines.append("\nBike Rentals Forecasting — Summary Report")
    lines.append("\n--------------------------------------------------")
    lines.append("\nTask 1 (Feature Engineering): ")
    lines.append(f"\n- Numeric features: {num_feats}")
    lines.append(f"\n- Categorical features (one-hot encoded): {cat_feats}")
    lines.append("\n- Scaling applied via StandardScaler inside modeling pipelines (no leakage).")
    lines.append("\n- Encoded features saved to 'bike_rental_features.csv'.")

    lines.append("\nTask 2 (Models without polynomial): ")
    for name, m in test_metrics_linear.items():
        lines.append(f"\n  * {name}: MAE={m['MAE']:.3f}, MSE={m['MSE']:.3f}, Rsquared={m['R2']:.4f}")
    lines.append("\n")

    lines.append("\nTask 3 (Polynomial features): ")
    for name, m in test_metrics_poly.items():
        lines.append(f"  * {name}: MAE={m['MAE']:.3f}, MSE={m['MSE']:.3f}, Rsquared={m['R2']:.4f}")
    lines.append(f"\n\n- Best polynomial model: {best_poly_name}")

    lines.append("\nTask 4 (Validation): ")
    lines.append("\n- 5-fold CV used on train splits to assess generalization.")
    lines.append("\n\n- Compared test metrics across all models (linear vs polynomial).")

    lines.append("\nTask 5 (Insights & Recommendations): ")
    lines.append("\n- Temperature, hour, humidity, and seasonal effects typically correlate with demand.")
    lines.append("\n- Polynomial models often capture non-linear relationships (e.g., temperature vs. demand).")
    lines.append("\n- For the business: consider dynamic pricing & staffing by hour/season/weather signals.")
    lines.append("\n- Further improvements: add lag features (previous hour/day), holiday calendars, weather forecasts,")
    lines.append("\ninteraction terms (e.g., temperature × hour), and try tree-based models (Random Forest, XGBoost).")

    with open(report_path, "w") as f:
        f.writelines(lines)

    print(f"Saved report to {report_path}")

#Main
def main():
    ensure_out_dir()

    #Task 1: Feature engineering
    df_eng, target, num_feats, cat_feats = task1_feature_engineering()

    #Task 2: Model building
    models_linear, metrics_linear, X_train, X_test, y_train, y_test, preproc = task2_model_building(
        df_eng, target, num_feats, cat_feats
    )

    #Task 3: Polynomial features
    models_poly, metrics_poly, best_poly_name, best_poly_model = task3_polynomial_models(
        X_train, X_test, y_train, y_test, preproc, num_feats, cat_feats
    )

    #Task 4: Evaluation & validation
    task4_compare_and_validate(metrics_linear, metrics_poly)

    #Task 5: Reporting
    task5_report(num_feats, cat_feats, metrics_linear, metrics_poly, best_poly_name)

    print("\nDone. Here are the outputs:")
    print(f"\n- Encoded dataset: {DATA_PROCESSED}")
    print(f"\n- Best polynomial model: {os.path.join(OUTPUT_DIR, BEST_MODEL_FILE)}")
    print(f"\n- Correlation heatmap: {os.path.join(OUTPUT_DIR, 'correlation_heatmap.png')}")
    print(f"\n- Report: {os.path.join(OUTPUT_DIR, 'report.txt')}")

if __name__ == "__main__":
    main()