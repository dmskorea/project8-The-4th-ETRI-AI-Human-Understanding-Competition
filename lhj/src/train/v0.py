import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.metrics import make_scorer

from src.utils import get_train_test_data


RANDOM_STATE = 42


def train(
    total_df: pd.DataFrame,
) -> pd.DataFrame:
    
    KEY_COLS = ["subject_id", "lifelog_date"]
    TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]
    
    train_df, test_df = get_train_test_data()

    X_train = train_df.merge(total_df, on=KEY_COLS, how="left")
    X_train.drop(columns=TARGET_COLS + ["sleep_date"], errors="ignore", inplace=True)
    Y_train = train_df[TARGET_COLS]

    X_test = test_df.merge(total_df, on=KEY_COLS, how="left")
    X_test.drop(columns=TARGET_COLS + ["sleep_date"], errors="ignore", inplace=True)

    train_groups = train_df["subject_id"]
    X_train = X_train.drop(columns=["subject_id"], errors="ignore")
    X_test = X_test.drop(columns=["subject_id"], errors="ignore")

    # Identify column types after the previous drops
    num_cols = X_train.select_dtypes(include=["number"]).columns
    cat_cols = X_train.select_dtypes(exclude=["number"]).columns

    # -----------------------------------------------------------
    # Pre‑processing pipeline
    # -----------------------------------------------------------
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    preprocess = ColumnTransformer([
        ("num", numeric_pipe, num_cols),
        ("cat", categorical_pipe, cat_cols),
    ])

    def multi_mae(y_true, y_pred):
        """Mean Absolute Error averaged across all six targets."""
        return np.mean(np.abs(y_true - y_pred))

    mae_scorer = make_scorer(multi_mae, greater_is_better=False)

    # -----------------------------------------------------------
    # Base estimator & wrapper
    # -----------------------------------------------------------
    base_estimator = RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model = MultiOutputRegressor(base_estimator)

    pipe = Pipeline([
        ("prep", preprocess),
        ("est", model),
    ])

    # -----------------------------------------------------------
    # Hyper‑parameter search with GroupKFold CV
    # -----------------------------------------------------------
    param_distributions = {
        "est__estimator__n_estimators": [300, 400, 600, 800],
        "est__estimator__max_depth": [None, 10, 20, 40],
        "est__estimator__min_samples_split": [2, 4, 8, 16],
        "est__estimator__min_samples_leaf": [1, 2, 4],
        "est__estimator__max_features": ["sqrt", "log2", None],
    }

    gkf = GroupKFold(n_splits=5)

    search = RandomizedSearchCV(
        estimator=pipe,
        param_distributions=param_distributions,
        n_iter=40,
        cv=gkf.split(X_train, Y_train, groups=train_groups),
        scoring=mae_scorer,
        verbose=2,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    # -----------------------------------------------------------
    # Train & select best model
    # -----------------------------------------------------------
    search.fit(X_train, Y_train)
    print(f"Best CV MAE: {-search.best_score_:.4f}")
    print("Best hyper‑parameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    best_model = search.best_estimator_

    # -----------------------------------------------------------
    # Make predictions & save submission
    # -----------------------------------------------------------
    preds = best_model.predict(X_test)

    # Competition rules appear to expect integers – adjust if necessary
    preds = np.rint(preds).astype(int)

    # (Optional) Clip within plausible bounds – adjust range as per data sheet
    preds = np.clip(preds, 0, 10)

    test_df[TARGET_COLS] = preds
    return test_df
