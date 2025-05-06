# -*- coding: utf-8 -*-
"""LightGBM training pipeline – **debug‑fixed & speed‑oriented**

Changes in this patch
---------------------
* **FIX** `build_pipeline` duplicated call (`Pipeline(...)(...)`) → now returns
  the `Pipeline` object correctly.
* **FIX** `_train_single` wrongly returning `search.best_params_(...)`.
* **Binary objective** no longer sets `num_class=1` (invalid for LightGBM).
* Added `min_child_samples = [1, 5, 10, 20]` to search space.
* Configurable *fast* mode (`FAST = True`) that halves search iterations.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from src.utils import get_train_test_data  # domain‑specific util

# ---------------------------- configuration ---------------------------------
RANDOM_STATE = 42
FAST = True  # True → faster dev run (N_ITER_SEARCH halved)
N_ITER_SEARCH = 30 // 2 if FAST else 30
CV_SPLITS = 3
EARLY_STOPPING = 100
TARGET_COLS: Sequence[str] = ("Q1", "Q2", "Q3", "S1", "S2", "S3")
KEY_COLS: Sequence[str] = ("subject_id", "lifelog_date")

# ----------------------------- helpers --------------------------------------
class NpEncoder(json.JSONEncoder):
    """JSON encoder that gracefully handles NumPy / pandas objects."""

    def default(self, obj):  # noqa: D401
        import numpy as np

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return super().default(obj)


def lgb_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[str, float, bool]:
    """LightGBM custom metric that converts scores → labels → macro‑F1."""

    # LightGBM passes raw scores (or probas if configured). Infer shape.
    if y_pred.ndim == 1:  # binary raw score/proba
        y_pred_labels = (y_pred > 0.5).astype(int)
    else:  # multiclass [n_samples * n_classes]
        y_pred_labels = np.argmax(y_pred, axis=1)

    return "macro_f1", f1_score(y_true, y_pred_labels, average="macro"), True


macro_f1_scorer = make_scorer(f1_score, average="macro")

warnings.filterwarnings(
    "ignore",
    message=r"X does not have valid feature names, but .* was fitted with feature names",
)

# ------------------------- pipeline components ------------------------------

def build_pipeline(
    categorical: List[str],
    *,
    n_classes: int,
    class_weight: Dict[int, float] | str | None = None,
) -> Pipeline:
    ct = ColumnTransformer(
        [
            (
                "cat",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                categorical,
            )
        ],
        remainder="passthrough",
    )

    lgb_params: Dict[str, object] = dict(
        objective="multiclass" if n_classes > 2 else "binary",
        n_estimators=1500,  # speed tweak (early stopping active)
        random_state=RANDOM_STATE,
        n_jobs=1,  # avoid nested parallelism
        verbosity=-1,
        class_weight=class_weight,
    )
    if n_classes > 2:
        lgb_params["num_class"] = n_classes

    return Pipeline([("prep", ct), ("lgbm", LGBMClassifier(**lgb_params))])


# ----------------------- hyper‑parameter search -----------------------------

def build_search(pipe: Pipeline, *, n_classes: int) -> RandomizedSearchCV:
    param_dist = {
        "lgbm__num_leaves": np.arange(20, 60, 10),
        "lgbm__learning_rate": np.logspace(-3, -1, 4),
        "lgbm__subsample": [0.6, 0.8, 1.0],
        "lgbm__colsample_bytree": [0.6, 0.8, 1.0],
        "lgbm__reg_alpha": [0.0, 0.1, 0.5],
        "lgbm__reg_lambda": [0.0, 0.1, 0.5],
        "lgbm__min_child_samples": [1, 5, 10, 20],
    }

    auc_metric = "roc_auc" if n_classes == 2 else "roc_auc_ovo_weighted"

    cv = StratifiedKFold(CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    return RandomizedSearchCV(
        pipe,
        param_dist,
        n_iter=N_ITER_SEARCH,
        scoring={"macro_f1": macro_f1_scorer, "auc": auc_metric},
        refit="macro_f1",
        cv=cv,
        n_jobs=8,
        verbose=1,
        random_state=RANDOM_STATE,
    )


# ------------------------------ trainer -------------------------------------

def _train_single(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    categorical: List[str],
):
    n_classes = int(y_tr.nunique())

    # imbalance handling ---------------------------------------------------
    if n_classes > 2:
        freq = y_tr.value_counts(normalize=True)
        class_weight = {cls: freq.max() / freq[cls] for cls in freq.index}
    else:
        class_weight = "balanced"

    pipe = build_pipeline(categorical, n_classes=n_classes, class_weight=class_weight)
    search = build_search(pipe, n_classes=n_classes)
    search.fit(X_tr, y_tr)

    best: Pipeline = search.best_estimator_

    best.fit(
        pd.concat([X_tr, X_val]),
        pd.concat([y_tr, y_val]),
        lgbm__eval_set=[(X_val, y_val)],
        lgbm__eval_metric=lgb_macro_f1,
        lgbm__callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )

    return best, float(search.best_score_), search.best_params_


# ---------------------------- public API ------------------------------------

def train(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    Y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    result_dir: str | Path,
):
    result_dir = Path(result_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    cat_cols = sorted(
        col
        for col in X_train.columns
        if str(X_train[col].dtype) == "category" and col not in (*KEY_COLS, *TARGET_COLS)
    )

    drop_cols = [*KEY_COLS, "sleep_date", "subject_id"]
    X_tr_f, X_val_f, X_test_f = (
        df.drop(columns=drop_cols, errors="ignore") for df in (X_train, X_valid, X_test)
    )

    _, test_df = get_train_test_data()

    metrics: Dict[str, object] = {}
    cv_means: List[float] = []

    for tgt in TARGET_COLS:
        print(f"\n▶▶ Training target: {tgt}")
        model, score, params = _train_single(
            X_tr_f, Y_train[tgt], X_val_f, Y_valid[tgt], cat_cols
        )
        metrics[tgt] = {"cv_macro_f1": score, "best_params": params}
        cv_means.append(score)
        test_df[tgt] = model.predict(X_test_f)

    metrics["macro_f1_mean"] = float(np.mean(cv_means))

    (result_dir / "metrics.json").write_text(
        json.dumps(metrics, cls=NpEncoder, indent=4, ensure_ascii=False)
    )
    sub_path = result_dir / f"submission_val_{metrics['macro_f1_mean']:.6f}.csv"
    test_df.to_csv(sub_path, index=False)
    print("Saved submission →", sub_path)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("Import the `train` function; do not run this file directly.")
