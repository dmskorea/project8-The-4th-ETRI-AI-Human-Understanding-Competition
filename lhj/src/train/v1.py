import os
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

from src.utils import get_train_test_data


class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        try:
            return super().default(obj)
        except TypeError:
            return str(obj)


def get_macro_f1_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # 예측 확률에서 가장 높은 index가 예측 클래스
    if len(y_pred.shape) > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        # 이진 분류일 경우: 확률값을 threshold로 변환
        y_pred_labels = (y_pred > 0.5).astype(int)
    score = f1_score(y_true, y_pred_labels, average='macro')
    return 'macro_f1', score, True


def _train(
    x_train: pd.DataFrame,
    y_train: pd.DataFrame,
    x_valid: pd.DataFrame,
    y_valid: pd.DataFrame,
    categorical_cols: list,
):  
    le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_train[categorical_cols] = le.fit_transform(x_train[categorical_cols])
    x_valid[categorical_cols] = le.transform(x_valid[categorical_cols])

    search = RandomizedSearchCV(
        estimator=LGBMClassifier(),
        param_distributions={
            'num_leaves': np.arange(10, 50, 5),
            'max_depth': [-1],
            'learning_rate': np.logspace(-4, -1, 5),
            'n_estimators': np.arange(100, 1000, 100),
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            'min_child_samples': [5, 10, 20, 50]
        },
        n_iter=10,
        scoring={
            "macro_f1": get_macro_f1_score,
            "auc": "roc_auc",
        },
        refit="macro_f1",
        cv=3,
        verbose=1,
        n_jobs=4,
        random_state=42,
    )

    search.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric=get_macro_f1_score,
    )

    best_model = search.best_estimator_

    return best_model, le


def train(
    X_train: pd.DataFrame,
    Y_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    Y_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    result_dir: str,
):
    
    KEY_COLS = ["subject_id", "lifelog_date"]
    TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]
    CATEGORICAL_COLS = list(set([
        col for col in X_train.columns if X_train[col].dtype == "category"
    ]) - set(KEY_COLS + TARGET_COLS))

    _, test_df = get_train_test_data()

    X_train = X_train.drop(columns=KEY_COLS + ["sleep_date", "subject_id"], errors="ignore")
    X_valid = X_valid.drop(columns=KEY_COLS + ["sleep_date", "subject_id"], errors="ignore")
    X_test = X_test.drop(columns=KEY_COLS + ["sleep_date", "subject_id"], errors="ignore")

    metrics = {}
    for col in TARGET_COLS:
        print(f"Training model for {col}...")
        model, le = _train(
            x_train=X_train,
            y_train=Y_train[col],
            x_valid=X_valid,
            y_valid=Y_valid[col],
            categorical_cols=CATEGORICAL_COLS,
        )
        print(f"Best params for {col}: {model.get_params()}")
        print(f"Best score for {col}: {model.best_score_}")

        metrics[col] = {
            "best_params": model.get_params(),
            "best_score": model.best_score_,
        }

        _x_test = X_test.copy()
        _x_test[CATEGORICAL_COLS] = le.transform(X_test[CATEGORICAL_COLS])

        test_df[col] = model.predict(_x_test)
    
    f1_scores = []
    for col in TARGET_COLS:
        f1_scores.append(metrics[col]['best_score']['valid_0']['macro_f1'])
    metrics["macro_f1"] = np.mean(f1_scores)

    # save tratin results
    with open(os.path.join(result_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False, cls=EnhancedJSONEncoder)

    test_df.to_csv(result_dir / f"submission_val_{metrics['macro_f1']:.6f}.csv", index=False)
