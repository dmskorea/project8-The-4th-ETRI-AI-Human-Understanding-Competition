import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import f1_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
from lightgbm import LGBMClassifier

from src.utils import get_train_test_data


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
    categorical_cols: list,
):
    x_train, x_val, y_train, y_val = train_test_split(
        x_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train.values,
    )
    
    le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    x_train[categorical_cols] = le.fit_transform(x_train[categorical_cols])
    x_val[categorical_cols] = le.transform(x_val[categorical_cols])

    search = RandomizedSearchCV(
        estimator=LGBMClassifier(),
        param_distributions={
            'num_leaves': np.arange(20, 30, 10),
            'max_depth': [-1],
            'learning_rate': np.logspace(-3, -1, 5),
            'n_estimators': np.arange(50, 500, 50),
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'reg_alpha': [0.0, 0.1, 0.5, 1.0],
            'reg_lambda': [0.0, 0.1, 0.5, 1.0],
            'min_child_samples': [5, 10, 20, 50]
        },
        n_iter=10,
        scoring=get_macro_f1_score,
        cv=3,
        verbose=1,
        n_jobs=4,
        random_state=42,
    )

    search.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        eval_metric=get_macro_f1_score,
    )

    best_model = search.best_estimator_

    return best_model, le


def train(
    total_df: pd.DataFrame,
) -> pd.DataFrame:
    
    KEY_COLS = ["subject_id", "lifelog_date"]
    TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]
    CATEGORICAL_COLS = list(set([
        col for col in total_df.columns if total_df[col].dtype == "category"
    ]) - set(KEY_COLS + TARGET_COLS))

    processed_df = total_df.copy()

    train_df, test_df = get_train_test_data()

    X_train = train_df.merge(processed_df, on=KEY_COLS, how="left")
    X_train.drop(columns=KEY_COLS + TARGET_COLS + ["sleep_date"], errors="ignore", inplace=True)
    Y_train = train_df[TARGET_COLS]

    X_test = test_df.merge(processed_df, on=KEY_COLS, how="left")
    X_test.drop(columns=KEY_COLS + TARGET_COLS + ["sleep_date"], errors="ignore", inplace=True)

    X_train = X_train.drop(columns=["subject_id"], errors="ignore")
    X_test = X_test.drop(columns=["subject_id"], errors="ignore")

    for col in TARGET_COLS:
        print(f"Training model for {col}...")
        model, le = _train(
            x_train=X_train,
            y_train=Y_train[col],
            categorical_cols=CATEGORICAL_COLS,
        )
        print(f"Best params for {col}: {model.get_params()}")
        print(f"Best score for {col}: {model.best_score_}")

        _x_test = X_test.copy()
        _x_test[CATEGORICAL_COLS] = le.transform(X_test[CATEGORICAL_COLS])

        test_df[col] = model.predict(_x_test)

    return test_df
