import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
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


def train(
    total_df: pd.DataFrame,
) -> pd.DataFrame:
    
    KEY_COLS = ["subject_id", "lifelog_date"]
    TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]

    processed_df = total_df.copy()

    train_df, test_df = get_train_test_data()

    X_train = train_df.merge(processed_df, on=KEY_COLS, how="left")
    X_train.drop(columns=KEY_COLS + TARGET_COLS + ["sleep_date"], errors="ignore", inplace=True)
    Y_train = train_df[TARGET_COLS]

    X_test = test_df.merge(processed_df, on=KEY_COLS, how="left")
    X_test.drop(columns=KEY_COLS + TARGET_COLS + ["sleep_date"], errors="ignore", inplace=True)

    X_train = X_train.drop(columns=["subject_id"], errors="ignore")
    X_test = X_test.drop(columns=["subject_id"], errors="ignore")


    # split
    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train,
        Y_train,
        test_size=0.2,
        random_state=42,
        stratify=Y_train.values.argmax(axis=1),
    )

    
    categorical_cols = list(set([
        col for col in total_df.columns if total_df[col].dtype == "category"
    ]) - set(KEY_COLS + TARGET_COLS))
    
    le = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_train[categorical_cols] = le.fit_transform(X_train[categorical_cols])
    X_val[categorical_cols] = le.transform(X_val[categorical_cols])
    X_test[categorical_cols] = le.transform(X_test[categorical_cols])

    # get model
    for col in TARGET_COLS:
        model = LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.01,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )

        model.fit(
            X_train,
            Y_train[col],
            eval_set=[(X_val, Y_val[col])],
            eval_metric=get_macro_f1_score,
        )

        preds = model.predict(X_test)
        test_df[col] = preds

    return test_df
