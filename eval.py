from __future__ import annotations
from typing import Callable

import pandas as pd
from sklearn.metrics import f1_score


def _build_target_subset(
    target: pd.DataFrame,
    test_splits: list[tuple[str, pd.Timestamp, pd.Timestamp]],
) -> pd.DataFrame:
    """
    Vectorised construction of the target subset for **all** test-splits.
    A single pass over `target` is cheaper than concatenating many small frames.
    """
    # Build a boolean mask for every split, then OR-reduce them column-wise
    target["sleep_date"] = target["sleep_date"].astype("datetime64[ns]")
    masks = [
        (target["subject_id"] == subj) & target["sleep_date"].between(pd.Timestamp(start), pd.Timestamp(end))
        for subj, start, end in test_splits
    ]
    combined_mask = pd.concat(masks, axis=1).any(axis=1)
    return target.loc[combined_mask]


def evaluate(
    submission: pd.DataFrame,
    target: pd.DataFrame,
    test_splits: list[tuple[str, pd.Timestamp, pd.Timestamp]],
    eval_function: Callable[[pd.Series, pd.Series], float],
) -> dict[str, float]:
    """
    Evaluate a submission against the ground-truth `target`
    on the requested `test_splits` and return per-label
    and macro-averaged F1 (or any metric supplied).
    """
    submission = submission.astype(
        {
            "subject_id": "category",
            "sleep_date": "datetime64[ns]",
        }
    )
    target = target.astype(
        {
            "subject_id": "category",
            "sleep_date": "datetime64[ns]",
        }
    )

    target_df = _build_target_subset(target, test_splits)

    # Align submission with target in a single hash-join merge
    aligned = (
        target_df[["subject_id", "sleep_date"]]
        .merge(
            submission,
            on=["subject_id", "sleep_date"],
            how="inner",
            validate="one_to_one",
        )
    )

    # Compute scores
    cols = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]
    scores = {col: eval_function(aligned[col], target_df[col]) for col in cols}
    scores["macro_f1"] = sum(scores.values()) / len(cols)
    return scores


def eval_function(
    y_pred: pd.Series,
    y_true: pd.Series,
) -> float:
    """
    Compute the macro-averaged F1 score for the given
    prediction and ground-truth series.
    """

    # Convert to numpy arrays
    y_pred = y_pred.to_numpy()
    y_true = y_true.to_numpy()

    # Compute the F1 score
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return f1



import inspect
import streamlit as st

TARGET_PATH = "ch2025_metrics_train.csv"

if "sleep_dates" not in st.session_state:
    target = pd.read_csv(TARGET_PATH)

    sleep_dates = {
        subject_id: (
            group["sleep_date"].min(),
            group["sleep_date"].max(),
        ) for subject_id, group in target.groupby("subject_id")
    }
    st.session_state["subject_ids"] = sorted(list(sleep_dates.keys()))
    st.session_state["sleep_dates"] = sleep_dates
    
    for subject_id in st.session_state["subject_ids"]:
        st.session_state[f"{subject_id}_sleep_date"] = (
            pd.to_datetime(sleep_dates[subject_id][0]),
            pd.to_datetime(sleep_dates[subject_id][1]),
        )

st.title("단머스 모의 제출기")
st.write("""
단머스 모의 제출기를 사용하여 모델의 성능을 평가할 수 있습니다.
각자 실험의 공정한 비교를 위해 만들어졌습니다.

평가 대상 기간은 제공된 ch2025_metrics_train.csv 의 subset 입니다.

기간 설정을 통해 평가 기간을 정하고 공유 및 불러오기를 통해 서로 같은 기간을 대상으로 평가할 수 있습니다.
""")


def date_setting(prefix: str):
    start_date, end_date = st.session_state["sleep_dates"][prefix]
    st.date_input(
        f"{prefix} 평가 대상 sleep_date",
        min_value=start_date,
        max_value=end_date,
        key=f"{prefix}_sleep_date",
    )

def get_dates():
    return [
        [subject_id, *st.session_state[f"{subject_id}_sleep_date"]]
        for subject_id in st.session_state["subject_ids"]
    ]


def set_dates(date_string: str) -> list[tuple[str, pd.Timestamp, pd.Timestamp]]:
    for line in date_string.split("\n"):
        if line.strip():
            subject_id, start_date, end_date = line.split(",")
            st.session_state[f"{subject_id}_sleep_date"] = (
                pd.to_datetime(start_date),
                pd.to_datetime(end_date),
            )
    print(st.session_state[f"{subject_id}_sleep_date"])


with st.expander("기간 설정"):
    ids = st.session_state["subject_ids"]
    for id in ids:
        date_setting(id)


dates = get_dates()
col1, col2 = st.columns(2)
with col1:
    st.download_button(
        "기간 공유",
        data="\n".join([f"{subject_id},{start_date},{end_date}" for subject_id, start_date, end_date in dates]),
        file_name="dates.txt",
        mime="text/plain"
    )
with col2:
    if st.button("기간 불러오기"):
        st.file_uploader(
            "Upload a file",
            type=["txt"],
            key="date_file_uploader",
            on_change=lambda: set_dates(st.session_state["date_file_uploader"].getvalue().decode("utf-8")),
        )


st.write("평가하기")
submission = st.file_uploader("제출 파일", type=["csv"])
if submission is not None:
    submission = pd.read_csv(submission)
    target = pd.read_csv(TARGET_PATH)

    target = _build_target_subset(
        target,
        get_dates(),
    )

    col1, col2 = st.columns(2)
    with col1:
        st.write(f"제출 파일 미리보기 {len(submission)} 개")
        st.dataframe(submission.head(3))
    with col2:
        st.write(f"정답 파일 미리보기 {len(target)} 개")
        st.dataframe(target.head(3))

    with st.expander("평가 함수"):
        st.code(
            inspect.getsource(eval_function),
        )

    if st.button("평가하기"):
        with st.spinner("평가 중..."):
            result = evaluate(
                submission,
                target,
                dates,
                eval_function,
            )
        st.success("평가 완료!")
        st.json(result)