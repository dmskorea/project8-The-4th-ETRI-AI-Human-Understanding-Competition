from pathlib import Path
from enum import Enum

import pandas as pd

DEFAULT_BASE_DIR = "./data/"


class DataType(Enum):
    mACStatus = "mACStatus"
    mActivity = "mActivity"
    mAmbience = "mAmbience"
    mBle = "mBle"
    mGps = "mGps"
    mLight = "mLight"
    mScreenStatus = "mScreenStatus"
    mUsageStats = "mUsageStats"
    mWifi = "mWifi"
    wHr = "wHr"
    wLight = "wLight"
    wPedo = "wPedo"


def load_data(data_type: DataType, base_dir=DEFAULT_BASE_DIR) -> pd.DataFrame:
    """
    Load data from the specified data type and base directory.

    Args:
        data_type (DataType): The type of data to load.
        base_dir (str): The base directory where the data files are located.

    Returns:
        pd.DataFrame: The loaded data as a DataFrame.
    """
    file_path = Path(base_dir) / f"ch2025_data_items/ch2025_{data_type.value}.parquet"
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_parquet(file_path)
    return df


def get_train_test_data(base_dir=DEFAULT_BASE_DIR) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get the train and test timestamps from the data.

    Args:
        base_dir (str): The base directory where the data files are located.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the train and test DataFrames.
    """
    train_file_path = Path(base_dir) / "ch2025_metrics_train.csv"
    test_file_path = Path(base_dir) / "ch2025_submission_sample.csv"

    train_df = pd.read_csv(train_file_path)
    test_df = pd.read_csv(test_file_path)

    train_df["subject_id"] = train_df["subject_id"].astype("category")
    test_df["subject_id"] = test_df["subject_id"].astype("category")
    train_df["lifelog_date"] = pd.to_datetime(train_df["lifelog_date"])
    test_df["lifelog_date"] = pd.to_datetime(test_df["lifelog_date"])

    return train_df, test_df
