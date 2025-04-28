from pathlib import Path
import pandas as pd

from functools import reduce


base_dir = "./data/"

def parse_mACStatus() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mACStatus.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mActivity() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mActivity.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mAmbience() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mAmbience.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mBle() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mBle.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mGps() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mGps.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mLight() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mLight.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mScreenStatus() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mScreenStatus.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mUsageStats() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mUsageStats.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_mWifi() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_mWifi.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_wHr() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_wHr.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_wLight() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_wLight.parquet"
    df = pd.read_parquet(file_path)
    return df


def parse_wPedo() -> pd.DataFrame:
    file_path = Path(base_dir) / "items/ch2025_wPedo.parquet"
    df = pd.read_parquet(file_path)
    return df


m_ac_status_df = parse_mACStatus()
m_activity_df = parse_mActivity()
m_ambience_df = parse_mAmbience()
m_ble_df = parse_mBle()
m_gps_df = parse_mGps()
m_light_df = parse_mLight()
m_screen_status_df = parse_mScreenStatus()
m_usage_stats_df = parse_mUsageStats()
m_wifi_df = parse_mWifi()
w_hr_df = parse_wHr()
w_light_df = parse_wLight()
w_pedo_df = parse_wPedo()


def merge_df(dfs) -> pd.DataFrame:
    merged = reduce(
        lambda left, right: pd.merge(
            left, right, on=["subject_id", "timestamp"], how="outer"
        ),
        dfs,
    )
    return merged

# usage
merged_df = merge_df([
    m_ac_status_df,
    m_activity_df,
    m_ambience_df,
    m_ble_df,
    m_gps_df,
    m_light_df,
    m_screen_status_df,
    m_usage_stats_df,
    m_wifi_df,
    w_hr_df,
    w_light_df,
    w_pedo_df,
])
