from functools import reduce

import pandas as pd
import numpy as np

from src.utils import DataType, load_data, get_train_test_data


def get_mACStatus_df() -> pd.DataFrame:
    df = load_data(DataType.mACStatus)
    df["m_charging"] = df["m_charging"].astype("category")

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(charge_ratio=("m_charging", lambda x: sum(x == 1) / len(x)))
        .astype({"charge_ratio": "float32"})
    )
    return df

def get_mActivity_df() -> pd.DataFrame:
    df = load_data(DataType.mActivity)
    df["m_activity"] = df["m_activity"].astype("category")

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(m_activity=("m_activity", lambda x: x.mode().iat[0]))
        .astype({"m_activity": "category"})
    )
    return df


def get_mAmbience_df() -> pd.DataFrame:
    df = load_data(DataType.mAmbience)

    def loudest(row):
        label, _ = max(row, key=lambda x: x[1])
        return label

    df["loudest_label"] = (
        df["m_ambience"]
        .apply(loudest)
        .astype("category")
    )

    df = df.drop(columns=["m_ambience"])

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(loudest_label=("loudest_label", lambda x: x.mode().iat[0]))
        .astype({"loudest_label": "category"})
    )

    return df


def get_mBle_df() -> pd.DataFrame:
    df = load_data(DataType.mBle)
    
    def closest(row):
        if len(row) == 0:
            return None
        c = max(row, key=lambda x: x["rssi"])
        return c["device_class"]
    
    df["closest_device_class"] = (
        df["m_ble"]
        .apply(closest)
        .astype("category")
    )

    df = df.drop(columns=["m_ble"])

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(closest_device_class=("closest_device_class", lambda x: x.mode().iat[0]))
        .astype({"closest_device_class": "category"})
    )

    return df


def get_mGps_df() -> pd.DataFrame:
    df = load_data(DataType.mGps)
    
    def speed_stats(lst):
        if lst is None or len(lst) == 0:
            return (0.0, 0.0)
        speeds = np.fromiter((d["speed"] for d in lst), dtype=float)
        return (speeds.mean(), speeds.var(ddof=1))   # sample variance

    df[["speed_mean", "speed_var"]] = (
        pd.DataFrame(
            df["m_gps"].apply(speed_stats).tolist(),
            index=df.index,
            columns=["speed_mean", "speed_var"])
        .astype("float32")
    )

    df.drop(columns=["m_gps"], inplace=True)

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(
            speed_mean=("speed_mean", "mean"),
            speed_var =("speed_var",  "mean"),
        )
        .astype({"speed_mean": "float32", "speed_var": "float32"})
    )

    return df


def get_mLight_df() -> pd.DataFrame:
    df = load_data(DataType.mLight)
    df["m_light"] = df["m_light"].astype("float32")

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(m_light=("m_light", lambda x: x.mean()))
        .astype({"m_light": "float32"})
    )

    return df


def get_mScreenStatus_df() -> pd.DataFrame:
    df = load_data(DataType.mScreenStatus)
    df["m_screen_use"] = df["m_screen_use"].astype("category")

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(m_screen_use=("m_screen_use", lambda x: x.mode().iat[0]))
        .astype({"m_screen_use": "category"})
    )

    return df


def get_mUsageStats_df() -> pd.DataFrame:
    df = load_data(DataType.mUsageStats)
    
    def most_used_app(row):
        c = max(row, key=lambda x: x["total_time"])
        return c["app_name"]
    
    df["most_used_app"] = (
        df["m_usage_stats"]
        .apply(most_used_app)
        .astype("category")
    )

    df = df.drop(columns=["m_usage_stats"])

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(most_used_app=("most_used_app", lambda x: x.mode().iat[0]))
        .astype({"most_used_app": "category"})
    )

    return df


def get_mWifi_df() -> pd.DataFrame:
    df = load_data(DataType.mWifi)
    
    def wifi_count(row):
        return sum(1 for item in row if item["rssi"] > -100)

    df["wifi_count"] = (
        df["m_wifi"]
        .apply(wifi_count)
        .astype("int32")
    )

    df = df.drop(columns=["m_wifi"])

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(wifi_count=("wifi_count", lambda x: x.mode().iat[0]))
        .astype({"wifi_count": "int32"})
    )

    return df


def get_wHr_df() -> pd.DataFrame:
    df = load_data(DataType.wHr)
    
    def hr_mean(row):
        return pd.Series(row).mean()
    
    df["hr_mean"] = (
        df["heart_rate"]
        .apply(hr_mean)
        .astype("float32")
    )

    def hr_var(row):
        return pd.Series(row).var()
    
    df["hr_var"] = (
        df["heart_rate"]
        .apply(hr_var)
        .astype("float32")
    )

    df = df.drop(columns=["heart_rate"])

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(
            hr_mean=("hr_mean", "mean"),
            hr_var =("hr_var",  "mean"),
        )
        .astype({"hr_mean": "float32", "hr_var": "float32"})
    )

    return df


def get_wLight_df() -> pd.DataFrame:
    df = load_data(DataType.wLight)
    df["w_light"] = df["w_light"].astype("float32")

    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(w_light=("w_light", "mean"))
        .astype({"w_light": "float32"})
    )

    return df


def get_wPedo_df() -> pd.DataFrame:
    df = load_data(DataType.wPedo)

    df["step"] = df["step"].astype("int32")
    df["step_frequency"] = df["step_frequency"].astype("float32")
    df["running_step"] = df["running_step"].astype("int32")
    df["walking_step"] = df["walking_step"].astype("int32")
    df["distance"] = df["distance"].astype("float32")
    df["speed"] = df["speed"].astype("float32")
    df["burned_calories"] = df["burned_calories"].astype("float32")


    df = (
        df.groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .agg(
            step=("step", "sum"),
            step_frequency=("step_frequency", "mean"),
            running_step=("running_step", "sum"),
            walking_step=("walking_step", "sum"),
            distance=("distance", "sum"),
            speed=("speed", "mean"),
            burned_calories=("burned_calories", "sum"),
        )
        .astype(
            {
                "step": "int32",
                "step_frequency": "float32",
                "running_step": "int32",
                "walking_step": "int32",
                "distance": "float32",
                "speed": "float32",
                "burned_calories": "float32"
            }
        )
    )

    return df


def get_train_test_df() -> tuple[pd.DataFrame, pd.DataFrame]:
    dfs = [
        get_mACStatus_df(),
        get_mActivity_df(),
        get_mAmbience_df(),
        get_mBle_df(),
        get_mGps_df(),
        get_mLight_df(),
        get_mScreenStatus_df(),
        get_mUsageStats_df(),
        get_mWifi_df(),
        get_wHr_df(),
        get_wLight_df(),
        get_wPedo_df(),
    ]


    total_df = reduce(
        lambda left, right: pd.merge(left, right, on=["subject_id", "lifelog_date"], how="outer"),
        dfs,
    )
    
    return total_df
