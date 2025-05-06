from functools import reduce
from pathlib import Path

import pandas as pd
import numpy as np

from src.utils import DataType, load_data, get_train_test_data


MIGHT_SLEEP_HOURS = list(range(0, 10)) + list(range(22, 24))  # 22:00 ~ 09:59
MIGHT_NOT_SLEEP_HOURS = list(range(8, 24))  # 09:00 ~ 23:59
category_groups = []


def shift_lifelog_date(df: pd.DataFrame) -> pd.DataFrame:
    sleep_hours = set(MIGHT_SLEEP_HOURS)
    not_sleep = set(MIGHT_NOT_SLEEP_HOURS)
    overlap = sleep_hours & not_sleep            # 이중 기록 구간
    only_sleep = sleep_hours - overlap           # 전날로만 이동할 구간

    # 1) 전날로만 이동 (오전 시간에만)
    mask_only = df['hour'].isin(only_sleep) & (df['hour'] < 12)
    df.loc[mask_only, 'lifelog_date'] = df.loc[mask_only, 'lifelog_date'] - pd.Timedelta(days=1)

    # 2) 이중 기록 (오전 시간에만)
    mask_overlap = df['hour'].isin(overlap) & (df['hour'] < 12)
    if mask_overlap.any():
        df_overlap = df[mask_overlap].copy()
        df_overlap['lifelog_date'] = df_overlap['lifelog_date'] - pd.Timedelta(days=1)
        df = pd.concat([df, df_overlap], ignore_index=True, sort=False)

    df = df.sort_values(by=['subject_id', 'lifelog_date', 'timestamp']).reset_index(drop=True)

    return df


def get_mACStatus_df() -> pd.DataFrame:
    df = load_data(DataType.mACStatus)
    df = shift_lifelog_date(df)
    df["m_charging"] = df["m_charging"].astype("int32")

    def inner(df):
        if len(df) == 0:
            charge_ratio = 0.0
            charge_state_changed = 0
        else:
            charge_ratio = sum(df["m_charging"] == 1) / len(df)
            charge_state_changed = df["m_charging"].diff().ne(0).astype("int32").sum()
        return pd.Series({"charge_ratio": charge_ratio, "charge_state_changed": charge_state_changed})

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"charge_ratio": "float32", "charge_state_changed": "int32"})
        .rename(columns={"charge_ratio": "s_charge_ratio", "charge_state_changed": "s_charge_state_changed"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"charge_ratio": "float32", "charge_state_changed": "int32"})
        .rename(columns={"charge_ratio": "ns_charge_ratio", "charge_state_changed": "ns_charge_state_changed"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total

def get_mActivity_df() -> pd.DataFrame:
    df = load_data(DataType.mActivity)
    df = shift_lifelog_date(df)
    df["m_activity"] = df["m_activity"].astype("category")

    ACTIVITY_MAP = {
        0: "IN_VEHICLE",
        1: "ON_BICYCLE",
        2: "ON_FOOT",
        8: "RUNNING",
        3: "STILL",
        5: "TILTING",
        4: "UNKNOWN",  # default
        7: "WALKING",
    }

    def inner(df):
        if len(df) == 0:
            activity_first_most = 4
            activity_second_most = 4
        else:
            value_counts = df["m_activity"].value_counts()
            activity_first_most = int(value_counts.index[0] if len(value_counts) > 0 else 4)
            activity_second_most = int(value_counts.index[1] if len(value_counts) > 1 else activity_first_most)
        return pd.Series({"activity_first_most": ACTIVITY_MAP[activity_first_most], "activity_second_most": ACTIVITY_MAP[activity_second_most]})

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"activity_first_most": "category", "activity_second_most": "category"})
        .rename(columns={"activity_first_most": "s_activity_first_most", "activity_second_most": "s_activity_second_most"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"activity_first_most": "category", "activity_second_most": "category"})
        .rename(columns={"activity_first_most": "ns_activity_first_most", "activity_second_most": "ns_activity_second_most"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    category_groups.append([
        "s_activity_first_most", "s_activity_second_most",
        "ns_activity_first_most", "ns_activity_second_most",
    ])

    return total


def get_mAmbience_df() -> pd.DataFrame:
    df = load_data(DataType.mAmbience)
    df = shift_lifelog_date(df)

    def inner(df):
        if len(df) == 0:
            ambience_first_most = 0
            ambience_second_most = 0
        else:
            probs = df["m_ambience"]
            values = probs.map(lambda x: x[0][0] if x is not None and len(x) > 0 else "UNKNOWN")
            value_counts = values.value_counts()
            ambience_first_most = value_counts.index[0]
            ambience_second_most = value_counts.index[1] if len(value_counts) > 1 else ambience_first_most
            ambience_third_most = value_counts.index[2] if len(value_counts) > 2 else ambience_second_most

        return pd.Series({
            "ambience_first_most": ambience_first_most, 
            "ambience_second_most": ambience_second_most,
            "ambience_third_most": ambience_third_most,
        })

    
    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"ambience_first_most": "category", "ambience_second_most": "category", "ambience_third_most": "category"})
        .rename(columns={"ambience_first_most": "s_ambience_first_most", "ambience_second_most": "s_ambience_second_most", "ambience_third_most": "s_ambience_third_most"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"ambience_first_most": "category", "ambience_second_most": "category", "ambience_third_most": "category"})
        .rename(columns={"ambience_first_most": "ns_ambience_first_most", "ambience_second_most": "ns_ambience_second_most", "ambience_third_most": "ns_ambience_third_most"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    category_groups.append([
        "s_ambience_first_most", "s_ambience_second_most", "s_ambience_third_most",
        "ns_ambience_first_most", "ns_ambience_second_most", "ns_ambience_third_most",
    ])

    return total


def get_mBle_df() -> pd.DataFrame:
    df = load_data(DataType.mBle)
    df = shift_lifelog_date(df)

    def inner(df):
        if len(df) == 0:
            num_devices = 0
        else:
            num_devices = sum([len(x) for x in df["m_ble"]])

        return pd.Series({"num_devices": num_devices})


    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"num_devices": "int32"})
        .rename(columns={"num_devices": "s_num_devices"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"num_devices": "int32"})
        .rename(columns={"num_devices": "ns_num_devices"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")
    
    return total


def get_mGps_df() -> pd.DataFrame:
    df = load_data(DataType.mGps)
    df = shift_lifelog_date(df)
    
    def inner(df):
        if len(df) == 0:
            mean_speed = 0.0
            std_speed = 0.0
        else:
            speeds = df["m_gps"].map(lambda x: [item["speed"] for item in x if item is not None])
            speeds = sum(speeds.values, [])
            mean_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
            std_speed = np.std(speeds) if len(speeds) > 0 else 0.0

        return pd.Series({"mean_speed": mean_speed, "std_speed": std_speed})

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_speed": "float32", "std_speed": "float32"})
        .rename(columns={"mean_speed": "s_mean_speed", "std_speed": "s_std_speed"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_speed": "float32", "std_speed": "float32"})
        .rename(columns={"mean_speed": "ns_mean_speed", "std_speed": "ns_std_speed"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total


def get_mLight_df() -> pd.DataFrame:
    df = load_data(DataType.mLight)
    df = shift_lifelog_date(df)

    def inner(df):
        if len(df) == 0:
            mean_light = 0.0
            std_light = 0.0
        else:
            lights = df["m_light"].values
            mean_light = np.mean(lights) if len(lights) > 0 else 0.0
            std_light = np.std(lights) if len(lights) > 0 else 0.0

        return pd.Series({"mean_light": mean_light, "std_light": std_light})

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_light": "float32", "std_light": "float32"})
        .rename(columns={"mean_light": "s_mean_light", "std_light": "s_std_light"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_light": "float32", "std_light": "float32"})
        .rename(columns={"mean_light": "ns_mean_light", "std_light": "ns_std_light"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total


def get_mScreenStatus_df() -> pd.DataFrame:
    df = load_data(DataType.mScreenStatus)
    df = shift_lifelog_date(df)

    def inner(df):
        if len(df) == 0:
            screen_use_first_most = 0
            screen_use_second_most = 0
        else:
            uses = df["m_screen_use"].astype("int").values
            screen_on_ratio = sum(uses) / len(uses) if len(uses) > 0 else 0.0
            screen_state_changed = np.sum(np.diff(uses) == 1) if len(uses) > 1 else 0
            
            return pd.Series({
                "screen_on_ratio": screen_on_ratio, 
                "screen_state_changed": screen_state_changed
            })

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"screen_on_ratio": "float32", "screen_state_changed": "int32"})
        .rename(columns={"screen_on_ratio": "s_screen_on_ratio", "screen_state_changed": "s_screen_state_changed"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"screen_on_ratio": "float32", "screen_state_changed": "int32"})
        .rename(columns={"screen_on_ratio": "ns_screen_on_ratio", "screen_state_changed": "ns_screen_state_changed"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total
        

def get_mUsageStats_df() -> pd.DataFrame:
    df = load_data(DataType.mUsageStats)
    df = shift_lifelog_date(df)
    
    def inner(df):
        if len(df) == 0:
            app_first_most = 0
            app_second_most = 0
            app_third_most = 0
        else:
            app_use = {}
            for apps in df["m_usage_stats"].values:
                for app in apps:
                    name = app["app_name"]
                    time = app["total_time"]
                    if name in app_use:
                        app_use[name] += time
                    else:
                        app_use[name] = time
            app_use = sorted(app_use.items(), key=lambda x: x[1], reverse=True)
            app_first_most = app_use[0][0] if len(app_use) > 0 else "UNKNOWN"
            app_second_most = app_use[1][0] if len(app_use) > 1 else app_first_most
            app_third_most = app_use[2][0] if len(app_use) > 2 else app_second_most

        return pd.Series({"app_first_most": app_first_most, "app_second_most": app_second_most, "app_third_most": app_third_most})
    
    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"app_first_most": "category", "app_second_most": "category", "app_third_most": "category"})
        .rename(columns={"app_first_most": "s_app_first_most", "app_second_most": "s_app_second_most", "app_third_most": "s_app_third_most"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"app_first_most": "category", "app_second_most": "category", "app_third_most": "category"})
        .rename(columns={"app_first_most": "ns_app_first_most", "app_second_most": "ns_app_second_most", "app_third_most": "ns_app_third_most"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    category_groups.append([
        "s_app_first_most", "s_app_second_most", "s_app_third_most",
        "ns_app_first_most", "ns_app_second_most", "ns_app_third_most",
    ])

    return total


def get_mWifi_df() -> pd.DataFrame:
    df = load_data(DataType.mWifi)
    df = shift_lifelog_date(df)
    
    def inner(df):
        if len(df) == 0:
            wifi_count = 0
        else:
            wifi_count = sum([len(x) for x in df["m_wifi"]])
        return pd.Series({"wifi_count": wifi_count})

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"wifi_count": "int32"})
        .rename(columns={"wifi_count": "s_wifi_count"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"wifi_count": "int32"})
        .rename(columns={"wifi_count": "ns_wifi_count"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total


def get_wHr_df() -> pd.DataFrame:
    df = load_data(DataType.wHr)
    df = shift_lifelog_date(df)
    
    def inner(df):
        if len(df) == 0:
            mean_hr = 0.0
            std_hr = 0.0
        else:
            values = []
            for hr in df["heart_rate"]:
                if hr is not None and len(hr) > 0:
                    values.extend(hr.tolist())
            mean_hr = np.mean(values) if len(values) > 0 else 0.0
            std_hr = np.std(values) if len(values) > 0 else 0.0

        return pd.Series({"mean_hr": mean_hr, "std_hr": std_hr})
    
    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_hr": "float32", "std_hr": "float32"})
        .rename(columns={"mean_hr": "s_mean_hr", "std_hr": "s_std_hr"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_hr": "float32", "std_hr": "float32"})
        .rename(columns={"mean_hr": "ns_mean_hr", "std_hr": "ns_std_hr"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total


def get_wLight_df() -> pd.DataFrame:
    df = load_data(DataType.wLight)
    df = shift_lifelog_date(df)
    df["w_light"] = df["w_light"].astype("float32")

    def inner(df):
        if len(df) == 0:
            mean_light = 0.0
            std_light = 0.0
        else:
            lights = df["w_light"].values
            mean_light = np.mean(lights) if len(lights) > 0 else 0.0
            std_light = np.std(lights) if len(lights) > 0 else 0.0

        return pd.Series({"mean_wlight": mean_light, "std_wlight": std_light})

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_wlight": "float32", "std_wlight": "float32"})
        .rename(columns={"mean_wlight": "s_mean_wlight", "std_wlight": "s_std_wlight"})
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({"mean_wlight": "float32", "std_wlight": "float32"})
        .rename(columns={"mean_wlight": "ns_mean_wlight", "std_wlight": "ns_std_wlight"})
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total


def get_wPedo_df() -> pd.DataFrame:
    df = load_data(DataType.wPedo)
    df = shift_lifelog_date(df)

    def inner(df):
        if len(df) == 0:
            step = 0
            step_frequency = 0
            running_step = 0
            walking_step = 0
            distance = 0
            speed = 0
            burned_calories = 0
        else:
            step = df["step"].sum()
            step_frequency = df["step_frequency"].mean()
            running_step = df["running_step"].sum()
            walking_step = df["walking_step"].sum()
            distance = df["distance"].sum()
            speed = df["speed"].mean()
            burned_calories = df["burned_calories"].sum()
        return pd.Series({
            "w_step": step,
            "w_step_frequency": step_frequency,
            # "w_running_step": running_step,
            # "w_walking_step": walking_step,
            "w_distance": distance,
            "w_speed": speed,
            "w_burned_calories": burned_calories,
        })

    sdf = (
        df[df["hour"].isin(MIGHT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({
            "w_step": "int32",
            "w_step_frequency": "float32",
            # "w_running_step": "int32",
            # "w_walking_step": "int32",
            "w_distance": "float32",
            "w_speed": "float32",
            "w_burned_calories": "float32",
        })
        .rename(columns={
            "w_step": "s_w_step",
            "w_step_frequency": "s_w_step_frequency",
            # "w_running_step": "s_w_running_step",
            # "w_walking_step": "s_w_walking_step",
            "w_distance": "s_w_distance",
            "w_speed": "s_w_speed",
            "w_burned_calories": "s_w_burned_calories",
        })
    )

    nsdf = (
        df[df["hour"].isin(MIGHT_NOT_SLEEP_HOURS)]
        .groupby(["subject_id", "lifelog_date"], as_index=False, sort=False, observed=True)
        .apply(inner)
        .reset_index(drop=True)
        .astype({
            "w_step": "int32",
            "w_step_frequency": "float32",
            # "w_running_step": "int32",
            # "w_walking_step": "int32",
            "w_distance": "float32",
            "w_speed": "float32",
            "w_burned_calories": "float32",
        })
        .rename(columns={
            "w_step": "ns_w_step",
            "w_step_frequency": "ns_w_step_frequency",
            # "w_running_step": "ns_w_running_step",
            # "w_walking_step": "ns_w_walking_step",
            "w_distance": "ns_w_distance",
            "w_speed": "ns_w_speed",
            "w_burned_calories": "ns_w_burned_calories",
        })
    )

    total = pd.merge(sdf, nsdf, on=["subject_id", "lifelog_date"], how="outer")

    return total


def get_train_test_df(result_dir: str = None) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    total_df["month"] = total_df["lifelog_date"].dt.month
    total_df["day"] = total_df["lifelog_date"].dt.day

    # Apply Category Groups
    for category_group in category_groups:
        names = []
        for category in category_group:
            names.extend(total_df[category].cat.categories.tolist())
        names = list(set(names))

        for category in category_group:
            total_df[category] = total_df[category].cat.set_categories(names)

    result_dir = Path(result_dir or "./results/")

    total_df.to_csv(Path(result_dir) / "total_df.csv", index=False)

    KEY_COLS = ["subject_id", "lifelog_date"]
    TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]
    CATEGORICAL_COLS = list(set([
        col for col in total_df.columns if total_df[col].dtype == "category"
    ]) - set(KEY_COLS + TARGET_COLS))

    train_df, test_df = get_train_test_data()
    
    valid_chunks = []

    for subject_id, group in train_df.groupby("subject_id", sort=False, observed=True):
        valid_chunks.append(group.sample(frac=0.2, random_state=42))

    # now concatenate all dev‐chunks in one go
    valid_df = pd.concat(valid_chunks, ignore_index=False)

    # remove those rows from train_df
    train_df = train_df.drop(valid_df.index)

    X_train = train_df.merge(total_df, on=KEY_COLS, how="left")
    X_train.drop(columns=TARGET_COLS, errors="ignore", inplace=True)
    Y_train = train_df[KEY_COLS].copy()
    Y_train[TARGET_COLS] = train_df[TARGET_COLS]

    X_valid = valid_df.merge(total_df, on=KEY_COLS, how="left")
    X_valid.drop(columns=TARGET_COLS, errors="ignore", inplace=True)
    Y_valid = valid_df[KEY_COLS].copy()
    Y_valid[TARGET_COLS] = valid_df[TARGET_COLS]

    X_test = test_df.merge(total_df, on=KEY_COLS, how="left")
    X_test.drop(columns=TARGET_COLS, errors="ignore", inplace=True)

    X_train.to_csv(Path(result_dir) / "X_train.csv", index=False)
    Y_train.to_csv(Path(result_dir) / "Y_train.csv", index=False)
    X_valid.to_csv(Path(result_dir) / "X_valid.csv", index=False)
    Y_valid.to_csv(Path(result_dir) / "Y_valid.csv", index=False)
    X_test.to_csv(Path(result_dir) / "X_test.csv", index=False)

    eda(
        df=pd.merge(X_train, Y_train, on=KEY_COLS, how="left"),
        result_dir=result_dir / "eda"
    )

    return (
        X_train,
        Y_train,
        X_valid,
        Y_valid,
        X_test,
    )


def eda(df: pd.DataFrame, result_dir: str) -> None:
    # from ydata_profiling import ProfileReport
    # Path(result_dir).mkdir(parents=True, exist_ok=True)
    # ProfileReport(df, title='EDA').to_file(Path(result_dir) / "eda_report.html")
    pass


if __name__ == "__main__":
    # total_df = get_train_test_df(result_dir="results/v1")
    X_train, Y_train = pd.read_csv("results/v1/X_train.csv"), pd.read_csv("results/v1/Y_train.csv")

    Y_train[["subject_id", "lifelog_date"]] = X_train[["subject_id", "lifelog_date"]]
    train_data = pd.merge(X_train, Y_train, on=["subject_id", "lifelog_date"], how="left")
    eda(train_data, result_dir="results/v1/eda")