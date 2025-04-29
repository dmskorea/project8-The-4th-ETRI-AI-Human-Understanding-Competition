import pandas as pd

from functools import reduce
from utils import DataType, load_data, get_train_test_data


#* Process mACStatus
m_ac_status_df = load_data(DataType.mACStatus)
m_ac_status_df["m_charging"] = m_ac_status_df["m_charging"].astype("int64")
# subject_id(object), timestamp(datetime64[ns]), m_charging (int64)

#* Process mActivity
m_activity_df = load_data(DataType.mActivity)
m_activity_df["m_activity"] = m_activity_df["m_activity"].astype("int64")
# subject_id(object), timestamp(datetime64[ns]), m_activity (int64)

#* Process mAmbience
m_ambience_df = load_data(DataType.mAmbience)
# subject_id(object), timestamp(datetime64[ns]), m_ambience (object)
# - m_ambience (object): [[Music, 0.30902..], [Vehicle, 0.30902..], ...]

# extract labels and probabilities
labels = []
props = []
for i in range(len(m_ambience_df)):
    labels.append([x[0] for x in m_ambience_df["m_ambience"][i]])
    props.append([x[1] for x in m_ambience_df["m_ambience"][i]])
m_ambience_df["labels"] = labels
m_ambience_df["probs"] = props
m_ambience_df = m_ambience_df.drop(columns=["m_ambience"])
# subject_id(object), timestamp(datetime64[ns]), labels (object), probs (object)
# - labels (object): [Music, Vehicle, ...]
# - probs (object): [0.30902.., 0.30902.., ...]

# parse loudest label
m_ambience_df["loudest_label"] = [
    m_ambience_df["labels"][i][
        m_ambience_df["probs"][i].index(max(m_ambience_df["probs"][i]))
    ]
    for i in range(len(m_ambience_df))
]
m_ambience_df = m_ambience_df.drop(columns=["labels", "probs"])
m_ambience_df["loudest_label"] = m_ambience_df["loudest_label"].astype("category")
# subject_id(object), timestamp(datetime64[ns]), loudest_label (category)
# - loudest_label (category): Music

#* Process mBle
m_ble_df = load_data(DataType.mBle)

# subject_id(object), timestamp(datetime64[ns]), m_ble (object)
# - m_ble (object): [{'address': '00:00:00:00:00:00', 'device_class': '0', 'rssi': -100}, ...]

# extract closest device_class
closest_device_classes = []
for i in range(len(m_ble_df)):
    device_classes = [x["device_class"] for x in m_ble_df["m_ble"][i]]
    rssis = [x["rssi"] for x in m_ble_df["m_ble"][i]]
    if len(device_classes) == 0:
        closest_device_class = None
    else:
        closest_device_class = device_classes[rssis.index(max(rssis))]
    closest_device_classes.append(closest_device_class)
m_ble_df["closest_device_class"] = closest_device_classes
m_ble_df = m_ble_df.drop(columns=["m_ble"])
m_ble_df["closest_device_class"] = m_ble_df["closest_device_class"].astype("category")
# subject_id(object), timestamp(datetime64[ns]), closest_device_class (category)


#* Process mGps
m_gps_df = load_data(DataType.mGps)
# subject_id(object), timestamp(datetime64[ns]), m_gps (object)
# - m_gps (object): [{'altitude': 0.0, 'latitude': 0.0, 'longitude': 0.0, 'speed': 0.0}, ...]

# extract variance of speed
m_gps_df["speed_var"] = m_gps_df["m_gps"].apply(
    lambda x: pd.Series([x["speed"] for x in x]).var()
)
m_gps_df = m_gps_df.drop(columns=["m_gps"])
m_gps_df["speed_var"] = m_gps_df["speed_var"].astype("float64")
# subject_id(object), timestamp(datetime64[ns]), speed_var (float64)

#* Process mLight
m_light_df = load_data(DataType.mLight)
m_light_df["m_light"] = m_light_df["m_light"].astype("float64")
# subject_id(object), timestamp(datetime64[ns]), m_light (float64)

#* Process mScreenStatus
m_screen_status_df = load_data(DataType.mScreenStatus)
m_screen_status_df["m_screen_use"] = m_screen_status_df["m_screen_use"].astype("int64")
# subject_id(object), timestamp(datetime64[ns]), m_screen_use (int64)

#* Process mUsageStats
m_usage_stats_df = load_data(DataType.mUsageStats)
# subject_id(object), timestamp(datetime64[ns]), m_usage_stats (object)
# - m_usage_stats (object): [{'app_name': 'kakaotalk', 'total_time': 0.0}, ...]

# extract most_used_app_name
most_used_app_names = []
for i in range(len(m_usage_stats_df)):
    app_names = [x["app_name"] for x in m_usage_stats_df["m_usage_stats"][i]]
    total_times = [x["total_time"] for x in m_usage_stats_df["m_usage_stats"][i]]
    if len(app_names) == 0:
        most_used_app_name = None
    else:
        most_used_app_name = app_names[total_times.index(max(total_times))]
    most_used_app_names.append(most_used_app_name)
m_usage_stats_df = m_usage_stats_df.drop(columns=["m_usage_stats"])
m_usage_stats_df["most_used_app_name"] = pd.Series(most_used_app_names, dtype="category")
# subject_id(object), timestamp(datetime64[ns]), most_used_app_name (category)

#* Process mWifi
m_wifi_df = load_data(DataType.mWifi)
# subject_id(object), timestamp(datetime64[ns]), m_wifi (object)
# - m_wifi (object): [{'bssid': 'SSID', 'rssi': -100}, ...]

# extract wifi num
m_wifi_df["wifi_num"] = m_wifi_df["m_wifi"].apply(
    lambda x: len([x["bssid"] for x in x if x["rssi"] > -100])
)
m_wifi_df = m_wifi_df.drop(columns=["m_wifi"])
m_wifi_df["wifi_num"] = m_wifi_df["wifi_num"].astype("int64")
# subject_id(object), timestamp(datetime64[ns]), wifi_num (int64)

#* Process wHr
w_hr_df = load_data(DataType.wHr)
# subject_id(object), timestamp(datetime64[ns]), heart_rate (object)
# - w_hr (object): [134, 135, 136, ...]

# extract mean and std
w_hr_df["mean"] = w_hr_df["heart_rate"].apply(lambda x: pd.Series(x).mean()).astype("float64")
w_hr_df["std"] = w_hr_df["heart_rate"].apply(lambda x: pd.Series(x).std()).astype("float64")
w_hr_df = w_hr_df.drop(columns=["heart_rate"])
# subject_id(object), timestamp(datetime64[ns]), mean (float64), std (float64)


#* Process wLight
w_light_df = load_data(DataType.wLight)
w_light_df["w_light"] = w_light_df["w_light"].astype("float64")
# subject_id(object), timestamp(datetime64[ns]), w_light (float64)

#* Process wPedo
w_pedo_df = load_data(DataType.wPedo)
w_pedo_df["step"] = w_pedo_df["step"].astype("int64")
w_pedo_df["step_frequency"] = w_pedo_df["step_frequency"].astype("float64")
w_pedo_df["running_step"] = w_pedo_df["running_step"].astype("int64")
w_pedo_df["walking_step"] = w_pedo_df["walking_step"].astype("int64")
w_pedo_df["distance"] = w_pedo_df["distance"].astype("float64")
w_pedo_df["speed"] = w_pedo_df["speed"].astype("float64")
w_pedo_df["burned_calories"] = w_pedo_df["burned_calories"].astype("float64")
# subject_id(object), timestamp(datetime64[ns]), step, step_frequency, running_step, walking_step, distance, speed, burned_calories



#* Gather rows to one row for a date
df_list = [
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
]

aggregated = []
for df in df_list:
    new_df = df.copy()
    new_df["subject_id"] = new_df["subject_id"].astype("category")
    new_df["lifelog_date"] = new_df["timestamp"].dt.date.astype("datetime64[ns]")

    agg_dict = {}
    for column in new_df.columns:
        if column != "subject_id" and column != "timestamp" and column != "lifelog_date":
            if pd.api.types.is_numeric_dtype(new_df[column]):
                agg_dict[column] = "mean"
            elif isinstance(new_df[column].dtype, pd.CategoricalDtype):
                agg_dict[column] = lambda x: x.mode().iat[0] if not x.mode().empty else pd.NA
            else:
                raise ValueError(f"Unsupported data type: {new_df[column].dtype}")

    new_df = new_df.groupby(["subject_id", "lifelog_date"]).agg(agg_dict).reset_index()
    aggregated.append(new_df)

total_df = reduce(
    lambda left, right: pd.merge(left, right, on=["subject_id", "lifelog_date"], how="outer"),
    aggregated,
)

#* Split Data
train_df, test_df = get_train_test_data()


#* train
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

TARGET_COLS = ["Q1", "Q2", "Q3", "S1", "S2", "S3"]
KEY_COLS    = ["subject_id", "lifelog_date"]

def make_X(df, total_df):
    X = (
        df.merge(total_df, on=KEY_COLS, how="inner")
          .drop(columns=TARGET_COLS + ["sleep_date"])
    )
    return X

X_train = make_X(train_df, total_df)
y_train = train_df[TARGET_COLS]

X_test  = make_X(test_df,  total_df)

num_cols = X_train.select_dtypes(include=["number"]).columns
cat_cols = X_train.select_dtypes(exclude=["number"]).columns

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", categorical_pipe, cat_cols),
])

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=400,
        max_depth=None,
        n_jobs=-1,
        random_state=42
    )
) 

pipe = Pipeline([
    ("prep", preprocess),
    ("est",  model),
])

pipe.fit(X_train, y_train)

preds = pipe.predict(X_test)
test_df[TARGET_COLS] = preds.round().astype(int)

print(test_df.head())

test_df.to_csv("submission.csv", index=False)
