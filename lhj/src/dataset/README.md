# Data 정보

## mACStatus
- m_charging
    | m_charging | Description      |
    |------------|------------------|
    | 0          | 충전 중 아님      |
    | 1          | 충전 중          |

## mActivity
- m_activity
    | m_activity | Description  |
    |------------|--------------|
    | 0          | IN_VEHICLE   |
    | 1          | ON_BICYCLE   |
    | 2          | ON_FOOT      |
    | 3          | STILL        |
    | 4          | UNKNOWN      |
    | 5          | TILTING      |
    | 7          | WALKING      |
    | 8          | RUNNING      |

## mAmbience_df
- m_ambience
    ```python
    [
        ["Music", "0.30112"],
        ["Vehicle", "0.30112"],
        ...
    ]
    ```


## mBle
- m_ble
    ```python
    [
        {'address': '~', 'device_class': '0', 'rssi': -82},
        ...
    ]
    ```


## mGps
- m_gps
    ```python
    [
        {'altitude': 0.0, 'latitude': 37.5665, 'longitude': 126.978, 'speed': 0.0},
        ...
    ]
    ```


## mLight
- m_light
  - 빛 강도 float 값


## mScreenStatus
- m_screen_use
    | m_screen_use | Description      |
    |--------------|------------------|
    | 0            | 화면 사용 중 아님 |
    | 1            | 화면 사용 중     |


## mUsageStats
- m_usage_stats
    ```python
    [
        {'app_name': 'kakaotalk', 'total_time': 0.0}
        ...
    ]
    ```


## mWifi
- m_wifi
    ```python
    [
        {'bssid': 'SSID', 'rssi': -100}
        ...
    ]
    ```


## wHr
- heart_rate
    - 심박수 int 리스트
        ```
        [72, 75, 78, ...]
        ```


## wLight
- w_light
    - 빛 강도 float 값


## wPedo
- step
- step_frequency
- running_step
- walking_step
- distance
- speed
- burned_calories