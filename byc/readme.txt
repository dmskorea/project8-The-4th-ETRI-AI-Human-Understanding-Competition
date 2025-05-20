1) 기존 Best Score Baseline Code ->  NaN 처리 -> train.fillna(-1) ,test.fillna(-1) = v2

2) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm -> Catboost 교체 = v3

3) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm -> Xgboost 교체 = v3_xgboost

4) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm + Xgboost soft voting  = lightgbm_xgboost_2_model_pv

5) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm + Xgboost + Catboost 3:3:4 soft voting  = 3-model-pv
