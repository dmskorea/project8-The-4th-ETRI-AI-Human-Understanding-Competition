1) 기존 Best Score Baseline Code ->  NaN 처리 -> train.fillna(-1) ,test.fillna(-1) = v2

2) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm -> Catboost 교체 = v3

3) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm -> Xgboost 교체 = v3_xgboost

4) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm + Xgboost soft voting  = lightgbm_xgboost_2_model_pv

5) 기존 etri_baseline_v7141_v3_0_3.ipynb Best Score Baseline Code -> Lightgbm + Xgboost + Catboost 3:3:4 soft voting  = 3-model-pv


[논문 작성을 위한 LLM 3가지 접근 실험코드]

1. 파인튜닝 (peft) -> 분류기
    1. https://github.com/dmskorea/project8-The-4th-ETRI-AI-Human-Understanding-Competition/blob/main/byc/dacon-etri-lifelog-llm-cf-peft-q3.ipynb
2. 퓨샷러닝 -> 분류기
    1. https://github.com/dmskorea/project8-The-4th-ETRI-AI-Human-Understanding-Competition/blob/main/byc/dacon-etri-lifelog-llm-fewshot.ipynb
3. 데이터 증강 -> 생성
    1. https://github.com/dmskorea/project8-The-4th-ETRI-AI-Human-Understanding-Competition/blob/main/byc/dacon-etri-lifelog-v4-3-model-weight-class-llm.ipynb
