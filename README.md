### 📦로컬 리더보드 
- submit 기준 : 로컬 리더보드 valid, oof(k=5) 보다 성능 향상 된 경우

| #  |  valid  |  oof(k=5)  | test    |       내용       |          코드                  | 
|----|---------|------------|--------|-------------------|--------------------------------|
| -  |  0.6403 |   0.6755   | 0.61260| lightgbm -> tabpfn | etri-baseline-dm-v03.ipynb|
| -  |  0.6419 |   0.6667   | 0.60828| ch5_sleeptime 이미지 추가 | etri_baseline_v7141_v3_0_7.ipynb|
| -  |  0.6376 |   0.6620   |        | 추정휴가 파생변수 추가 | etri_baseline_v7141_v3_0_6.ipynb|
| -  |  0.6439 |   0.6612   | 0.61513| mLight기반 수면시간 추가 | etri_baseline_v7141_v3_0_4.ipynb|
| -  |  0.6383 |   -  | 0.61617|lightgbm -> catboost 모델변경 | dacon-etri-lifelog-best-score-baseline-v3.ipynb|
| -  |  0.6351 |   0.6584    | 0.62574|lightgbm -> xgboost 모델변경 | dacon-etri-lifelog-best-score-baseline-v3-xgboost.ipynb|
| -  |  0.6553 |   0.6585   | 0.62482|lightgbm+xgboost+catboost valid best score select | basic_sum_data_3_data_3_model_ensemble.ipynb|
| -  |  - |   -   | 0.6269 |lightgbm+xgboost+catboost hard voting | basic_sum_data_3_hard_voting.ipynb|
| 1  |  0.6359 |   0.6620   | 0.62832| 취침전 앱사용통계 파생변수 추가 | etri_baseline_v7141_v3_0_3.ipynb|
| 2  |  0.6322 |   0.6595   | 0.62305| 수면시간 평균 산출로직 변경  | etri_baseline_v7141_v3_0_2.ipynb|
| 4  |  0.6308 |   0.6526   | 0.6176 | parameter common으로 변경  | etri_baseline_v7141_v3_0_1.ipynb|
| 6  |  0.6172 |   0.6463   | 0.60724| 수면시간 lag1, rolling 변수추가 | etri_baseline_v7141_v3_0_0.ipynb|
| 6  |  0.6076 |   0.6540   | 0.60622| NaN -> -1 일괄치환      |dacon-etri-lifelog-best-score-baseline-v2.ipynb|
| 5  |  0.6069 |   0.6491   | 0.60523| 로컬검증 변경      |etri_baseline_v7141_v2.0.0.ipynb|
| 7  |  0.7769 |   0.7679   | 0.52101| target encoding weekend -> weekday      |둘다 개선 퍼블릭 리더보드 하락|

![image](https://github.com/user-attachments/assets/85620ffd-d7ac-4b99-ad74-f58a44d7a8d9)

---------------------------------------------------------------------------------------------------------------------------
### 📦 회의 아젠다

- 논문 title 및 contents 논의  

### 📦 데이터 탐색 인사이트

- id별 추정 휴가일수 학습에서 제외 (학습에서 제외하고 예외 조건으로  post processing 필요)

- awake_blocks 로직 확인 필요

- mLight기반  수면시간 효과 ? 

- 휴가 : 1) 주중 2) 수명시간 > 평균수명시간+1hr 3) 기상시간+1hr>평균기상시간 4) 7-8월 5) id01, id03, id05, id07, id10 6) (금,토,일) 붙여서 휴가 7) 만약 S1이 있다면  전날 S1이 2인 경우
- (불끈 시간 - 핸드폰 이용한 마지막 시간) / 추정 수면시간 = 추정 수면 효율 (S2) <br>

- 평균 취침,기상,수면시간  <br>
![image](https://github.com/user-attachments/assets/e88712f8-3087-4cfc-8199-d5f9b25718d6)

[1] 추정 휴가기간
- id02, id04, id06, id08, id09는 S1(수면의질)이 train 에서 연속해서 발생 (휴가기간)
- id01, id03, id05, id07, id10은 S1이 연속해서 발생하는 케이스 x ->  test 로 샘플 되었을 가능성 존재

<br>
 
[2] S1(level=2) 분포 <br>

![image](https://github.com/user-attachments/assets/cf954df4-8183-4448-84a4-b124454351b8)

### 📦타겟 EDA
![image](https://github.com/user-attachments/assets/8325d54d-a7fc-4007-b786-f8e8f208daf0)

### 📦오차 탐색
![image](https://github.com/user-attachments/assets/5c999889-0843-401a-bbf2-bf15609c5141)

![image](https://github.com/user-attachments/assets/8e81a314-520b-4953-a44b-dacec57e8040)

![image](https://github.com/user-attachments/assets/345c8acc-ac35-4f1d-81fd-70161c43d8d6)


### 📦평가지표 
> Q1: 기상 직후 본인이 인지한 전반적인 수면의 질
 - 0: 개인 평균 이하
 - 1: 개인 평균 이상

> Q2: 취침 직전 본인이 느낀 신체적 피로 수준
 - 0: 높은 피로 수준
 - 1: 낮은 피로 수준
   
> Q3: 취침 직전 본인이 느낀 스트레스 수준
 - 0: 높은 스트레스 수준
 - 1: 낮은 스트레스 수준
   
> S1: 총 수면 시간(TST) 가이드라인을 준수했는지 3LEVELS 
 - 0: 가이드라인 미준수
 - 1: 가이드라인 부분적 준수
 - 2: 가이드라인 완전 준수

> S2: 수면 효율(SE) 가이드라인을 준수했는지 여부 (SE: 잠자리에 누워 있었던 전체 시간 대비, 실제로 잠든 시간의 비율)
 - 0: 가이드라인 미준수
 - 1: 가이드라인 준수

> S3: 수면 잠들기 지연 시간(SOL 또는 SL) 가이드라인을 준수했는지 여부 (SOL: 잠자리에 누운 순간부터 실제로 잠드는 데까지 걸린 시간)
 - 0: 가이드라인 미준수 
 - 1: 가이드라인 준수

### 📦참고자료 
- 1등. 통못자핫도그 (PixleepFlow) – https://github.com/seongjiko/Pixleep 
- 2등. 민바 (TraM) – https://github.com/jin-jae/ETRI-Paper-Contest 
- 3등. VLAB – https://github.com/pknu-v-lab/ETRI_lifelogs 
