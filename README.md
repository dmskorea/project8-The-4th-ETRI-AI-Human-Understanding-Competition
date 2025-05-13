### 📦로컬 리더보드 
- submit 기준 : 로컬 리더보드 valid, oof(k=5) 보다 성능 향상 된 경우

| #  |  valid  |  oof(k=5)  | test    |       내용       |          코드                  | 
|----|---------|------------|--------|-------------------|--------------------------------|
| 2  |  0.6322 |   0.6595   | 0.62305| 수면시간 평균 산출로직 변경  | etri_baseline_v7141_v3_0_2.ipynb|
| 4  |  0.6308 |   0.6526   | 0.6176 | parameter common으로 변경  | etri_baseline_v7141_v3_0_1.ipynb|
| 6  |  0.6172 |   0.6463   | 0.60724| 수면시간 lag1, rolling 변수추가 | etri_baseline_v7141_v3_0_0.ipynb|
| 6  |  0.6076 |   0.6540   | 0.60622| NaN -> -1 일괄치환      |dacon-etri-lifelog-best-score-baseline-v2.ipynb|
| 5  |  0.6069 |   0.6491   | 0.60523| 로컬검증 변경      |etri_baseline_v7141_v2.0.0.ipynb|
| 7  |  0.7769 |   0.7679   | 0.52101| target encoding weekend -> weekday      |둘다 개선되었으나 퍼블릭 리더보드 하락함 (과적합)|

---------------------------------------------------------------------------------------------------------------------------

### 📦 데이터 탐색

[🔥] 
- 휴가 : 1) 주중 2) 수명시간 > 평균수명시간+1hr 3) 기상시간+1hr>평균기상시간 4) 7-8월 5) id01, id03, id05, id07, id10 6) (금,토,일) 붙여서 휴가 7) 만약 S1이 있다면  전날 S1이 2인 경우 

[🔥] 
- (불끈 시간 - 핸드폰 이용한 마지막 시간) / 추정 수면시간 = 추정 수면 효율 (S2)

[🔥] 평균 취침,기상,수면시간
subject_id	week_type	평균 취침시간	평균 기상시간	평균 수면시간
0	id01	weekday	22:42	05:55	429.2292
1	id01	weekend	22:21	06:09	467.4500
2	id02	weekday	22:54	07:13	496.0000
3	id02	weekend	23:13	07:27	494.9583
4	id03	weekday	00:21	09:03	457.4359
5	id03	weekend	00:18	08:54	450.8667
6	id04	weekday	00:03	06:50	396.6721
7	id04	weekend	00:09	06:59	401.1739
8	id05	weekday	22:52	07:25	500.1064
9	id05	weekend	22:39	07:42	518.2778
10	id06	weekday	23:56	08:30	505.7115
11	id06	weekend	00:33	08:54	490.3500
12	id07	weekday	00:08	07:16	424.9273
13	id07	weekend	00:18	07:56	433.2500
14	id08	weekday	01:35	08:22	360.4906
15	id08	weekend	01:47	08:54	380.7619
16	id09	weekday	00:08	07:47	358.4783
17	id09	weekend	00:19	08:53	383.5294
18	id10	weekday	01:45	08:25	369.9714
19	id10	weekend	01:58	09:33	407.4444

[1]
- id02, id04, id06, id08, id09는 S1(수면의질)이 train 에서 연속해서 발생 (휴가기간)
- id01, id03, id05, id07, id10은 S1이 연속해서 발생하는 케이스 x ->  test 로 샘플 되었을 가능성 존재

[2] S1(level=2) 분포  
![image](https://github.com/user-attachments/assets/8337cefd-7b5b-43ff-b0c8-519e67f3d62a)

[3] 수면시간 관련 파생변수 
![image](https://github.com/user-attachments/assets/b5243181-832c-4b8d-81ed-07f1d139f0f3)


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
