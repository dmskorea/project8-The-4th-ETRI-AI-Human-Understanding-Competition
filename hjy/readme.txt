☆ 테스트 데이터와 유사한 검증 데이터셋 만들기 
☆ target lag feature 추가 lag -1 -2 -3 전날 건강상태 변수 
☆ id06 S1준수o vs id08 S1준수x

# backbone_name='resnext101_32x32d'
# backbone_name='resnet18'
# resnet34	resnet18보다 깊지만 속도 손해 적음
# resnet50	널리 검증된 모델, 일반적인 이미지 작업에 매우 안정적
# efficientnet_b0	파라미터 수 적고 정확도 높음, 모바일/경량 환경에 적합
# convnext_tiny	최신 ConvNet 구조로 성능 높고 연산 효율적
# resnet101	더 깊은 네트워크로 복잡한 패턴에 강함
# efficientnet_b3 ~ b5	성능은 뛰어나지만 학습 시간이 더 큼
# convnext_base	ConvNet 중 최근 가장 높은 성능
# beit_base_patch16_224	Vision Transformer 기반, 사전학습 필수

# ============================================================================================================================================================

[1]

# X shape: (450, 210)
# test_X shape: (250, 210)

 STEP1: 실험 결과 확인
=============== Validation Results ==============
[test_size=0.5] 평균 F1: 0.6536 / [상세] Q1(기상직후수면질):0.5714 Q2(취침전신체적피로):0.6549 Q3(취침전스트레스):0.7267 S2(수면효율):0.7953 S3(수면잠들기시간):0.8012 S1(S1):0.3718
[test_size=0.4] 평균 F1: 0.6614 / [상세] Q1(기상직후수면질):0.5629 Q2(취침전신체적피로):0.6781 Q3(취침전스트레스):0.7149 S2(수면효율):0.8015 S3(수면잠들기시간):0.7955 S1(S1):0.4155
[test_size=0.3] 평균 F1: 0.6605 / [상세] Q1(기상직후수면질):0.5581 Q2(취침전신체적피로):0.6667 Q3(취침전스트레스):0.7363 S2(수면효율):0.7817 S3(수면잠들기시간):0.7822 S1(S1):0.4377
[test_size=0.2] 평균 F1: 0.6709 / [상세] Q1(기상직후수면질):0.5952 Q2(취침전신체적피로):0.6226 Q3(취침전스트레스):0.7344 S2(수면효율):0.7692 S3(수면잠들기시간):0.8235 S1(S1):0.4807

[best_iteration_dict]
Q1: 496
Q2: 408
Q3: 344
S2: 857
S3: 641
S1: 919

[iteration 변화 그래프 저장]
# 전체 평균 F1: 0.6616
================================================

 STEP2: 전체 데이터로 모델 재학습
====== modoling with 100% train & no valid =====
[Q1] avg_rssi(352), device_class_others_ratio_worktime(343), sleep_duration_min_mScreenStatus(295), wake_time_min_mScreenStatus(274), rssi_mean_sleeptime(273), 통화_time(267), light_night_mean(267), device_class_0_ratio_worktime(253), hr_early_morning_std(246), walk_minutes(237)
[Q2] speed_le5_max(273), total_screen_time(272), charging_ratio(241), Q2_te(216), walk_minutes(214), sleeptime_entropy(208), rssi_mean_afterwork(205), hr_afternoon_mean(204), afterwork_sum_human_related(176), activity_7_ratio(174)
[Q3] device_class_0_ratio_afterwork(231), total_screen_time(207), unique_app_count(197), hr_evening_mean(196), screen_time_vs_avg_pct(191), rssi_mean_afterwork(183), OneUI홈_time(180), avg_rssi(162), wake_time_min_mScreenStatus(147), light_mean(145)
[S2] avg_rssi(540), hr_morning_min(506), wake_time_min_mScreenStatus(502), sleeptime_entropy(473), afterwork_entropy(453), sleep_duration_min_mScreenStatus(440), 전화_time(436), screen_on_ratio(396), hr_morning_mean(395), worktime_entropy(381)
[S3] max_rssi(449), wlight_evening_mean(388), device_class_0_ratio_sleeptime(366), rssi_mean_afterwork(364), wlight_early_morning_max(358), S3_te(351), rssi_min_afterwork(332), avg_rssi(329), sleep_time_min_min(299), sleeptime_entropy(277)
[S1] sleep_duration_min_mScreenStatus(1479), wake_time_min_mScreenStatus(1371), rssi_mean_afterwork(1254), worktime_entropy(1198), hr_morning_min(1174), active_minutes(1088), rssi_max_sleeptime(1077), screen_on_duration_avg(973), total_screen_time(968), speed_le5_max(944)
# submission_0.661585461523865.csv 저장 완료
# submission shape:(250, 9)
================================================

 STEP3: 예측결과 비교표
학습sum	학습len	학습mean	테스트sum	테스트len	테스트mean
Q1	223	450	0.4956	128	250	0.5120
Q2	253	450	0.5622	174	250	0.6960
Q3	270	450	0.6000	203	250	0.8120
S1	390	450	0.8667	188	250	0.7520
S2	293	450	0.6511	192	250	0.7680
S3	298	450	0.6622	198	250	0.7920

[a]
낮에 활동지수 
- 얼마나 움직였는지
- 기계장치가 인체에 미치는 영향 

상수값
- 이사람이 평소에 잠을 잘 자는 사람인지

중간에 깼는지 안깻는지 체크하는 변수 
- 잠자려고 불을 끄고 새벽에 다시 켰는지 
- 새벽에 움직임이 있었는지 
- 새벽에 외부소리가 발생했는지 (티비소리, 코골이 등) 
- 새벽에 화장실 다녀온 경우 

[1] 주중,주말 다름 
각각의 ID의 주중 주말 취침 시간 기상 시간 산출 하기 
- 검증하기 (ID 10개니깐 검증하기 쉬움) 

[2] S1,S2,S3
취침시간에 소리 불빛 움직임이 발생했는지 
- 총수면시간, 수면효율, 수면잠들기시간 

[3] Q1,Q2,Q3
데이라이프 

// 취침전 신체적피로 : 걸음수, 심박동수, 추정행동, 핸드폰위치

◎ 가설1: 걸음수가 많은날 참가자들이 신체적피로도가 높았는지? 
◎ 가설2: 심박동수가 높은날 참가자들이 신체적피로도가 높았는지? 
◎ 가설3: 
  - 추정행동(걷기) 장시간   -> 신체적피로도가 높았는지? 
  - 추정행동(달리기) 장시간 -> 신체적피로도가 높았는지? 
  - 추정해동(자동차) 장시간 -> 신체적피로도가 높았는지? 
◎ 가설4: 
  - 핸드폰 위치 GPS (걷기) 장시간 -> 신체적피로도가 높았는지? 
  - 핸드폰 위치 GPS (조깅) 장시간 -> 신체적피로도가 높았는지? 
  - 핸드폰 위치 GPS (자동차) 장시간 -> 신체적피로도가 높았는지? 

// 취침전 스트레스 : 걸음수, ,추정주변소리, 심박동수

◎ 가설1: 걸음수가 많은날 참가자들이 스트레스가 높았는지? 
◎ 가설2: 추정주변소리가 많거나 or 장시간 지속되거나 스트레스를 받는지? 
◎ 가설3: 심박동수가 증가가 스트레스와 관련이 있는지? 즐거워도 심박동수는 증가... 스트레스를 받아도 증가.. 
-> 심박동수를 분석해서 구분가능하다고 함,,

# 걸음수가 커질수록 신체적피로 Q2-> 1 낮은 스트레스를 받는다는 의미  (10명 중 8명)
# 걸음수가 많을수록 대체적으로 신체적 피로도는 낮음 
train_df[['subject_id','step_sum','Q2']].groupby(['subject_id']).apply(lambda x: pd.Series({
     '상관관계':x[['step_sum','Q2']].corr().iloc[0,1]
    ,'걸음수' : x['step_sum'].mean()
    ,'신체적피로': x['Q2'].mean()
})).reset_index().sort_values(['상관관계'])
"""
subject_id	상관관계	걸음수	신체적피로
1	id02	-0.5199	4896.3617	0.6667
8	id09	-0.4044	4117.6098	0.4390
4	id05	-0.2111	4845.0857	0.4318
7	id08	-0.1508	2303.9143	0.7857
6	id07	-0.1146	4230.8980	0.5102
2	id03	-0.1141	6660.2121	0.8182
0	id01	-0.0784	3384.3000	0.5610
9	id10	-0.0614	3383.5152	0.5455
5	id06	0.0971	2976.5556	0.3958
3	id04	0.1717	3784.1957	0.4912
"""

# ============================================================================================================================================================

Q1. 기상 직후 수면의 질

활동 패턴:
afterwork_activity_3_ratio (휴식 시간 비율)
sleeptime_activity_3_count (수면 중 움직임 횟수)

주변 환경:
sleeptime_entropy (수면 중 주변 소리 복잡도)
sleeptime_sum_human_related (수면 중 인간 활동 관련 소리)

밝기 영향:
light_night_mean (수면 중 평균 밝기)
light_night_ratio (야간 밝기 비율)

화면 사용:
screen_on_ratio (취침 전 화면 사용 비율)
YouTube_time (취침 전 YouTube 사용 시간)

Q2. 취침 전 피로 수준

신체 활동:
step_sum (하루 총 걸음 수)
worktime_activity_7_count (근무 중 걷기 횟수)

심박수:
hr_evening_above_100_ratio (저녁 심박수 100 이상 비율)

피로 지표:
burned_calories_sum (하루 소모 칼로리)
active_minutes (활동 시간)

Q3. 취침 전 스트레스 수준

주변 소음:
afterwork_has_human_related (퇴근 후 인간 활동 소음 유무)
worktime_sum_vehicle_related (근무 중 차량 소음)

심리적 부하:
통화_time (전화 사용 시간)
메신저_time (메신저 사용 시간)

생리적 반응:
hr_evening_std (저녁 심박수 변동성)

S1. 총 수면 시간

수면 환경:
sleep_time_hhmm_x (취침 시간)
wake_up_early_minutes (조기 기상 시간)

활동 영향:
vehicle_minutes (차량 이동 시간)
worktime_activity_0_ratio (근무 중 차량 탑승 비율)

화면 사용:
total_screen_time (하루 총 화면 사용 시간)

S2. 수면 효율

수면 중 환경:
sleeptime_max_prob (수면 중 가장 빈번한 소리 유형)
sleeptime_label_count (수면 중 감지된 소리 종류 수)

생체 신호:
hr_early_morning_min (새벽 최저 심박수)

기기 사용:
charging_transitions (충전 상태 변화 횟수)

S3. 수면 잠들기 (지연) 시간

취침 전 활동:
afterwork_activity_8_ratio (퇴근 후 달리기 비율)
speed_mean (평균 이동 속도)

빛 노출:
wlight_evening_mean (저녁 평균 밝기)

심리적 요인:

Q3 (직접적인 스트레스 수준)
screen_on_transitions (화면 켜짐 횟수)
