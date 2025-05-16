// CNN 모델 목록 

| 모델 이름          | 파라미터 수    | 주요 기술                      | 특징                      | 발표 연도 | 소속 기관          |
|-------------------|---------------|-------------------------------|---------------------------|-----------|-------------------|
| VGG16             | 약 138백만    | 3×3 컨볼루션                   | 단순 구조, 고전적 CNN       | 2014      | Oxford VGG        |
| ResNet50V2        | 약 25.6백만   | 잔차 연결, 사전 활성화          | 깊은 네트워크 최적화        | 2016      | Microsoft Research|
| Xception          | 약 22.9백만   | depthwise separable conv      | Inception의 극한 확장       | 2016      | Google Research   |
| InceptionResNetV2 | 약 55.9백만   | Inception + Residual          | 복잡한 구조, 높은 정확도     | 2016      | Google Research   |
| EfficientNetB0    | 약 5.3백만    | MBConv, compound scaling      | 효율성과 정확도의 균형       | 2019      | Google Research   |
| EfficientNetB7    | 약 66.3백만   | MBConv, compound scaling      | 가장 큰 모델, 높은 정확도    | 2019      | Google Research   |
| ConvNeXtBase      | 약 28.6백만   | depthwise conv, large kernel  | Transformer와 CNN의 결합    | 2022      | Meta FAIR         |

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
