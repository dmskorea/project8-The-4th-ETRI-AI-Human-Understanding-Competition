// CNN 모델 목록 

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
