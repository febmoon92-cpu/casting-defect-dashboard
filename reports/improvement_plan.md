# 모델 고도화 방향 (Casting Defect Classification)

## 0. 현재 성능 요약 (test set, n=715)

| 모델 | Accuracy | Precision(def) | Recall(def) | F1(def) | ROC-AUC | TN | FP | FN | TP |
|------|----------|----------------|-------------|---------|---------|----|----|----|----|
| Baseline CNN | 0.9063 | 1.0000 | 0.8521 | 0.9201 | 0.9893 | 262 | 0 | **67** | 386 |
| ResNet18 (transfer) | **0.9972** | 1.0000 | 0.9956 | 0.9978 | **1.0000** | 262 | 0 | **2** | 451 |

핵심 관찰:
- 두 모델 모두 **False Positive = 0** (양품을 불량으로 잘못 통과시킨 사례 없음).
- 차이는 **False Negative (불량 놓침)** — 검사 도메인에서 가장 치명적인 오류 유형.
- ResNet18 threshold sweep을 보면 **threshold=0.25에서 acc/F1=1.0** → 임계값 튜닝만으로 현재 테스트셋 100% 가능 (단, 일반화 검증 필요).

## 1. 데이터 관점

### 1-A. FN 케이스 보강
- 현재 ResNet18이 놓친 2장은 모델이 약한 신호 영역을 보고 있을 가능성 (Grad-CAM 검증 결과 참고).
- 동일 유형의 결함(특히 약한 텍스처 변화) 샘플을 **수집/보강**.
- 가능하다면 결함 유형(균열, 기공, 변형 등) 메타라벨을 추가해 multi-label 또는 stratified 분석.

### 1-B. 증강 정책 다양화
- 현재: HFlip / Rotation15 / ColorJitter / Translate — 보수적.
- 추가 후보:
  - **Albumentations**: ElasticTransform, GridDistortion, CoarseDropout — 결함 위치 가림으로 모델 강건성 ↑
  - **MixUp / CutMix** — 일반화 향상, 다만 정상 패치가 결함 위에 덮이면 라벨 노이즈 위험 → 신중하게 alpha 설정
  - **Brightness/Gamma jitter 확장** — 조명 변동 대응

### 1-C. 합성 결함 증강
- OK 이미지에 결함 패턴(crack mask)을 합성하여 클래스 균형 조정.
- 단, 분포 mismatch 위험 → 검증 셋엔 원본만 사용.

## 2. 모델 관점

### 2-A. 더 큰 백본
| 모델 | 파라미터 | VRAM (224, batch 32, AMP) | 기대 효과 |
|------|----------|---------------------------|-----------|
| EfficientNet-B0 | ~5M | ~1GB | 더 효율적인 표현, 약간 더 높은 성능 |
| ResNet34 | ~21M | ~1.5GB | 깊이 확대 |
| ConvNeXt-Tiny | ~28M | ~2.5GB | 최신 SoTA에 가까움 |

VRAM 4GB 환경에서는 EfficientNet-B0가 가성비 우수.

### 2-B. 손실 함수 / 학습 전략
- **Focal Loss(gamma=2)**: 쉬운 다수 샘플 가중치를 낮춰 어려운 FN에 집중.
- **Label Smoothing(0.05)**: 과신 방지로 일반화 향상.
- **2-stage fine-tuning**: (1) head만 학습 → (2) 전체 unfreeze + 낮은 lr — 사전학습 표현 보존.

### 2-C. TTA (Test-Time Augmentation)
- 단순 평균: original + HFlip + Rotate(±5°) → 결정 확률 안정화.
- 적은 비용으로 FN 1~2건 추가 회수 기대.

### 2-D. 앙상블
- ResNet18 + EfficientNet-B0 확률 평균.
- 결함 도메인 특성상 다른 inductive bias 결합 시 robust.

## 3. 운영 관점

### 3-A. 임계값 정책
- 현재 0.5 기본값 → ResNet18 sweep 결과 **0.25에서 100% acc** (테스트셋).
- 운영 시 **threshold=0.20~0.25 권장**, 단 추가 hold-out에서 PR 곡선 재확인.
- **Conservative mode**: prob ∈ [0.1, 0.4] 구간을 "회색지대"로 두고 **Human-in-the-loop 검토 큐**로 라우팅.

### 3-B. 신뢰도 보정 (calibration)
- **Temperature scaling** 또는 Platt scaling으로 prob 분포를 보정 → 임계값 의미가 더 직관적.
- 회색지대 정의 시 calibration이 안정적 의사결정 근거.

### 3-C. 모니터링 & 드리프트 감지
- 운영 환경에서 다음을 주기적으로 측정:
  - 입력 이미지 픽셀 평균/표준편차 분포 변화 (PSI/KS test)
  - 모델 출력 확률 분포 변화
  - 사람 검토 큐 결과로 정량적 recall 재추정
- 임계값 자동 조정 트리거.

### 3-D. 데이터 라벨링 파이프라인
- 운영 중 발견되는 FN/FP 케이스 → 자동 수집 → 분기/월 단위 재학습.
- Active learning: 불확실한 prob ≈ 0.5 샘플 우선 라벨링.

## 4. 확장 (분류 → 검출)

현재는 "있다/없다" 이진 분류. 운영자는 종종 "어디에 결함이 있는가"도 알고 싶어함.
- **단기**: Grad-CAM 오버레이를 검사 UI에 노출 (Streamlit 대시보드에 구현됨).
- **중기**: 일부 샘플에 결함 박스 라벨링 → **YOLOv8** object detection으로 task 확장.
- **장기**: pixel-level 라벨 → Segmentation (U-Net, Segformer) — 결함 면적 정량화 가능.

## 5. 우선순위 로드맵

| 우선순위 | 액션 | 예상 효과 | 비용 |
|----------|------|-----------|------|
| P0 | 운영 임계값 0.25로 재평가 + calibration | FN 즉시 감소 | 매우 낮음 |
| P0 | TTA 적용 | 추가 FN 회수 | 낮음 |
| P1 | EfficientNet-B0 fine-tune & 앙상블 | 일반화 ↑ | 중 |
| P1 | Focal Loss 재학습 | 어려운 FN 집중 | 중 |
| P2 | Albumentations + 합성 결함 데이터 보강 | robustness ↑ | 중 |
| P2 | YOLOv8 결함 위치 검출 라벨링·학습 | 운영 가치 ↑ | 높음 |
| P3 | Calibration + drift monitoring pipeline | 운영 안정성 | 중 |

## 6. 측정 지표 (KPI)

- 1차: **Recall(defect) ≥ 0.99**, FP 유지 ≤ 0.5%
- 2차: PR-AUC ≥ 0.999, ROC-AUC ≥ 0.999
- 운영: 회색지대(human review) 비율 ≤ 5%, 일일 drift 알람 ≤ 1회
