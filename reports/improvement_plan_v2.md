# 모델 고도화 방안 v2 (대시보드 인사이트 반영)

> v1과의 차이: 추가 실험(ResNet18+Focal, EfficientNet-B0, TTA)과 통계검증(부트스트랩 CI, McNemar) +
> 전체 테스트셋 Grad-CAM 집계 + 컨센서스 오분류 분석을 통해 얻은 새 인사이트를 반영하여 우선순위와 액션을 재구성.

## 1. 새로 확인된 핵심 사실

### 1-A. 5개 실험 결과 비교 (test n=715, threshold=0.5, 95% bootstrap CI)

| Tag | Accuracy (95% CI) | Recall(def) | F1(def) | TP | TN | FP | FN | 비고 |
|-----|-------------------|-------------|---------|----|----|----|----|------|
| baseline | 0.9063 (0.884-0.929) | 0.852 | 0.920 | 386 | 262 | 0 | 67 | scratch CNN |
| resnet18 | **0.9972 (0.993-1.000)** | 0.996 | 0.998 | 451 | 262 | 0 | **2** | ImageNet transfer |
| resnet18_focal | 0.9972 (0.993-1.000) | 0.996 | 0.998 | 451 | 262 | 0 | 2 | Focal gamma=2 |
| resnet18_tta | 0.9972 (0.993-1.000) | 0.996 | 0.998 | 451 | 262 | 0 | 2 | + HFlip TTA |
| efficientnet_b0 | 0.9972 (0.993-1.000) | 0.996 | 0.998 | 451 | 262 | 0 | 2 | 다른 백본 |

- 4개 강한 모델의 95% CI가 **완전히 겹침** → 통계적으로 구분 불가.
- McNemar 결과: 강한 모델끼리는 b=c=0 (p=1.0) → 오답이 동일.

### 1-B. 컨센서스 오분류 (라벨 의심 후보)
| 파일 | label | mean prob_defect (5개 모델) | 어떤 모델이 놓쳤나 |
|------|-------|------------------------------|---------------------|
| `cast_def_0_150.jpeg` | def_front (1) | 0.240 | **5/5 전부** |
| `cast_def_0_1591.jpeg` | def_front (1) | 0.288 | **5/5 전부** |

→ 모든 모델이 confidently OK로 예측. 이 두 케이스를 잡지 못하는 한 test FN=2의 천장이 존재.

### 1-C. 오분류 카테고리 (close vs large margin, threshold=0.5)
| Tag | FN_close | FN_large | 해석 |
|-----|----------|----------|------|
| baseline | 29 | 38 | close 케이스가 절반 → 임계값 조정 ROI 큼 |
| resnet18 | 1 | 1 | close 1건은 임계값 조정으로 회수 가능 |
| resnet18_tta | 1 | 1 | TTA로 한 케이스 prob이 임계값 근처로 이동 |
| resnet18_focal | **2** | **0** | Focal Loss가 confidence 분포를 평탄화 → 모두 close-margin → 임계값 조정 시 즉시 회수 |
| efficientnet_b0 | 0 | 2 | 다른 백본은 동일 case를 더 확신하고 틀림 |

### 1-D. Grad-CAM 집계 (전체 test set)
- **TP**의 평균 어텐션은 TN 대비 cam_p90 평균이 수 배 강함 → 모델이 결함 영역에 일관되게 집중.
- **TN**의 어텐션은 분산되어 약함 (구체적 결함 신호 없음).
- **FN 2건의 어텐션**: TP 패턴과 유사한 위치에 약한 신호 → 모델이 결함을 보긴 했으나 confidence를 충분히 끌어올리지 못함.
- 7×7 region grid에서 TP의 hot spot은 이미지 중앙·중상단에 집중 → 결함의 spatial bias 존재 (카메라 표준화로 활용 가능).

## 2. v1 대비 우선순위 변경

| 순위 v1 → v2 | 액션 | 변경 이유 |
|--------------|------|-----------|
| **신규 P0** | **컨센서스 오분류 2건 라벨 검수** | 모든 모델이 동일하게 틀림 → 데이터 측 문제일 가능성 ↑. 모델 측 노력 ROI 낮음 |
| **신규 P0** | **앙상블 (RN18 + RN18_focal + EffNetB0)** + Temperature scaling | 다른 백본/손실 조합이 4개 강한 모델의 분산을 줄여 calibration 확보 가능 |
| P0 유지 | TTA 운영 적용 | 무비용, 안정성 ↑ |
| P0 유지 | Threshold tuning (0.20-0.35 권장) | sweep 결과 ResNet18은 0.25에서 100% acc |
| P1 강화 | **Multi-seed 학습 (3-5 seed)** | 단일 시드 결과의 분산 정량화 — 현재 4개 모델 동률은 시드 효과 가능성도 있음 |
| P1 유지 | EfficientNet-B0 / 더 큰 백본 | 단일 모델로는 한계가 명확. 앙상블 일원으로 활용 |
| P1 유지 | Focal Loss | 단독 효과는 미미하나 close-margin 회수에 유리 → 운영 모델 후보 |
| P2 유지 | Albumentations / 합성 결함 증강 | 데이터 보강 |
| P2 유지 | YOLOv8 결함 위치 검출 | 운영 가치 ↑ |

## 3. 새 액션 상세

### 3-A. 라벨 검수 프로토콜
1. 컨센서스 오분류 2건을 도메인 전문가가 시각 확인.
2. 라벨 정정 시 즉시 test set 메트릭 재계산 (현재 평가 파이프라인 그대로 사용).
3. 정정 결과를 별도 `reports/labels_audit.csv`에 기록.
4. 학습 셋 전체에 대해서도 유사한 컨센서스 룰(예: 3개 이상 모델이 자신 있게 틀리는 케이스)을 적용해 label noise 후보 추가 탐색.

### 3-B. 앙상블 + Calibration
- 단순 평균 앙상블: `prob_ensemble = mean(prob_resnet18, prob_focal, prob_effnet_b0)`
- Temperature scaling (val set fit): `T*` 추정 후 production 확률 보정.
- 회색지대 정의: `prob_ensemble ∈ [0.20, 0.40]` → Human-in-the-loop 큐.

### 3-C. Multi-seed 실험 설계
- ResNet18 + CE 기본 설정으로 seed ∈ {0, 1, 2, 3, 4} 학습.
- 각 시드 test metric의 mean±std와 부트스트랩 CI 비교.
- 목적: 현재 "4개 모델 동일 결과"가 모델 측 saturation인지, 시드 우연인지 구분.

### 3-D. CAM-guided 데이터 보강
- TP 평균 어텐션 영역을 OK 이미지에 합성하여 hard negative 생성.
- FN 2건의 어텐션 영역을 augmentation 시 CutOut으로 가린 augment를 추가 → 모델이 다른 region도 보도록 강제.

### 3-E. Detection 확장 (장기)
- 컨센서스 오분류와 close-margin FN들을 우선 라벨링 (박스).
- YOLOv8n으로 학습 → 분류 모델과 cascade로 운영.
- "분류기는 강하지만 위치를 모름" → 자동 위치 표시로 검사원 신뢰성 ↑.

## 4. 측정 가능한 목표 (KPI)

- 1차 (P0 완료 시점):
  - 라벨 검수 후 test 재평가: 새 FN 수, 새 Recall(def) ≥ 0.999
  - 앙상블 calibration: ECE (Expected Calibration Error) < 0.02
- 2차 (P1 완료):
  - Multi-seed 5회: F1(def) std < 0.005
- 운영:
  - 회색지대 비율 ≤ 5%
  - 드리프트 알람 ≤ 1회/일
  - human review queue 처리 시간 ≤ 1분/건

## 5. 결론
- 단일 모델 측면에서는 본 데이터셋이 사실상 **모델로는 더 짤 수 없는 천장**에 도달.
- 추가 향상의 **레버리지는 "데이터 품질(라벨)"과 "운영 정책(임계값/앙상블/HITL)"** 쪽에 있음.
- 향후 가치는 **분류 → 위치 검출**로 task 자체를 확장할 때 가장 크게 나타날 것.
