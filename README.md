# Casting Defect Classification

주조 제품(submersible pump impeller) 이미지를 입력으로 **양품(ok_front)**과 **불량(def_front)**을 자동 판별하는 이진 분류 프로젝트.
EDA → 전처리 → CNN 모델링(베이스라인 + 전이학습) → Streamlit 성능 대시보드 → Grad-CAM 기반 오분류 분석 → 모델 고도화 방향까지 1싸이클을 다룬다.

- **데이터**: Kaggle [`ravirajsinh45/real-life-industrial-dataset-of-casting-product`](https://www.kaggle.com/datasets/ravirajsinh45/real-life-industrial-dataset-of-casting-product) (~100MB, 300x300 grayscale)
- **모델**: Baseline CNN (scratch) vs ResNet18 (ImageNet transfer)
- **환경**: Windows + Python 3.12 + PyTorch 2.4.1 (CUDA 11.8) + NVIDIA RTX 3050 Ti (4GB)

## 폴더 구조

```
Newworld/
  .env.example             # KAGGLE 자격증명 템플릿 (로컬용)
  requirements.txt         # Streamlit Cloud 배포용 (CPU torch)
  requirements-dev.txt     # 로컬 학습/EDA 용 (kaggle, jupyter)
  packages.txt             # (없음 - opencv-python-headless 사용으로 불필요)
  .streamlit/config.toml   # Streamlit 테마/서버 설정
  streamlit_app.py         # ★ Streamlit Cloud 엔트리 (루트)
  README.md
  src/
    config.py              # 경로, 하이퍼파라미터, seed
    utils.py
    download.py            # Kaggle API 다운로드
    dataset.py             # Dataset / Transform / DataLoader
    models.py              # Baseline CNN + ResNet18 + EfficientNet-B0
    losses.py              # FocalLoss
    train.py               # 학습 루프 (AMP, EarlyStopping, Cosine LR)
    evaluate.py            # CM, ROC/PR, threshold sweep
    experiments.py         # 부트스트랩 CI, McNemar
    fn_analysis.py         # 오분류 카테고리/컨센서스
    cam_analysis.py        # Grad-CAM 집계 (per-class mean CAM)
    gradcam.py             # Grad-CAM 유틸
  notebooks/
    01_eda.ipynb
    02_misclassification_analysis.ipynb
  app/
    streamlit_app.py       # 본체 (4탭 대시보드)
  models/
    *.pt                   # 학습된 가중치 (배포 시 함께 커밋)
    metrics.json           # epoch별 train/val 지표
  reports/
    figures/               # EDA·평가 그래프 (정적)
    cam_analysis/{tag}/    # Tab 2 입력 (mean CAM npy + per-image stats)
    misclassified/{FN,FP}/ # 오분류 이미지 (배포 제외)
    *_test_metrics.json
    *_predictions.csv      # Tab 1, Tab 3 입력
    improvement_plan*.md
```

## 빠른 실행 순서

### 0. 사전 준비

- Python 3.11+ 설치 (이 프로젝트에서는 3.12 사용)
- NVIDIA GPU + 드라이버 (CUDA 11.7+ 호환)
- `.env`에 본인의 Kaggle 토큰 입력:

```ini
KAGGLE_USERNAME=<your_username>
KAGGLE_API_TOKEN=<your_token>
```

### 1. 가상환경 + 패키지 설치

**대시보드만 돌려보고 싶다면 (CPU만으로 충분)**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

**학습/평가까지 직접 돌리고 싶다면 (GPU 권장)**:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# CUDA 11.8 wheel 먼저 설치
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-dev.txt
python -c "import torch; print('cuda:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

CUDA가 잡히지 않으면 `cu117` 빌드 (`torch==2.0.1+cu117`)로 폴백.

### 2. 데이터 다운로드

```powershell
python -m src.download
```

`data/raw/casting_data/casting_data/{train,test}/{def_front,ok_front}/` 구조로 자동 해제됩니다.

### 3. EDA

```powershell
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb --output 01_eda.ipynb
```

산출물: `reports/figures/eda_*.png`

### 4. 학습

```powershell
python -m src.train --model baseline --epochs 10
python -m src.train --model resnet18 --epochs 10
```

체크포인트: `models/baseline_cnn.pt`, `models/resnet18_best.pt`, `models/metrics.json` (epoch별 train/val 메트릭)

### 5. 평가

```powershell
python -m src.evaluate --model baseline
python -m src.evaluate --model resnet18
```

산출물:
- `reports/{model}_test_metrics.json` (CM, threshold sweep)
- `reports/{model}_predictions.csv`
- `reports/figures/{model}_confusion_matrix.png`, `{model}_roc.png`, `{model}_pr.png`, `{model}_threshold_sweep.png`
- `reports/misclassified/{FN,FP}/{model}__*` (오분류 이미지 복사)

### 6. 오분류 분석

```powershell
jupyter nbconvert --to notebook --execute notebooks/02_misclassification_analysis.ipynb --output 02_misclassification_analysis.ipynb
```

산출물: `reports/figures/misclassification_*.png` (prob 분포, 픽셀 통계, Grad-CAM)

### 7. Streamlit 대시보드 (로컬)

```powershell
streamlit run streamlit_app.py
# 또는 (구버전 호환): streamlit run app/streamlit_app.py
```

탭 구성:
1. **실험 결과** — 부트스트랩 95% CI, McNemar paired test, 모델 비교, 학습 히스토리
2. **입력 변수 분석 (Grad-CAM)** — 클래스별 평균 어텐션 맵, 7×7 hot spot, 어텐션 통계
3. **오분류 심화 분석** — close vs large margin FN, 컨센서스 오분류 (라벨 의심)
4. **Live Inference** — 이미지 업로드 → 예측 + Grad-CAM

## Streamlit Cloud 배포

이 저장소는 [Streamlit Community Cloud](https://share.streamlit.io)에 그대로 배포할 수 있도록 구성되어 있습니다.

### 핵심 구성

| 항목 | 내용 |
|------|------|
| 엔트리 파일 | `streamlit_app.py` (루트, 자동 인식) |
| 의존성 | `requirements.txt` (CPU PyTorch 2.5.1 휠 사용) |
| OS 패키지 | `packages.txt` (`libgl1` 만 — grad-cam이 끌어오는 non-headless cv2 보호용) |
| Streamlit 설정 | `.streamlit/config.toml` |
| Python 버전 | **3.11 고정** (`runtime.txt` 명시 + Advanced settings 에서 선택 권장) |

### 배포 절차

1. 본 저장소를 GitHub에 push (이미 완료된 상태).
2. [share.streamlit.io](https://share.streamlit.io) 에서 **New app** 클릭.
3. 다음과 같이 입력:
   - **Repository**: `<your_username>/casting-defect-dashboard`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
4. **Advanced settings → Python version 을 `3.11` 로 선택** (★중요).
   - Streamlit Cloud의 기본 Python 은 3.13 인데, 일부 ML 휠이 3.13 을 늦게 지원하므로 3.11이 가장 안정적.
   - 저장소 루트의 `runtime.txt`(`python-3.11`)는 보조 안전장치이며, Advanced settings 선택이 우선합니다.
5. **Deploy** 클릭 → 첫 빌드는 PyTorch + grad-cam 설치 때문에 5–10분 소요.

### 트러블슈팅: "Error installing requirements"

| 증상 | 원인 / 해결 |
|------|------|
| `ERROR: Could not find a version that satisfies the requirement torch==X` | Cloud Python 버전과 휠 호환 안 됨. → Advanced settings 에서 Python **3.11** 로 다시 배포. |
| `ImportError: libGL.so.1: cannot open shared object file` | `opencv-python`(non-headless)이 함께 설치된 경우. → `packages.txt` 에 `libgl1` 추가 (본 저장소 포함됨). |
| `libglib2.0-0 ... Depends libffi7 but not installable` | Cloud의 base OS가 Debian trixie 인데 `libglib2.0-0` 을 bullseye 버전으로 끌어오면서 충돌. → `packages.txt` 에서 `libglib2.0-0` 를 **요청하지 말 것**. 해당 라이브러리는 base image 에 이미 존재합니다. |
| 빌드가 매우 느리거나 OOM | Free tier 자원 부족. → 사용하지 않는 모델 가중치를 `models/` 에서 제거하고 `app/streamlit_app.py` 의 `CHECKPOINTS` 딕셔너리도 정리. |
| `ResolutionImpossible` (numpy 등) | 의존성 충돌. → `pip install -r requirements.txt` 를 로컬 Python 3.11 에서 먼저 검증한 뒤 push. |

### 메모리 / 자원 관련 주의

- Free tier 인스턴스 메모리 ≈ 1GB. 본 앱은 모델 1개당 CPU 메모리 200–400MB 사용.
- `@st.cache_resource` 가 모델을 lazy-load 합니다 — 사용자가 사이드바에서 다른 모델을 고를 때마다 새 모델이 로드되므로 4개 모두 동시에 메모리에 있지는 않습니다.
- 그래도 자원이 빠듯하다면 `app/streamlit_app.py` 상단의 `CHECKPOINTS` 딕셔너리에서 사용하지 않을 모델을 제거하세요.

### 비밀값(secrets)

- 데모 자체는 외부 API 키가 필요 없습니다.
- Kaggle 데이터 재다운로드 기능을 Cloud에서 활성화하려면 Streamlit Cloud 대시보드의 **Secrets** 메뉴에 다음을 추가하세요:

```toml
KAGGLE_USERNAME = "your_username"
KAGGLE_API_TOKEN = "your_token"
```

## 결과 요약 (test set, n=715)

| Model | Accuracy | Precision(def) | Recall(def) | F1(def) | ROC-AUC | FN | FP |
|-------|----------|----------------|-------------|---------|---------|----|----|
| Baseline CNN | 0.9063 | 1.0000 | 0.8521 | 0.9201 | 0.9893 | 67 | 0 |
| ResNet18 | **0.9972** | 1.0000 | 0.9956 | 0.9978 | **1.0000** | 2 | 0 |

- 검사 도메인 관점에서 가장 비싼 오류는 **FN (불량 놓침)** → ResNet18이 67건 → 2건으로 감소.
- 두 모델 모두 **FP=0** (양품을 불량으로 오인하지 않음).
- ResNet18 threshold sweep 결과 **0.25에서 100% acc** — 운영 임계값 조정만으로도 즉시 효과 있음.

상세 고도화 방향은 [`reports/improvement_plan.md`](reports/improvement_plan.md) 참고.

## 주요 설계 결정

- **클래스 가중치**: `CrossEntropyLoss(weight=class_weights)` + `WeightedRandomSampler` 동시 적용으로 모더리트 불균형 (defect 57%) 보정.
- **AMP (혼합정밀)**: 4GB VRAM 환경에서 ResNet18 batch=32 안전하게 학습.
- **보수적 augmentation**: 결함 신호 자체를 왜곡하지 않도록 ColorJitter/Translate를 약하게.
- **EarlyStopping (val F1 patience=5)**: 과적합 방지 + 학습 시간 단축.
- **데이터 파일 누수 차단**: 파일명 prefix가 클래스와 상관 있을 수 있으나, 픽셀-only 파이프라인으로 모델은 파일명 정보를 보지 못함.

## 트러블슈팅

- `cuda available: False` — `pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu118` 재실행. 그래도 안 되면 NVIDIA 드라이버 업데이트.
- `OSError: KAGGLE_USERNAME ...` — `.env` 파일 위치/내용 재확인 (프로젝트 루트).
- VRAM OOM — `src/config.py`에서 `RESNET_BATCH`를 16 또는 8로 낮춤.
- Notebook 실행 실패 — `pip install jupyter ipykernel` 확인 후 재시도.
