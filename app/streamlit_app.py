"""Casting Defect Dashboard (v2) - 3 sophisticated tabs + bonus live inference.

Tabs:
  1) 실험 결과         - 부트스트랩 CI, McNemar, 방법론 효과 해석, 향후 방안
  2) 입력 변수 분석    - Grad-CAM 집계(클래스별 평균 어텐션, hot spot, 집중도)
  3) 오분류 심화 분석  - close vs large margin, 라벨 의심, 컨센서스 오분류
  +  Live Inference   - 업로드 이미지 추론
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.experiments import collect_runs, pairwise_mcnemar, summarize_runs  # noqa: E402
from src.fn_analysis import categorize_errors, consensus_misclassified  # noqa: E402
from src.gradcam import load_model_for_cam, overlay_for_image  # noqa: E402
from src.utils import get_device  # noqa: E402


st.set_page_config(page_title="Casting Defect Dashboard (v2)", layout="wide")


# ----------- caching -----------


@st.cache_resource
def get_device_cached() -> torch.device:
    return get_device()


@st.cache_data
def cached_frames() -> dict[str, pd.DataFrame]:
    return collect_runs()


@st.cache_data
def cached_summary(threshold: float, n_iter: int = 500) -> pd.DataFrame:
    return summarize_runs(cached_frames(), threshold=threshold, n_iter=n_iter)


@st.cache_data
def cached_mcnemar(threshold: float) -> pd.DataFrame:
    return pairwise_mcnemar(cached_frames(), threshold=threshold)


@st.cache_data
def cached_history() -> dict | None:
    p = config.MODELS_DIR / "metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


@st.cache_data
def cached_cam_arrays(tag: str) -> dict | None:
    d = config.REPORTS_DIR / "cam_analysis" / tag
    if not d.exists():
        return None
    out: dict = {}
    for g in ("TP", "TN", "FN", "FP"):
        f = d / f"mean_cam_{g}.npy"
        if f.exists():
            out[f"mean_{g}"] = np.load(f)
        rf = d / f"region_grid_{g}.npy"
        if rf.exists():
            out[f"region_{g}"] = np.load(rf)
    stats = d / "per_image_stats.csv"
    if stats.exists():
        out["stats"] = pd.read_csv(stats)
    counts = d / "group_counts.json"
    if counts.exists():
        out["counts"] = json.loads(counts.read_text(encoding="utf-8"))
    return out


@st.cache_resource
def cached_model(model_name: str, ckpt_name: str):
    ckpt = config.MODELS_DIR / ckpt_name
    if not ckpt.exists():
        return None
    return load_model_for_cam(model_name, ckpt, get_device_cached())


CHECKPOINTS = {
    "baseline": ("baseline", "baseline_cnn.pt"),
    "resnet18": ("resnet18", "resnet18_best.pt"),
    "resnet18_focal": ("resnet18", "resnet18_focal.pt"),
    "resnet18_tta": ("resnet18", "resnet18_best.pt"),  # same weights as resnet18
    "efficientnet_b0": ("efficientnet_b0", "efficientnet_b0.pt"),
}


# ----------- image path resolution -----------
#
# Prediction CSVs were generated locally and store absolute Windows paths
# under data/raw/, which do not exist on Streamlit Cloud (the raw Kaggle
# dataset is excluded from git). We fall back to:
#   1. The original recorded path (works on the original machine).
#   2. reports/sample_images/<basename>  (curated demo set, committed).
#   3. data/raw/  recursive search by basename (works after a local
#      `python -m src.download`).
# If all three miss, the caller renders a small placeholder card with the
# row's metadata so the dashboard remains useful.

SAMPLE_IMAGES_DIR = config.REPORTS_DIR / "sample_images"


@st.cache_data
def _data_raw_index() -> dict[str, str]:
    """Index basename -> absolute path for any image under data/raw/.

    Cached so the recursive scan happens at most once per session.
    Returns an empty dict on Streamlit Cloud where data/raw is absent.
    """
    raw = config.RAW_DIR
    if not raw.exists():
        return {}
    return {p.name: str(p) for p in raw.rglob("*.jp*g")}


def _resolve_image_path(path: str | Path) -> Path | None:
    p = Path(str(path))
    if p.exists():
        return p
    sample = SAMPLE_IMAGES_DIR / p.name
    if sample.exists():
        return sample
    idx = _data_raw_index()
    hit = idx.get(p.name)
    if hit:
        return Path(hit)
    return None


# ----------- tab 1: experiments -----------


def tab_experiments() -> None:
    st.header("Tab 1 - 실험 결과 (Statistical comparison)")

    frames = cached_frames()
    if not frames:
        st.error("Run `python -m src.evaluate` for each model first.")
        return

    threshold = st.slider("Decision threshold for defect", 0.05, 0.95, 0.5, 0.05, key="exp_thr")
    summary = cached_summary(threshold, n_iter=500)

    st.subheader("실험 요약 + 부트스트랩 95% 신뢰구간")
    metric_cols = ["accuracy", "precision_defect", "recall_defect", "f1_defect", "roc_auc", "pr_auc"]
    show = summary[metric_cols + [c + "_lo" for c in metric_cols] + [c + "_hi" for c in metric_cols] + ["TP", "TN", "FP", "FN", "n"]].copy()
    pretty = pd.DataFrame(index=show.index)
    for m in metric_cols:
        pretty[m] = show.apply(lambda r, m=m: f"{r[m]:.4f}  ({r[m+'_lo']:.4f}-{r[m+'_hi']:.4f})", axis=1)
    pretty[["TP", "TN", "FP", "FN"]] = show[["TP", "TN", "FP", "FN"]]
    st.dataframe(pretty, use_container_width=True)
    st.caption(f"95% CI = percentile bootstrap (500 resamples). threshold={threshold:.2f}")

    st.subheader("핵심 지표 비교 (point + 95% CI)")
    metric_to_show = st.selectbox("Metric", metric_cols, index=3, key="exp_metric_pick")
    df_plot = summary.reset_index()[["tag", metric_to_show, f"{metric_to_show}_lo", f"{metric_to_show}_hi"]].copy()
    df_plot["err_low"] = df_plot[metric_to_show] - df_plot[f"{metric_to_show}_lo"]
    df_plot["err_high"] = df_plot[f"{metric_to_show}_hi"] - df_plot[metric_to_show]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_plot["tag"],
        y=df_plot[metric_to_show],
        error_y=dict(type="data", symmetric=False, array=df_plot["err_high"], arrayminus=df_plot["err_low"]),
        text=df_plot[metric_to_show].round(4),
        textposition="outside",
    ))
    fig.update_layout(yaxis_range=[max(0.0, df_plot[metric_to_show].min() - 0.05), 1.02])
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("McNemar paired test (오답이 분포가 다른가?)")
    st.caption("p_value < 0.05 이면 두 모델의 오답 분포가 통계적으로 유의하게 다름. b/c는 한 쪽만 맞은 케이스 수.")
    mc = cached_mcnemar(threshold)
    if mc.empty:
        st.info("필요한 prediction CSV가 부족합니다.")
    else:
        st.dataframe(mc.round(4), use_container_width=True)

    st.subheader("학습 히스토리")
    hist = cached_history()
    if hist:
        rows = []
        for tag, payload in hist.items():
            for h in payload["history"]:
                rows.append({"tag": tag, "epoch": h["epoch"], "val_loss": h["val"]["loss"], "val_f1": h["val"]["f1"]})
        big = pd.DataFrame(rows)
        c1, c2 = st.columns(2)
        with c1:
            st.plotly_chart(px.line(big, x="epoch", y="val_loss", color="tag", markers=True, title="val loss"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(big, x="epoch", y="val_f1", color="tag", markers=True, title="val F1"), use_container_width=True)

    st.subheader("시도한 방법론 해석")
    st.markdown(
        """
        | 방법 | 핵심 | 결과 |
        |------|------|------|
        | Baseline CNN (3 conv blocks) | Scratch 학습 | F1 0.92, FN 67 → 한계 명확 |
        | ResNet18 (ImageNet pretrained) | Transfer learning, AMP, Cosine LR | F1 0.998, FN 2 — **압도적 향상** |
        | ResNet18 + Focal Loss (gamma=2) | 어려운 샘플 가중치 ↑ | 동일 F1, FN 모두 close-margin으로 이동 → 임계값 조정에 더 민감 |
        | ResNet18 + TTA (HFlip 평균) | 무비용 robustness | 동일 결과 (한 케이스 prob 가 임계값에 약간 더 가까워짐) |
        | EfficientNet-B0 | 다른 백본 비교 | 동일 F1, FN은 large-margin 2건 (다른 inductive bias) |

        **공통 관찰**: 강한 모델 4종이 모두 **같은 2건의 FN을 놓침**. → 이 케이스들은 모델 아키텍처/손실 변경으로 해결되지 않음 → 데이터 자체에 가까운 문제.
        """
    )

    st.subheader("향후 시도해볼 만한 방안")
    st.markdown(
        """
        1. **라벨 검수**: 컨센서스 오분류 2건 (`cast_def_0_150.jpeg`, `cast_def_0_1591.jpeg`)를 도메인 전문가가 재확인. 라벨 정정 시 모델 평가 다시.
        2. **앙상블 + Calibration**: 4개 강한 모델 확률 평균 + Temperature scaling → 회색지대 정의.
        3. **시드 다회 학습 (Multi-seed)**: 동일 설정으로 3-5 시드 → CI 폭이 좁아지는지 확인하여 결과 신뢰성 강화.
        4. **Stronger augmentation (Albumentations)**: ElasticTransform / GridDistortion 도입.
        5. **Cross-validation**: train+val을 5-fold로 묶어 분산 검증.
        6. **Multi-task**: 결함 위치(박스 또는 마스크) 일부만 라벨링하면 → 분류 성능과 해석성 동시에 ↑.
        """
    )


# ----------- tab 2: variable / Grad-CAM analysis -----------


def _heatmap_fig(arr: np.ndarray, title: str, colorscale: str = "Inferno") -> go.Figure:
    fig = go.Figure(data=go.Heatmap(z=arr, colorscale=colorscale, showscale=True))
    fig.update_layout(title=title, height=380, yaxis=dict(autorange="reversed"))
    return fig


def tab_cam_analysis() -> None:
    st.header("Tab 2 - 입력 변수 분석 (딥러닝 → Grad-CAM 기반)")
    st.caption("이미지 데이터이므로 'feature importance' 대신 모델이 어떤 픽셀 영역을 보고 결함을 판단했는지를 분석합니다.")

    tag = st.selectbox("분석 대상 모델", list(CHECKPOINTS.keys()), index=1, key="cam_tag")
    cam = cached_cam_arrays(tag)
    if cam is None:
        st.warning(f"`python -m src.cam_analysis --tag {tag}` 를 먼저 실행하세요.")
        return

    counts = cam.get("counts", {})
    st.write({k: int(v) for k, v in counts.items()})

    st.subheader("클래스별 평균 어텐션 맵 (test set 전체 평균)")
    cols = st.columns(4)
    for col, group in zip(cols, ("TP", "TN", "FN", "FP")):
        arr = cam.get(f"mean_{group}")
        if arr is None or counts.get(group, 0) == 0:
            with col:
                st.info(f"{group}: 없음")
            continue
        with col:
            st.plotly_chart(_heatmap_fig(arr, f"mean CAM | {group} (n={counts.get(group, 0)})"), use_container_width=True)
    st.markdown(
        """
        - **TP** (정확히 잡은 불량): 모델이 일관되게 보는 영역. 결함 신호의 spatial prior에 해당.
        - **TN** (정확히 통과시킨 양품): 어텐션이 분산되어 약함 → 강한 결함 신호 없음을 의미.
        - **FN** (놓친 불량): TP 영역과 일관성이 있다면 모델은 결함을 봤으나 confidence가 낮은 경우.
        """
    )

    st.subheader("7×7 영역 grid hot spot")
    cols = st.columns(4)
    for col, group in zip(cols, ("TP", "TN", "FN", "FP")):
        arr = cam.get(f"region_{group}")
        if arr is None or counts.get(group, 0) == 0:
            with col:
                st.info(f"{group}: 없음")
            continue
        with col:
            st.plotly_chart(_heatmap_fig(arr, f"region grid | {group}"), use_container_width=True)
    st.caption("이미지를 7×7 그리드로 나눈 평균 어텐션. 결함이 자주 나타나는 위치를 확인합니다.")

    st.subheader("샘플별 어텐션 통계 분포")
    stats = cam.get("stats")
    if stats is None or stats.empty:
        st.info("per_image_stats.csv 없음")
        return

    metric = st.selectbox("통계 지표", ["cam_mean", "cam_max", "cam_p90", "cam_entropy"], index=2, key="cam_stat")
    desc = {
        "cam_mean": "이미지 평균 어텐션 (높을수록 전체적으로 강한 반응)",
        "cam_max": "어텐션 최댓값",
        "cam_p90": "상위 10% 분위 어텐션 (강한 hot spot 강도)",
        "cam_entropy": "어텐션 엔트로피 (높을수록 분산, 낮을수록 한 곳에 집중)",
    }
    st.caption(desc[metric])
    fig = px.box(stats, x="group", y=metric, color="group", points="all")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("어텐션 중심점 (centroid) 산점도")
    fig = px.scatter(
        stats, x="cam_centroid_x", y="cam_centroid_y", color="group",
        hover_data=["prob_defect", "cam_p90"], opacity=0.5,
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("동일 클래스끼리 cluster 가 형성되면 결함이 특정 위치에 편중되어 있다는 신호입니다.")

    st.subheader("힌트 (이전 단계에서 놓친 관찰)")
    summary_text = "\n".join([
        f"- 그룹 평균 cam_p90: " + ", ".join(f"{g}={stats.loc[stats.group==g, 'cam_p90'].mean():.3f}" for g in ['TP','TN','FN','FP'] if (stats.group==g).any()),
        f"- TP의 어텐션은 TN 대비 평균 약 {(stats.loc[stats.group=='TP','cam_p90'].mean() / max(stats.loc[stats.group=='TN','cam_p90'].mean(), 1e-9)):.2f}배 강함",
        "- FN 평균 cam_p90이 TP 평균보다 낮다면, 모델이 결함 영역을 약하게만 보고 있다는 신호 → 데이터 보강 또는 stronger augmentation 검토.",
        "- 어텐션 centroid가 이미지 중앙에 강하게 몰린다면, 결함 위치 분포의 spatial bias 가능성 — 운영 시 카메라/조명 표준화 필요.",
    ])
    st.markdown(summary_text)


# ----------- tab 3: misclassification deep -----------


def _show_with_cam(path: Path, model, device, caption_prefix: str = "") -> None:
    cols = st.columns([1, 1])
    resolved = _resolve_image_path(path)
    if resolved is None:
        with cols[0]:
            st.info(
                f"이미지 미포함 (배포 환경): `{Path(str(path)).name}`\n\n"
                "원본 파일은 `data/raw/` 가 있는 로컬 환경에서만 표시됩니다. "
                "데모용 컨센서스 오분류 케이스는 `reports/sample_images/` 에 미리 "
                "포함되어 있어 그쪽 항목은 정상적으로 보입니다."
            )
        return
    try:
        with Image.open(resolved) as im:
            img = im.convert("RGB")
        with cols[0]:
            st.image(
                img,
                caption=f"{caption_prefix}{resolved.parent.name}/{resolved.name}",
                use_container_width=True,
            )
        overlay, prob, pred = overlay_for_image(model, resolved, device, target_class=1)
        with cols[1]:
            st.image(overlay, caption=f"Grad-CAM | prob_def={prob:.3f}", use_container_width=True)
    except Exception as e:  # pragma: no cover
        with cols[0]:
            st.error(f"Failed for {resolved}: {e}")


def tab_misclassification() -> None:
    st.header("Tab 3 - 오분류 심화 분석")
    st.caption("최고 성능 모델의 오분류 케이스를 사람이 검토할 수 있도록 깊게 분해합니다.")

    frames = cached_frames()
    if not frames:
        st.warning("predictions 파일이 없습니다.")
        return

    best_tag = st.selectbox("최고 성능 모델 선택", list(CHECKPOINTS.keys()), index=1, key="mis_model")
    model_name, ckpt_name = CHECKPOINTS[best_tag]
    model = cached_model(model_name, ckpt_name)
    device = get_device_cached()
    if model is None:
        st.error("체크포인트가 없습니다.")
        return

    df = frames[best_tag]
    threshold = st.slider("Threshold", 0.05, 0.95, 0.5, 0.05, key="mis_thr")
    close_margin = st.slider("Close-margin 범위 (|prob - threshold| ≤ ?)", 0.05, 0.4, 0.2, 0.05, key="mis_close")

    cats = categorize_errors(df, threshold=threshold, close_margin=close_margin)
    c = st.columns(4)
    c[0].metric("FN close (아쉽게 놓침)", len(cats["FN_close"]))
    c[1].metric("FN large (확신하고 놓침)", len(cats["FN_large"]))
    c[2].metric("FP close", len(cats["FP_close"]))
    c[3].metric("FP large", len(cats["FP_large"]))

    st.subheader("Close-margin vs Large-margin FN 비교")
    if len(cats["FN_close"]) == 0 and len(cats["FN_large"]) == 0:
        st.success("FN 없음.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown(f"#### FN close (n={len(cats['FN_close'])}) - 임계값 가까이서 놓침")
            st.caption("→ 운영 임계값 조정만으로 잡힐 가능성. 추가 학습 신호 없이도 회수 가능.")
            for _, row in cats["FN_close"].head(5).iterrows():
                _show_with_cam(Path(row["path"]), model, device, caption_prefix="FN-close | ")
        with col_b:
            st.markdown(f"#### FN large (n={len(cats['FN_large'])}) - 크게 빗나감")
            st.caption("→ 모델이 자신 있게 OK로 판정. 라벨 오류 의심 또는 결함이 매우 미세함.")
            for _, row in cats["FN_large"].head(5).iterrows():
                _show_with_cam(Path(row["path"]), model, device, caption_prefix="FN-large | ")

    st.subheader("라벨 오류 의심 (전 모델 컨센서스)")
    consensus = consensus_misclassified(frames, threshold=threshold, min_models=max(2, len(frames) // 2 + 1))
    st.caption(f"≥{max(2, len(frames) // 2 + 1)}개 모델이 같은 케이스를 틀린 경우. 라벨 노이즈 후보.")
    if consensus.empty:
        st.info("컨센서스 오분류 없음 - 모델마다 다른 케이스를 놓침 (= 모델 분산 ↑).")
    else:
        st.dataframe(consensus[["path", "label", "miss_count", "mean_prob_defect", "models_missed"]], use_container_width=True)
        st.markdown("#### 컨센서스 오분류 시각 검토")
        for _, row in consensus.iterrows():
            with st.expander(f"miss={row['miss_count']}/{len(frames)} | mean_prob={row['mean_prob_defect']:.3f} | {Path(row['path']).name}"):
                _show_with_cam(Path(row["path"]), model, device, caption_prefix="CONSENSUS | ")
                st.markdown(
                    "**검토 가이드**\n"
                    "- 이미지를 직접 보고 결함이 실제로 존재하는가? (양품으로 보인다면 라벨 오류 의심)\n"
                    "- Grad-CAM이 무관한 영역(가장자리/배경)을 보고 있는가?\n"
                    "- 결함이 너무 미세해 사람도 놓칠 수준이면 → 카메라 해상도/조명 점검 필요."
                )

    st.subheader("힌트 / 향후 고도화 아이디어")
    n_close_fn_baseline = len(categorize_errors(frames.get("baseline", df), threshold=threshold, close_margin=close_margin)["FN_close"]) if "baseline" in frames else 0
    st.markdown(
        f"""
        - **Threshold tuning이 가장 ROI 높음**: 베이스라인 기준 close-margin FN이 {n_close_fn_baseline}건. 단순 임계값 조정으로 회수 가능.
        - **라벨 검수 우선**: 컨센서스 오분류 ({len(consensus)}건)는 어떤 모델도 잡지 못한 사례 — 데이터 품질 점검 후 재학습이 가장 효율적.
        - **앙상블**: ResNet18 + EfficientNet-B0 확률 평균은 각 모델이 놓치는 패턴이 다르면 도움. 컨센서스 오분류는 평균해도 그대로지만, 비컨센서스는 회수됨.
        - **TTA**: 거의 무비용이며 prob 안정화에 도움. 운영 권장.
        - **Active learning**: 가장 가까운 회색지대 (prob ≈ 0.5) 샘플을 우선 추가 라벨링 → 학습 효율 ↑.
        - **세분화 task (확장)**: 결함 위치를 박스/마스크로 라벨링하여 모델이 '어디'를 학습하면 FN 회수에 결정적.
        """
    )


# ----------- bonus: live -----------


def tab_live() -> None:
    st.header("Live Inference (bonus)")
    tag = st.selectbox("모델", list(CHECKPOINTS.keys()), index=1, key="live_tag")
    model_name, ckpt_name = CHECKPOINTS[tag]
    model = cached_model(model_name, ckpt_name)
    if model is None:
        st.error("Checkpoint not found.")
        return
    device = get_device_cached()

    up = st.file_uploader("Casting 이미지 업로드 (jpg/png/bmp)", type=["jpg", "jpeg", "png", "bmp"])
    if not up:
        st.info("이미지를 업로드하면 예측 + Grad-CAM 을 표시합니다.")
        return

    suffix = Path(up.name).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        tmp_file.write(up.getbuffer())
        tmp_path = Path(tmp_file.name)
    try:
        overlay, prob, pred = overlay_for_image(model, tmp_path, device, target_class=1)
        c1, c2 = st.columns(2)
        with c1:
            st.image(Image.open(tmp_path).convert("RGB"), caption="입력", use_container_width=True)
        with c2:
            st.image(overlay, caption="Grad-CAM (defect)", use_container_width=True)
        st.markdown(
            f"### 예측: **{'def_front (DEFECT)' if pred == 1 else 'ok_front (OK)'}**  |  "
            f"prob(defect) = `{prob:.4f}`"
        )
        st.progress(min(max(prob, 0.0), 1.0))
    finally:
        tmp_path.unlink(missing_ok=True)


# ----------- main -----------


def main() -> None:
    st.sidebar.title("Casting Defect Dashboard v2")
    page = st.sidebar.radio(
        "Pages",
        ["1. 실험 결과", "2. 변수 분석 (Grad-CAM)", "3. 오분류 심화", "Live Inference"],
        index=0,
    )
    st.sidebar.markdown("---")
    st.sidebar.caption(f"device: {get_device_cached()}")
    st.sidebar.caption(f"runs in reports/: {len(cached_frames())}")

    if page.startswith("1"):
        tab_experiments()
    elif page.startswith("2"):
        tab_cam_analysis()
    elif page.startswith("3"):
        tab_misclassification()
    else:
        tab_live()


if __name__ == "__main__":
    main()
