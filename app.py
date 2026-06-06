"""Streamlit entry point for RouteGuard."""

import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.config import AnalysisConfig
from src.detector import DEFAULT_OBSTACLE_CLASSES, ObjectDetector
from src.reporting import build_recommendations, compare_results, event_rows, render_text_report, summarize_events
from src.video_analyzer import analyze_video


os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("Ultralytics").resolve()))


SAMPLE_DIR = Path("data/samples")
UPLOAD_DIR = Path("outputs/uploads")
RESULT_DIR = Path("outputs/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


@st.cache_resource
def load_detector(model_name: str, confidence_threshold: float, image_size: int):
    """Load YOLO once per model setting so repeated analyses are faster."""
    config = AnalysisConfig(
        model_name=model_name,
        confidence_threshold=confidence_threshold,
        inference_image_size=image_size,
    )
    return ObjectDetector(config)


def build_config() -> AnalysisConfig:
    """Build runtime settings from the sidebar controls."""
    st.sidebar.header("분석 설정")
    confidence = st.sidebar.slider("탐지 confidence", 0.20, 0.70, 0.35, 0.05)
    path_width = st.sidebar.slider("중앙 통로 폭", 0.30, 0.60, 0.42, 0.02)
    sample_interval = st.sidebar.slider("프레임 샘플 간격", 3, 12, 5, 1)
    min_observations = st.sidebar.slider("위험 반복 최소 횟수", 1, 5, 2, 1)
    image_size = st.sidebar.select_slider("YOLO 입력 크기", options=[416, 512, 640, 768], value=640)
    max_seconds = st.sidebar.slider("최대 분석 길이(초)", 5, 30, 20, 5)
    draw_safe = st.sidebar.checkbox("안전 후보 박스도 표시", value=False)

    return AnalysisConfig(
        confidence_threshold=confidence,
        inference_image_size=image_size,
        path_width_ratio=path_width,
        frame_sample_interval=sample_interval,
        min_event_observations=min_observations,
        max_video_seconds=max_seconds,
        draw_safe_detections=draw_safe,
    )


def save_uploaded_file(uploaded_file) -> Path:
    """Persist an uploaded video under outputs/uploads."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    input_path = UPLOAD_DIR / f"{uuid4().hex}{suffix}"
    input_path.write_bytes(uploaded_file.getbuffer())
    return input_path


def sample_options() -> dict[str, Path]:
    """Return bundled sample videos keyed by readable labels."""
    videos = {}
    for path in sorted(SAMPLE_DIR.glob("**/*")):
        if path.suffix.lower() in {".mp4", ".mov", ".avi"}:
            videos[f"{path.parent.name}/{path.name}"] = path
    return videos


def analyze_path(input_path: Path, config: AnalysisConfig, detector: ObjectDetector, tag: str) -> dict:
    """Run RouteGuard analysis for a video path."""
    output_path = RESULT_DIR / f"{input_path.stem}_{tag}_analyzed.mp4"
    return analyze_video(input_path, output_path, config=config, detector=detector)


def render_score_card(result: dict) -> None:
    """Render top-level score and performance metrics."""
    summary = summarize_events(result["events"])
    cols = st.columns(5)
    cols[0].metric("안전 점수", f"{result['score']} / 100")
    cols[1].metric("위험 수준", result["risk_level"])
    cols[2].metric("위험 이벤트", summary["event_count"])
    cols[3].metric("최대 통로 겹침", f"{summary['max_overlap']:.0%}")
    cols[4].metric("처리 속도", f"{result['processing_fps']:.1f} FPS")


def render_event_section(result: dict) -> None:
    """Render event table and recommendations."""
    events = result["events"]
    left, right = st.columns([2, 1])

    with left:
        st.subheader("시간대별 위험 요소")
        if events:
            st.dataframe(pd.DataFrame(event_rows(events)), use_container_width=True, hide_index=True)
        else:
            st.success("통로 영역과 크게 겹치는 위험 후보가 발견되지 않았습니다.")

    with right:
        st.subheader("개선 피드백")
        for recommendation in build_recommendations(events, result["score"]):
            st.write(f"- {recommendation}")


def render_diagnostics(result: dict) -> None:
    """Render detection and runtime diagnostics."""
    with st.expander("분석 근거와 성능 지표"):
        cols = st.columns(4)
        cols[0].metric("분석 프레임", result["frames"])
        cols[1].metric("샘플 프레임", result["sampled_frames"])
        cols[2].metric("위험 샘플 비율", f"{result['risk_frame_ratio']:.0%}")
        cols[3].metric("처리 시간", f"{result['elapsed_seconds']:.1f}s")

        st.write("객체 후보 카운트")
        counts = result.get("detection_counts", {})
        if counts:
            count_rows = [{"객체": label, "탐지 횟수": count} for label, count in sorted(counts.items())]
            st.dataframe(pd.DataFrame(count_rows), use_container_width=True, hide_index=True)
        else:
            st.caption("탐지된 후보 객체가 없습니다.")

        st.write("사용 설정")
        st.json(result["config"], expanded=False)


def render_downloads(result: dict) -> None:
    """Render result download buttons."""
    output_path = Path(result["output_path"])
    col_video, col_report = st.columns(2)
    if output_path.exists():
        col_video.download_button(
            "분석 결과 영상 다운로드",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="video/mp4",
        )
    col_report.download_button(
        "분석 리포트 다운로드",
        data=render_text_report(result).encode("utf-8"),
        file_name=f"{output_path.stem}_report.txt",
        mime="text/plain",
    )


def render_single_analysis(config: AnalysisConfig, detector: ObjectDetector) -> None:
    """Render single-video analysis workflow."""
    st.subheader("영상 선택")
    source = st.radio("입력 방식", ["파일 업로드", "샘플 영상"], horizontal=True)
    input_path = None

    if source == "파일 업로드":
        uploaded = st.file_uploader("대피 경로 영상을 업로드하세요", type=["mp4", "mov", "avi"])
        if uploaded:
            input_path = save_uploaded_file(uploaded)
    else:
        samples = sample_options()
        if not samples:
            st.warning("data/samples 폴더에 샘플 영상이 없습니다.")
        else:
            label = st.selectbox("샘플 선택", list(samples))
            input_path = samples[label]

    if input_path is None:
        st.info("방에서 출입구 방향으로 촬영한 5~15초 영상을 선택하면 분석을 시작할 수 있습니다.")
        return

    st.video(str(input_path))
    if st.button("분석 시작", type="primary"):
        with st.spinner("YOLO 객체 탐지와 통로 겹침 분석을 실행하는 중입니다..."):
            result = analyze_path(input_path, config, detector, "single")
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if not result:
        return

    st.divider()
    render_score_card(result)
    st.subheader("분석 결과 영상")
    st.video(result["output_path"])
    render_event_section(result)
    render_diagnostics(result)
    render_downloads(result)


def render_compare_analysis(config: AnalysisConfig, detector: ObjectDetector) -> None:
    """Render before/after comparison workflow."""
    st.subheader("정리 전후 비교")
    before = st.file_uploader("정리 전 영상", type=["mp4", "mov", "avi"], key="before_video")
    after = st.file_uploader("정리 후 영상", type=["mp4", "mov", "avi"], key="after_video")

    if not before or not after:
        st.info("같은 위치에서 촬영한 정리 전/후 영상을 넣으면 점수 변화를 비교합니다.")
        return

    before_path = save_uploaded_file(before)
    after_path = save_uploaded_file(after)
    cols = st.columns(2)
    cols[0].video(str(before_path))
    cols[1].video(str(after_path))

    if st.button("전후 비교 분석", type="primary"):
        with st.spinner("두 영상을 같은 기준으로 분석하는 중입니다..."):
            before_result = analyze_path(before_path, config, detector, "before")
            after_result = analyze_path(after_path, config, detector, "after")
        st.session_state["compare_result"] = (before_result, after_result, compare_results(before_result, after_result))

    compare_state = st.session_state.get("compare_result")
    if not compare_state:
        return

    before_result, after_result, comparison = compare_state
    st.divider()
    cols = st.columns(4)
    cols[0].metric("정리 전 점수", comparison["before_score"])
    cols[1].metric("정리 후 점수", comparison["after_score"])
    cols[2].metric("점수 변화", comparison["score_delta"])
    cols[3].metric("위험 이벤트 변화", comparison["event_delta"])
    st.success(comparison["summary"])

    cols = st.columns(2)
    with cols[0]:
        st.subheader("정리 전 결과")
        st.video(before_result["output_path"])
        render_event_section(before_result)
    with cols[1]:
        st.subheader("정리 후 결과")
        st.video(after_result["output_path"])
        render_event_section(after_result)


st.set_page_config(page_title="RouteGuard", layout="wide")
st.title("RouteGuard")
st.caption("영상 기반 실내 비상 대피 경로 위험 요소 탐지 시스템")

st.markdown(
    """
    RouteGuard는 스마트폰으로 촬영한 짧은 실내 영상을 분석해 중앙 이동 경로를 막는 객체 후보를 찾습니다.
    단순히 물체를 찾는 데서 끝나지 않고, 통로 겹침 정도와 반복 감지를 이용해 안전 점수와 시간대별 피드백을 제공합니다.
    """
)

with st.expander("탐지 가능한 위험 후보와 한계"):
    st.write(", ".join(sorted(DEFAULT_OBSTACLE_CLASSES)))
    st.write(
        "전선이나 멀티탭처럼 작고 가는 물체는 기본 YOLO 모델만으로 안정적으로 탐지하기 어렵습니다. "
        "이 프로젝트는 법적 안전 인증이 아니라 대피 경로 점검을 돕는 보조 도구입니다."
    )

runtime_config = build_config()
runtime_detector = load_detector(
    runtime_config.model_name,
    runtime_config.confidence_threshold,
    runtime_config.inference_image_size,
)

tab_single, tab_compare, tab_about = st.tabs(["단일 영상 분석", "정리 전후 비교", "프로젝트 설명"])

with tab_single:
    render_single_analysis(runtime_config, runtime_detector)

with tab_compare:
    render_compare_analysis(runtime_config, runtime_detector)

with tab_about:
    st.subheader("왜 직접 보는 것보다 RouteGuard가 필요한가요?")
    st.write("- 익숙한 방에서는 위험 물체를 과소평가하기 쉽습니다.")
    st.write("- RouteGuard는 어느 시점에 어떤 후보가 통로를 막았는지 기록합니다.")
    st.write("- 감으로 판단하지 않고 통로 영역과 객체 박스의 겹침 정도를 점수화합니다.")
    st.write("- 정리 전후 영상을 같은 기준으로 비교할 수 있어 개선 효과를 README에 보여주기 좋습니다.")
