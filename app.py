"""Streamlit entry point for RouteGuard."""

import base64
import hashlib
import os
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from src.config import AnalysisConfig
from src.detector import ObjectDetector
from src.reporting import build_recommendations, compare_results, event_rows, render_text_report, summarize_events
from src.video_analyzer import ANALYZER_VERSION, analyze_video


os.environ.setdefault("YOLO_CONFIG_DIR", str(Path("Ultralytics").resolve()))


SAMPLE_DIR = Path("data/samples")
UPLOAD_DIR = Path("outputs/uploads")
RESULT_DIR = Path("outputs/results")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)


def find_background_image() -> Path | None:
    """Return an optional local background image from assets."""
    for name in ["background.jpg", "background.jpeg", "background.png", "background.webp"]:
        path = Path("assets") / name
        if path.exists():
            return path
    return None


def background_css() -> str:
    """Build CSS background using a local image when available."""
    background = find_background_image()
    if not background:
        return """
        background:
            radial-gradient(circle at 20% 15%, rgba(34, 197, 94, 0.22), transparent 28rem),
            radial-gradient(circle at 85% 10%, rgba(59, 130, 246, 0.20), transparent 24rem),
            linear-gradient(135deg, #0F172A 0%, #111827 42%, #172554 100%);
        """

    mime = "image/png" if background.suffix.lower() == ".png" else "image/jpeg"
    if background.suffix.lower() == ".webp":
        mime = "image/webp"
    encoded = base64.b64encode(background.read_bytes()).decode("utf-8")
    return f"""
        background:
            linear-gradient(135deg, rgba(15, 23, 42, 0.68), rgba(17, 24, 39, 0.62)),
            url("data:{mime};base64,{encoded}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    """


def inject_theme() -> None:
    """Apply RouteGuard visual theme."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            {background_css()}
            color: #F8FAFC;
        }}

        [data-testid="stHeader"] {{
            background: transparent;
        }}

        [data-testid="stSidebar"] {{
            background: rgba(2, 6, 23, 0.94);
            border-right: 1px solid rgba(148, 163, 184, 0.24);
            backdrop-filter: blur(18px);
        }}

        .block-container {{
            padding-top: 2.0rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }}

        .rg-hero {{
            padding: 2.1rem 2.3rem;
            border: 1px solid rgba(148, 163, 184, 0.24);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(2, 6, 23, 0.88), rgba(15, 23, 42, 0.78)),
                linear-gradient(135deg, rgba(34, 197, 94, 0.18), rgba(59, 130, 246, 0.12));
            box-shadow: 0 24px 80px rgba(0, 0, 0, 0.34);
            backdrop-filter: blur(14px);
            margin-bottom: 1.25rem;
        }}

        .rg-eyebrow {{
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            padding: 0.35rem 0.72rem;
            border-radius: 999px;
            color: #BBF7D0;
            background: rgba(22, 101, 52, 0.36);
            border: 1px solid rgba(134, 239, 172, 0.28);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
        }}

        .rg-title {{
            margin: 0.78rem 0 0.2rem;
            font-size: clamp(2.6rem, 6vw, 5.4rem);
            line-height: 0.95;
            font-weight: 900;
            letter-spacing: -0.07em;
            color: #F8FAFC;
        }}

        .rg-subtitle {{
            max-width: 760px;
            color: #CBD5E1;
            font-size: 1.08rem;
            line-height: 1.7;
            margin-top: 0.85rem;
        }}

        .rg-badges {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1.15rem;
        }}

        .rg-badge {{
            padding: 0.48rem 0.75rem;
            border-radius: 999px;
            color: #E2E8F0;
            background: rgba(15, 23, 42, 0.64);
            border: 1px solid rgba(148, 163, 184, 0.22);
            font-size: 0.9rem;
        }}

        .rg-panel {{
            padding: 1.25rem 1.35rem;
            border: 1px solid rgba(148, 163, 184, 0.22);
            border-radius: 22px;
            background: rgba(2, 6, 23, 0.84);
            box-shadow: 0 18px 55px rgba(0, 0, 0, 0.20);
            backdrop-filter: blur(16px);
            margin: 0.8rem 0 1rem;
        }}

        .rg-section-title {{
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin-bottom: 0.75rem;
        }}

        .rg-section-title span {{
            display: inline-flex;
            width: 2rem;
            height: 2rem;
            border-radius: 12px;
            align-items: center;
            justify-content: center;
            background: rgba(34, 197, 94, 0.22);
            border: 1px solid rgba(134, 239, 172, 0.28);
            font-weight: 900;
        }}

        .rg-note-card {{
            padding: 0.9rem 1rem;
            border-radius: 18px;
            background: rgba(15, 23, 42, 0.74);
            border: 1px solid rgba(148, 163, 184, 0.20);
            margin: 0.55rem 0;
        }}

        .rg-side-title {{
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: -0.04em;
            color: #F8FAFC;
            margin-bottom: 0.25rem;
        }}

        .rg-side-copy {{
            color: #CBD5E1;
            line-height: 1.65;
            font-size: 0.94rem;
        }}

        .rg-side-step {{
            padding: 0.72rem 0.78rem;
            border-radius: 15px;
            background: rgba(15, 23, 42, 0.76);
            border: 1px solid rgba(148, 163, 184, 0.18);
            margin: 0.45rem 0;
        }}

        .rg-panel h3 {{
            margin-top: 0;
        }}

        div[data-testid="stMetric"] {{
            background: rgba(15, 23, 42, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 12px 32px rgba(0, 0, 0, 0.18);
        }}

        div[data-testid="stMetric"] label {{
            color: #E2E8F0 !important;
            font-weight: 700 !important;
        }}

        div[data-testid="stMetricValue"] {{
            color: #FFFFFF !important;
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap: 0.45rem;
        }}

        .stTabs [data-baseweb="tab"] {{
            border-radius: 999px;
            padding: 0.55rem 1rem;
            background: rgba(2, 6, 23, 0.90);
            border: 1px solid rgba(148, 163, 184, 0.18);
            color: #F8FAFC;
        }}

        .stButton > button, .stDownloadButton > button {{
            border-radius: 999px;
            border: 1px solid rgba(134, 239, 172, 0.32);
            background: linear-gradient(135deg, #16A34A, #2563EB);
            color: white;
            font-weight: 800;
            box-shadow: 0 12px 28px rgba(37, 99, 235, 0.28);
        }}

        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: rgba(187, 247, 208, 0.72);
            transform: translateY(-1px);
        }}

        div[data-testid="stDataFrame"], div[data-testid="stExpander"], div[data-testid="stAlert"] {{
            border-radius: 18px;
        }}

        h1, h2, h3 {{
            color: #F8FAFC;
        }}

        p, li, label, span, .stMarkdown, [data-testid="stMarkdownContainer"] {{
            color: #F1F5F9 !important;
        }}

        div[data-testid="stCaptionContainer"], .stCaption {{
            color: #CBD5E1 !important;
        }}

        .stRadio label, .stSelectbox label, .stFileUploader label, .stSlider label, .stCheckbox label {{
            color: #F8FAFC !important;
            font-weight: 700;
        }}

        input, textarea, div[data-baseweb="select"] {{
            color: #0F172A !important;
        }}

        [data-testid="stFileUploader"] {{
            background: rgba(248, 250, 252, 0.96);
            border: 1px solid rgba(203, 213, 225, 0.95);
            border-radius: 14px;
            padding: 0.6rem 0.75rem;
        }}

        [data-testid="stFileUploader"] * {{
            color: #0F172A !important;
        }}

        [data-testid="stFileUploader"] button {{
            background: #FFFFFF !important;
            color: #0F172A !important;
            border: 1px solid #CBD5E1 !important;
            box-shadow: none !important;
        }}

        [data-testid="stFileUploader"] small {{
            color: #475569 !important;
        }}

        [data-testid="stRadio"] label span {{
            color: #F8FAFC !important;
            font-weight: 800;
        }}

        div[data-testid="stExpander"] {{
            background: rgba(2, 6, 23, 0.80);
            border: 1px solid rgba(148, 163, 184, 0.22);
            backdrop-filter: blur(12px);
        }}

        div[data-testid="stAlert"] {{
            background: rgba(15, 23, 42, 0.86);
            border: 1px solid rgba(148, 163, 184, 0.22);
        }}

        .stVideo video {{
            max-height: 420px;
            border-radius: 18px;
            object-fit: contain;
            background: rgba(2, 6, 23, 0.78);
        }}

        [data-testid="stImage"] img {{
            max-height: 520px;
            object-fit: contain;
            border-radius: 18px;
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: rgba(2, 6, 23, 0.72);
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_detector():
    """Load YOLO once so repeated analyses are faster."""
    return ObjectDetector(AnalysisConfig())


def build_config() -> AnalysisConfig:
    """Return the fixed runtime settings used by the demo app."""
    return AnalysisConfig()


def save_uploaded_file(uploaded_file) -> Path:
    """Persist an uploaded video under outputs/uploads."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    data = uploaded_file.getbuffer()
    digest = hashlib.sha1(data).hexdigest()[:12]
    safe_stem = "".join(char if char.isalnum() else "_" for char in Path(uploaded_file.name).stem)[:40]
    input_path = UPLOAD_DIR / f"{safe_stem}_{digest}{suffix}"
    if not input_path.exists():
        input_path.write_bytes(data)
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
    output_path = RESULT_DIR / f"{input_path.stem}_{tag}_{uuid4().hex[:8]}_analyzed.mp4"
    result = analyze_video(input_path, output_path, config=config, detector=detector)
    result["input_signature"] = input_signature(input_path)
    return result


def input_signature(input_path: Path) -> str:
    """Return a stable signature so stale Streamlit results can be discarded."""
    stat = input_path.stat()
    return f"{input_path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"


def is_fresh_result(result: dict | None, input_path: Path) -> bool:
    """Check whether the displayed result belongs to the current analyzer and input."""
    if not result:
        return False
    return (
        result.get("analyzer_version") == ANALYZER_VERSION
        and result.get("input_signature") == input_signature(input_path)
    )


def render_score_card(result: dict) -> None:
    """Render top-level score and performance metrics."""
    summary = summarize_events(result["events"])
    st.subheader("점검 결과")
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
        st.subheader("시간대별 발견 내용")
        if events:
            st.dataframe(pd.DataFrame(event_rows(events)), use_container_width=True, hide_index=True)
        else:
            st.success("중앙 통로를 크게 막는 물체가 발견되지 않았습니다.")

    with right:
        st.subheader("바로 할 일")
        for recommendation in build_recommendations(events, result["score"]):
            st.write(f"- {recommendation}")


def top_detection_rows(result: dict) -> list[dict]:
    """Return the most frequently detected objects in a readable table."""
    counts = result.get("detection_counts", {})
    return [
        {"물체": label, "확인된 장면 수": count}
        for label, count in sorted(counts.items(), key=lambda item: item[1], reverse=True)[:6]
    ]


def decision_rows(result: dict) -> list[dict]:
    """Explain why RouteGuard produced the score without exposing raw settings."""
    events = result.get("events", [])
    rows = [
        {
            "판단 항목": "통로를 막은 정도",
            "결과": f"가장 큰 겹침 {result.get('max_path_overlap', 0.0):.0%}",
            "의미": "중앙 이동 경로와 물체 박스가 많이 겹칠수록 위험하게 봅니다.",
        },
        {
            "판단 항목": "반복 감지",
            "결과": f"위험 장면 {result.get('risk_sampled_frames', 0)}개 / 확인 장면 {result.get('sampled_frames', 0)}개",
            "의미": "한순간만 지나간 물체보다 여러 장면에서 계속 보인 장애물을 더 중요하게 봅니다.",
        },
        {
            "판단 항목": "위험 후보 수",
            "결과": f"{len(events)}개 이벤트",
            "의미": "서로 다른 장애물이 동시에 있으면 이동 공간이 더 좁다고 판단합니다.",
        },
    ]

    total_penalty = sum(int(event.get("penalty", 0)) for event in events)
    rows.append(
        {
            "판단 항목": "감점 합계",
            "결과": f"-{total_penalty}점" if total_penalty else "0점",
            "의미": "의자처럼 큰 물체, 중앙을 크게 막은 물체, 반복적으로 보인 물체에 더 큰 감점을 줍니다.",
        }
    )
    return rows


def render_result_media(result: dict) -> None:
    """Render a reliable visual result, with video as a secondary view."""
    st.subheader("분석 결과 미리보기")
    if not result.get("events"):
        st.success("감점 이벤트가 없습니다. 중앙 통로를 크게 막는 물체가 발견되지 않았습니다.")
        return

    gif_value = result.get("gif_path") or ""
    gif_path = Path(gif_value) if gif_value else None
    preview_value = result.get("preview_path") or ""
    preview_path = Path(preview_value) if preview_value else None

    if preview_path and preview_path.exists():
        st.image(
            str(preview_path),
            caption=f"가장 큰 감점 이벤트가 표시된 장면: {result.get('preview_timestamp', 0.0):.1f}초",
            use_container_width=True,
        )
    elif gif_path and gif_path.exists():
        st.image(str(gif_path), caption="방해물 박스가 표시된 분석 장면 요약", use_container_width=True)
    else:
        st.info("대표 장면 이미지를 만들지 못했습니다.")

    if gif_path and gif_path.exists():
        with st.expander("전체 분석 흐름 GIF 보기", expanded=False):
            st.image(str(gif_path), caption="프레임별 방해물 박스 요약", use_container_width=True)


def render_source_preview(input_path: Path) -> None:
    """Keep uploaded video preview compact so it does not dominate the page."""
    st.caption(f"선택된 영상: {input_path.name}")
    with st.expander("원본 영상 미리보기", expanded=False):
        st.video(str(input_path))


def render_diagnostics(result: dict) -> None:
    """Render user-friendly analysis evidence."""
    with st.expander("자세한 분석 근거", expanded=True):
        st.markdown(
            "RouteGuard는 영상의 중앙 하단을 이동 통로로 보고, 물체가 그 영역을 얼마나 오래 그리고 얼마나 크게 막는지 확인합니다."
        )
        cols = st.columns(4)
        cols[0].metric("분석 길이", f"{result.get('duration_seconds', 0):.1f}초")
        cols[1].metric("확인한 장면", result["sampled_frames"])
        cols[2].metric("위험 장면 비율", f"{result['risk_frame_ratio']:.0%}")
        cols[3].metric("처리 시간", f"{result['elapsed_seconds']:.1f}초")

        st.write("점수 판단 요약")
        st.dataframe(pd.DataFrame(decision_rows(result)), use_container_width=True, hide_index=True)

        st.write("영상에서 자주 확인된 물체")
        rows = top_detection_rows(result)
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("반복적으로 확인된 물체가 없습니다.")


def render_downloads(result: dict) -> None:
    """Render result download buttons."""
    output_path = Path(result["output_path"])
    gif_value = result.get("gif_path") or ""
    gif_path = Path(gif_value) if gif_value else None
    col_video, col_gif, col_report = st.columns(3)
    if output_path.exists():
        col_video.download_button(
            "분석 결과 영상 다운로드",
            data=output_path.read_bytes(),
            file_name=output_path.name,
            mime="video/mp4",
        )
    if gif_path and gif_path.exists():
        col_gif.download_button(
            "분석 GIF 다운로드",
            data=gif_path.read_bytes(),
            file_name=gif_path.name,
            mime="image/gif",
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
        st.info("점검할 실내 공간 영상을 선택해 주세요.")
        return

    render_source_preview(input_path)
    if st.button("분석 시작", type="primary"):
        with st.spinner("영상을 분석하고 안전 점수를 계산하는 중입니다..."):
            result = analyze_path(input_path, config, detector, "single")
        st.session_state["last_result"] = result

    result = st.session_state.get("last_result")
    if result and not is_fresh_result(result, input_path):
        st.session_state.pop("last_result", None)
        result = None
        st.info("분석 기준이 업데이트되었습니다. 현재 영상으로 다시 분석을 시작해 주세요.")
    if not result:
        return

    st.divider()
    render_score_card(result)
    render_result_media(result)
    render_event_section(result)
    render_diagnostics(result)
    render_downloads(result)


def render_compare_analysis(config: AnalysisConfig, detector: ObjectDetector) -> None:
    """Render before/after comparison workflow."""
    st.subheader("정리 전후 비교")
    before = st.file_uploader("정리 전 영상", type=["mp4", "mov", "avi"], key="before_video")
    after = st.file_uploader("정리 후 영상", type=["mp4", "mov", "avi"], key="after_video")

    if not before or not after:
        st.info("같은 위치에서 촬영한 정리 전 영상과 정리 후 영상을 넣으면 변화가 한눈에 보입니다.")
        return

    before_path = save_uploaded_file(before)
    after_path = save_uploaded_file(after)
    cols = st.columns(2)
    with cols[0]:
        render_source_preview(before_path)
    with cols[1]:
        render_source_preview(after_path)

    if st.button("전후 비교 분석", type="primary"):
        with st.spinner("정리 전후 영상을 같은 기준으로 비교하는 중입니다..."):
            before_result = analyze_path(before_path, config, detector, "before")
            after_result = analyze_path(after_path, config, detector, "after")
        st.session_state["compare_result"] = (before_result, after_result, compare_results(before_result, after_result))

    compare_state = st.session_state.get("compare_result")
    if compare_state:
        before_result, after_result, _comparison = compare_state
        if not is_fresh_result(before_result, before_path) or not is_fresh_result(after_result, after_path):
            st.session_state.pop("compare_result", None)
            compare_state = None
            st.info("비교 분석 기준이 업데이트되었습니다. 두 영상을 다시 분석해 주세요.")
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
        render_result_media(before_result)
        render_event_section(before_result)
    with cols[1]:
        st.subheader("정리 후 결과")
        render_result_media(after_result)
        render_event_section(after_result)


st.set_page_config(page_title="RouteGuard", layout="wide")
inject_theme()

st.markdown(
    """
    <section class="rg-hero">
      <div class="rg-eyebrow">Indoor Route Safety Vision</div>
      <div class="rg-title">RouteGuard</div>
      <div class="rg-subtitle">
        스마트폰으로 촬영한 실내 영상을 분석해 중앙 이동 경로를 막는 물체를 찾고,
        통로 겹침 정도와 반복 감지를 이용해 안전 점수와 시간대별 피드백을 제공합니다.
      </div>
      <div class="rg-badges">
        <span class="rg-badge">출입구 동선 점검</span>
        <span class="rg-badge">안전 점수</span>
        <span class="rg-badge">시간대별 피드백</span>
        <span class="rg-badge">정리 전후 비교</span>
      </div>
    </section>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="rg-side-title">RouteGuard</div>
    <div class="rg-side-copy">
      방뿐만 아니라 연구실, 동아리방, 소형 사무실처럼 물건이 많은 실내 공간에서도 사용할 수 있는 안전 점검 도구입니다.
    </div>
    <div class="rg-side-step"><b>1. 영상 선택</b><br/>점검할 실내 공간 영상을 올립니다.</div>
    <div class="rg-side-step"><b>2. 자동 점검</b><br/>중앙 통로를 막는 물체와 반복되는 위험 구간을 찾습니다.</div>
    <div class="rg-side-step"><b>3. 결과 확인</b><br/>안전 점수, 발견 시간, 표시된 결과 영상을 확인합니다.</div>
    """,
    unsafe_allow_html=True,
)

st.sidebar.markdown(
    """
    <div class="rg-note-card">
      <b>점수 기준</b><br/>
      80점 이상: 안전<br/>
      50~79점: 주의<br/>
      50점 미만: 위험
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("어떻게 판단하나요?"):
    st.write("화면 아래쪽에서 출입구 방향으로 이어지는 중앙 통로를 기준으로 봅니다.")
    st.write("물체가 그 통로와 많이 겹치고 여러 순간 반복해서 보이면 점수가 낮아집니다.")
    st.write("정확한 물체 이름보다 실제로 지나가는 길을 막는지가 더 중요합니다.")

runtime_config = build_config()
runtime_detector = load_detector()

tab_single, tab_compare, tab_about = st.tabs(["내 통로 점검", "정리 전후 비교", "도움말"])

with tab_single:
    render_single_analysis(runtime_config, runtime_detector)

with tab_compare:
    render_compare_analysis(runtime_config, runtime_detector)

with tab_about:
    st.subheader("언제 쓰면 좋나요?")
    st.write("- 방, 연구실, 동아리방, 소형 사무실처럼 물건이 많은 실내 공간을 점검하고 싶을 때")
    st.write("- 기숙사나 원룸처럼 통로가 좁은 공간에서 큰 물건이 길을 막는지 확인할 때")
    st.write("- 정리 전후 영상을 비교해 실제로 동선이 좋아졌는지 보고 싶을 때")
    st.write("- 안전 점수와 결과 영상을 남겨 간단한 점검 기록으로 보관하고 싶을 때")
