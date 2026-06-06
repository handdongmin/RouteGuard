"""Tests for RouteGuard reporting helpers."""

from src.reporting import build_recommendations, compare_results, event_rows, render_text_report, summarize_events


def test_event_rows_and_summary():
    events = [
        {
            "timestamp": "00:03",
            "object_name": "가방",
            "severity": "danger",
            "overlap_ratio": 0.64,
            "observations": 3,
            "penalty": 25,
            "message": "가방 후보가 이동 경로 중앙을 크게 막고 있습니다.",
        }
    ]

    rows = event_rows(events)
    summary = summarize_events(events)

    assert rows[0]["객체"] == "가방"
    assert rows[0]["통로 겹침"] == "64%"
    assert summary["event_count"] == 1
    assert summary["danger_count"] == 1
    assert summary["total_penalty"] == 25


def test_recommendations_and_comparison():
    before = {"score": 55, "events": [{"object_name": "의자"}]}
    after = {"score": 90, "events": []}

    comparison = compare_results(before, after)
    recommendations = build_recommendations(before["events"], before["score"])

    assert comparison["score_delta"] == 35
    assert "상승" in comparison["summary"]
    assert any("의자" in item for item in recommendations)


def test_render_text_report():
    result = {
        "score": 80,
        "risk_level": "안전",
        "duration_seconds": 6.0,
        "frames": 180,
        "sampled_frames": 36,
        "risk_frame_ratio": 0.0,
        "processing_fps": 42.5,
        "events": [],
    }

    report = render_text_report(result)

    assert "RouteGuard Analysis Report" in report
    assert "Safety score: 80 / 100" in report
    assert "Recommendations" in report
