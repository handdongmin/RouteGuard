"""Analysis report helpers for RouteGuard."""

from collections import Counter


def event_rows(events: list[dict]) -> list[dict]:
    """Convert risk events into table-friendly rows."""
    rows = []
    for event in events:
        rows.append(
            {
                "시간": event.get("timestamp", "00:00"),
                "객체": event.get("object_name", event.get("label", "object")),
                "위험도": "위험" if event.get("severity") == "danger" else "주의",
                "통로 겹침": f"{event.get('overlap_ratio', 0.0):.0%}",
                "반복": int(event.get("observations", 1)),
                "감점": int(event.get("penalty", 0)),
                "피드백": event.get("message", ""),
            }
        )
    return rows


def summarize_events(events: list[dict]) -> dict:
    """Return compact event summary metrics."""
    labels = Counter(event.get("object_name", event.get("label", "object")) for event in events)
    severity = Counter(event.get("severity", "safe") for event in events)
    total_penalty = sum(int(event.get("penalty", 0)) for event in events)
    max_overlap = max((float(event.get("overlap_ratio", 0.0)) for event in events), default=0.0)
    return {
        "event_count": len(events),
        "danger_count": severity.get("danger", 0),
        "caution_count": severity.get("caution", 0),
        "total_penalty": total_penalty,
        "max_overlap": max_overlap,
        "top_objects": labels.most_common(3),
    }


def build_recommendations(events: list[dict], score: int) -> list[str]:
    """Generate practical cleanup recommendations from risk events."""
    if not events:
        return [
            "중앙 이동 경로가 비교적 확보되어 있습니다.",
            "출입구 앞 1m 정도는 계속 비워두면 더 안전합니다.",
        ]

    object_names = [event.get("object_name", "물체") for event in events]
    unique_names = list(dict.fromkeys(object_names))
    recommendations = [
        f"{', '.join(unique_names[:3])} 후보를 벽면이나 수납 공간으로 이동하세요.",
        "결과 영상에서 빨간색으로 표시된 구간을 먼저 정리하세요.",
        "출입구 주변과 화면 중앙 하단의 바닥 통로를 우선 확보하세요.",
    ]
    if score < 50:
        recommendations.append("위험 후보가 여러 번 반복되어 감지됐으므로 정리 후 재촬영을 권장합니다.")
    else:
        recommendations.append("주의 단계이므로 큰 장애물부터 치우면 점수가 빠르게 개선될 수 있습니다.")
    return recommendations


def compare_results(before: dict, after: dict) -> dict:
    """Compare two analysis results for before/after cleanup demos."""
    score_delta = int(after["score"]) - int(before["score"])
    event_delta = len(after.get("events", [])) - len(before.get("events", []))
    if score_delta > 0:
        summary = f"정리 후 안전 점수가 {score_delta}점 상승했습니다."
    elif score_delta < 0:
        summary = f"정리 후 안전 점수가 {abs(score_delta)}점 하락했습니다."
    else:
        summary = "정리 전후 안전 점수가 동일합니다."

    return {
        "before_score": before["score"],
        "after_score": after["score"],
        "score_delta": score_delta,
        "before_events": len(before.get("events", [])),
        "after_events": len(after.get("events", [])),
        "event_delta": event_delta,
        "summary": summary,
    }


def render_text_report(result: dict) -> str:
    """Render a plain-text report for download or README notes."""
    summary = summarize_events(result.get("events", []))
    lines = [
        "RouteGuard Analysis Report",
        "=" * 28,
        f"Safety score: {result['score']} / 100",
        f"Risk level: {result['risk_level']}",
        f"Video duration: {result.get('duration_seconds', 0):.1f}s",
        f"Processed frames: {result.get('frames', 0)}",
        f"Sampled frames: {result.get('sampled_frames', 0)}",
        f"Risk sampled-frame ratio: {result.get('risk_frame_ratio', 0):.0%}",
        f"Processing speed: {result.get('processing_fps', 0):.1f} FPS",
        f"Risk events: {summary['event_count']}",
        f"Total penalty: {summary['total_penalty']}",
        "",
        "Timeline",
        "-" * 28,
    ]

    if result.get("events"):
        for row in event_rows(result["events"]):
            lines.append(
                f"{row['시간']} | {row['위험도']} | {row['객체']} | "
                f"overlap {row['통로 겹침']} | penalty -{row['감점']}"
            )
            lines.append(f"  {row['피드백']}")
    else:
        lines.append("No path-blocking risk events were detected.")

    lines.extend(["", "Recommendations", "-" * 28])
    for recommendation in build_recommendations(result.get("events", []), result["score"]):
        lines.append(f"- {recommendation}")

    return "\n".join(lines)
