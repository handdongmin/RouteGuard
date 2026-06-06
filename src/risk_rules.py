"""Risk-scoring rules for indoor evacuation routes."""


SEVERITY_PENALTIES = {
    "caution": 10,
    "danger": 20,
}


KOREAN_LABELS = {
    "backpack": "가방",
    "bed": "침대",
    "bench": "벤치",
    "book": "책",
    "bottle": "병",
    "bowl": "그릇",
    "chair": "의자",
    "couch": "소파",
    "cup": "컵",
    "dining table": "테이블",
    "handbag": "가방",
    "keyboard": "키보드",
    "laptop": "노트북",
    "mouse": "마우스",
    "potted plant": "화분",
    "skateboard": "보드",
    "sports ball": "공",
    "suitcase": "캐리어",
    "teddy bear": "인형",
    "umbrella": "우산",
    "vase": "화병",
}


def build_risk(frame_analysis: dict) -> dict | None:
    """Convert a path-overlap measurement into a risk record."""
    severity = frame_analysis.get("severity", "safe")
    if severity == "safe":
        return None

    label = frame_analysis.get("label", "object")
    object_name = KOREAN_LABELS.get(label, label)
    overlap = frame_analysis.get("overlap_ratio", 0.0)
    penalty = SEVERITY_PENALTIES[severity]
    if severity == "danger" and frame_analysis.get("center_in_path"):
        penalty += 5

    if severity == "danger":
        message = f"{object_name} 후보가 이동 경로 중앙을 크게 막고 있습니다."
    else:
        message = f"{object_name} 후보가 이동 경로 일부와 겹칩니다."

    return {
        "label": label,
        "object_name": object_name,
        "severity": severity,
        "penalty": penalty,
        "overlap_ratio": overlap,
        "bbox_area_ratio": frame_analysis.get("bbox_area_ratio", 0.0),
        "risk_group": frame_analysis.get("risk_group", "other"),
        "message": message,
    }


def calculate_route_score(risks: list[dict]) -> int:
    """Return a 0-100 score where a higher value indicates a clearer route."""
    penalty = sum(int(risk.get("penalty", 0)) for risk in risks)
    return max(0, min(100, 100 - penalty))


def risk_level(score: int) -> str:
    """Return a compact Korean risk level for the route score."""
    if score >= 80:
        return "안전"
    if score >= 50:
        return "주의"
    return "위험"
