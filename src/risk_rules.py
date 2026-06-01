"""Risk-scoring rules for indoor evacuation routes."""


def calculate_route_score(risks: list[dict]) -> int:
    """Return a 0-100 score where a higher value indicates a clearer route."""
    penalty = sum(int(risk.get("penalty", 0)) for risk in risks)
    return max(0, min(100, 100 - penalty))
