"""Tests for route safety scoring rules."""

from src.risk_rules import build_risk, calculate_route_score, risk_level


def test_chair_receives_larger_penalty_than_bag():
    chair = build_risk(
        {
            "label": "chair",
            "severity": "danger",
            "overlap_ratio": 0.45,
            "center_in_path": True,
            "bbox_area_ratio": 0.2,
            "risk_group": "large",
        }
    )
    bag = build_risk(
        {
            "label": "backpack",
            "severity": "danger",
            "overlap_ratio": 0.45,
            "center_in_path": True,
            "bbox_area_ratio": 0.2,
            "risk_group": "large",
        }
    )

    assert chair["penalty"] > bag["penalty"]


def test_multiple_risks_trigger_congestion_penalty():
    risks = [
        {"penalty": 17},
        {"penalty": 37},
        {"penalty": 30},
    ]

    assert calculate_route_score(risks) == 20


def test_score_levels():
    assert risk_level(90) == "안전"
    assert risk_level(65) == "주의"
    assert risk_level(20) == "위험"
