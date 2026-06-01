"""Rendering helpers for annotated videos and reports."""


def render_report(risks: list[dict]) -> list[str]:
    """Convert risk records into short timestamped report lines."""
    return [
        f"{risk.get('timestamp', '00:00')}  {risk.get('message', 'Potential risk')}"
        for risk in risks
    ]
