"""Tests for sample report helpers."""

from pathlib import Path

from src.sample_report import discover_sample_videos


def test_discover_sample_videos(tmp_path):
    (tmp_path / "safe").mkdir()
    (tmp_path / "safe" / "safe.mp4").write_bytes(b"fake")
    (tmp_path / "safe" / "note.txt").write_text("ignore", encoding="utf-8")

    videos = discover_sample_videos(tmp_path)

    assert videos == [Path(tmp_path / "safe" / "safe.mp4")]
