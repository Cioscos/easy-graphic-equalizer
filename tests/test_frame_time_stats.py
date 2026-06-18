"""
Verifica headless (niente pytest nel repo): asserzioni pure su FrameTimeStats.
Eseguire dalla radice del repo con:  uv run python tests/test_frame_time_stats.py
Uscita 0 + "OK" = passato.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from thread.frame_time_stats import FrameTimeStats


def test_no_report_before_window():
    stats = FrameTimeStats(report_every=4, budget_s=0.01)
    assert stats.add(0.005) is None
    assert stats.add(0.005) is None
    assert stats.add(0.005) is None


def test_report_at_window_boundary():
    stats = FrameTimeStats(report_every=4, budget_s=0.01)
    stats.add(0.005)
    stats.add(0.005)
    stats.add(0.015)  # oltre budget
    report = stats.add(0.005)
    assert report is not None, "il 4° frame deve produrre un report"
    # avg = (0.005+0.005+0.015+0.005)/4 = 0.0075 s = 7.50 ms
    assert "avg=7.50ms" in report, report
    assert "min=5.00ms" in report, report
    assert "max=15.00ms" in report, report
    # 1 frame su 4 oltre budget = 25%
    assert "oltre-budget=25%" in report, report


def test_resets_after_report():
    stats = FrameTimeStats(report_every=2, budget_s=0.01)
    assert stats.add(0.005) is None
    assert stats.add(0.005) is not None   # primo report
    assert stats.add(0.005) is None       # finestra azzerata
    assert stats.add(0.005) is not None   # secondo report


def main():
    test_no_report_before_window()
    test_report_at_window_boundary()
    test_resets_after_report()
    print("OK")


if __name__ == "__main__":
    main()
