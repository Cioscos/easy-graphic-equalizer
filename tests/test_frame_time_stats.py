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


def test_report_format_median_max_lag():
    stats = FrameTimeStats(report_every=4, budget_s=0.01)  # soglia lag = 0.015 s
    stats.add(0.005)
    stats.add(0.005)
    stats.add(0.020)  # lag spike (> 15 ms)
    report = stats.add(0.005)
    assert report is not None, "il 4° frame deve produrre un report"
    # ordinati [5,5,5,20] ms → mediana = 5.00 ms, max = 20.00 ms, media = 8.75 ms
    assert "median=5.00ms" in report, report
    assert "avg=8.75ms" in report, report
    assert "max=20.00ms" in report, report
    # soglia lag = budget*1.5 = 15.0 ms; 1 frame oltre
    assert "lag-spikes(>15.0ms)=1" in report, report
    # gli fps sono calcolati dalla MEDIANA (robusta agli outlier): 1/0.005 = 200
    assert "~200fps mediani" in report, report


def test_resets_after_report():
    stats = FrameTimeStats(report_every=2, budget_s=0.01)
    assert stats.add(0.005) is None
    assert stats.add(0.005) is not None   # primo report
    assert stats.add(0.005) is None       # finestra azzerata
    assert stats.add(0.005) is not None   # secondo report


def main():
    test_no_report_before_window()
    test_report_format_median_max_lag()
    test_resets_after_report()
    print("OK")


if __name__ == "__main__":
    main()
