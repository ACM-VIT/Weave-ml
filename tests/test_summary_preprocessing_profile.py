"""Check timing boundaries without relying on noisy wall-clock thresholds."""
from unittest.mock import Mock

from scripts import profile_summary_preprocessing as profile


def test_cpu_clock_is_read_between_batches_not_per_action(monkeypatch):
    clock = Mock(side_effect=[0.0, 0.0, 0.0, 0.125])
    monkeypatch.setattr(profile.time, 'process_time', clock)
    action = Mock(return_value='measured result')
    result, report = profile.measure(action, repeats=5)
    assert result == 'measured result'
    assert report['cpu_iterations'] == 5 + 10 + 20
    assert action.call_count == 36  # Includes the separate allocation sample.
    assert clock.call_count == 4
    assert report['cpu_seconds_per_call'] == 0.125 / 35
    assert report['cpu_target_reached'] is True


def test_coarse_clock_cannot_create_an_unbounded_profile(monkeypatch):
    clock = Mock(return_value=0.0)
    monkeypatch.setattr(profile.time, 'process_time', clock)
    calls = 0

    def action():
        nonlocal calls
        calls += 1

    _, report = profile.measure(action, repeats=5)
    assert report['cpu_iterations'] == 100_000
    assert calls == 100_001
    assert clock.call_count < 100
    assert report['cpu_target_reached'] is False
