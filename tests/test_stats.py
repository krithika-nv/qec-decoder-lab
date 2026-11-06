import numpy as np
import pytest

from qec_decoder_lab.stats import compare_paired, summarize_errors, wilson_interval


def test_wilson_zero_events_has_nonzero_upper_bound():
    lo, hi = wilson_interval(0, 1000)
    assert lo == 0 and 0 < hi < 0.01


def test_wilson_all_events():
    lo, hi = wilson_interval(100, 100)
    assert 0.9 < lo < 1 and hi == 1


def test_failure_summary():
    r = summarize_errors(np.array([0, 1, 1, 0], dtype=np.uint8))
    assert r["logical_failures"] == 2 and r["logical_failure_rate"] == .5


def test_paired_comparison_uses_same_shots():
    x = np.array([1, 0, 1, 0])
    y = np.array([0, 1, 1, 0])
    d = compare_paired(x, y)
    assert d["mwpm_only_wrong"] == 1
    assert d["bayesian_only_wrong"] == 1
    assert d["both_wrong"] == 1
    assert d["both_correct"] == 1
    assert d["bayesian_minus_mwpm_rate"] == 0


def test_no_empty_sample():
    with pytest.raises(ValueError):
        summarize_errors(np.array([]))
