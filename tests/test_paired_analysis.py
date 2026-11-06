import numpy as np
import pytest

from qec_decoder_lab.paired_analysis import (
    analyze_pair, logical_error_per_round, mcnemar_exact_p,
)


def test_exact_mcnemar_extremes_and_symmetry():
    assert mcnemar_exact_p(0, 0) == 1
    assert mcnemar_exact_p(0, 5) == pytest.approx(0.0625)
    assert mcnemar_exact_p(5, 0) == pytest.approx(0.0625)
    assert mcnemar_exact_p(6, 6) == 1
    assert 0 < mcnemar_exact_p(1, 800) < 1e-200


def test_paired_resampling_and_direction():
    first = np.array([1, 0, 1, 0, 0, 0], bool)
    second = np.array([0, 1, 1, 0, 0, 0], bool)
    result = analyze_pair(first, second, seed=24, bootstrap_repeats=100)
    assert result['first_only_wrong'] == 1
    assert result['second_only_wrong'] == 1
    assert result['both_wrong'] == 1
    assert result['both_correct'] == 3
    assert result['second_minus_first_failure_rate'] == 0
    assert result == analyze_pair(first, second, seed=24, bootstrap_repeats=100)


def test_per_round_equivalent_inversion():
    assert logical_error_per_round(0, 9) == 0
    assert logical_error_per_round(.2, 1) == pytest.approx(.2)
    p = logical_error_per_round(.15, 5)
    assert (1-(1-2*p)**5)/2 == pytest.approx(.15)
    with pytest.raises(ValueError):
        logical_error_per_round(.6, 5)
