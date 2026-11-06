from itertools import product

import numpy as np
import pytest

from qec_decoder_lab.exact import ExactBayesianDEMDecoder, Fault, exact_joint_distribution


def test_exact_joint_distribution_matches_exhaustive_enumeration():
    faults = [Fault(.2, 1), Fault(.3, 3), Fault(.1, 2), Fault(.4, 0)]
    joint = exact_joint_distribution(1, 1, faults)
    brute = np.zeros((2, 2))
    for bits in product((0, 1), repeat=len(faults)):
        p = 1.0
        outcome = 0
        for bit, fault in zip(bits, faults):
            p *= fault.probability if bit else 1 - fault.probability
            if bit:
                outcome ^= fault.mask
        brute[outcome >> 1, outcome & 1] += p
    np.testing.assert_allclose(joint, brute, atol=1e-14)


def test_exact_map_marginalizes_faults_not_just_min_weight():
    # Two independent logical faults lead to P(logical=1) = .8*.7+.2*.3=.62.
    d = ExactBayesianDEMDecoder(0, 1, [Fault(.8, 1), Fault(.3, 1)])
    assert d.decode_batch(np.zeros((2, 0), dtype=np.uint8)).tolist() == [[1], [1]]


def test_syndrome_observable_prediction():
    d = ExactBayesianDEMDecoder(1, 1, [Fault(.2, 3), Fault(.05, 1)])
    predictions = d.decode_batch(np.array([[0], [1]], dtype=np.uint8))
    assert predictions.shape == (2, 1)
    assert predictions[:, 0].tolist() == [0, 1]


def test_probabilities_normalized():
    joint = exact_joint_distribution(2, 1, [Fault(.01, 1), Fault(.02, 4)])
    assert abs(joint.sum() - 1) < 1e-14


def test_rejects_exponential_states():
    with pytest.raises(ValueError, match="max_bits"):
        exact_joint_distribution(20, 1, [], max_bits=16)


def test_rejects_unseen_syndrome():
    d = ExactBayesianDEMDecoder(1, 1, [])
    with pytest.raises(ValueError, match="zero probability"):
        d.decode_batch(np.array([[1]]))


def test_rejects_nonnumeric_probabilities():
    with pytest.raises(ValueError):
        exact_joint_distribution(1, 1, [Fault(-0.1, 1)])
