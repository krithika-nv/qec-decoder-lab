import numpy as np
import pytest

from qec_decoder_lab.belief_propagation import BeliefPropagationDEMDecoder
from qec_decoder_lab.exact import ExactBayesianDEMDecoder, Fault


def test_tree_fault_logical_marginals_match_exact():
    # One parity-check factor and only one fault changes the logical frame.
    faults = [Fault(.1, 0b11), Fault(.05, 0b01)]
    bp = BeliefPropagationDEMDecoder(1, 1, faults, iterations=50, damping=0)
    exact = ExactBayesianDEMDecoder(1, 1, faults)
    for y in ([0], [1]):
        result = bp.decode(np.array(y))
        expected = exact.joint[1, y[0]] / exact.evidence[y[0]]
        assert result.logical_one_probabilities[0] == pytest.approx(expected, abs=1e-10)
        assert result.prediction[0] == exact.decode_batch(np.array([y]))[0, 0]
        assert result.converged


def test_independent_logical_fault_without_detectors():
    bp = BeliefPropagationDEMDecoder(0, 1, [Fault(.2, 0b1)])
    r = bp.decode(np.empty(0, dtype=int))
    assert r.logical_one_probabilities[0] == pytest.approx(.2, abs=1e-10)
    assert r.iterations >= 1


def test_duplicate_syndromes_batch_and_diagnostics():
    bp = BeliefPropagationDEMDecoder(1, 1, [Fault(.1, 3), Fault(.05, 1)])
    y = np.array([[0], [1], [0], [1], [1]], dtype=np.uint8)
    preds, probs, stats = bp.decode_batch(y, with_details=True)
    assert preds.shape == probs.shape == (5, 1)
    assert stats['unique_syndromes'] == 2
    assert stats['graph_edges'] == 2
    assert preds[:, 0].tolist() == [0, 1, 0, 1, 1]
    assert np.array_equal(probs[[0, 2]], probs[[0, 0]])


def test_degree_two_factor_parity_symmetry():
    # Identically weighted faults: odd syndrome gives 50/50 marginals.
    bp = BeliefPropagationDEMDecoder(1, 2, [Fault(.1, 0b011), Fault(.1, 0b101)])
    r = bp.decode(np.array([1]))
    assert np.allclose(r.logical_one_probabilities, [0.5, 0.5], atol=1e-8)
    # Factorization makes no joint logical posterior claim.


def test_invalid_inputs_and_isolated_detector():
    with pytest.raises(ValueError):
        BeliefPropagationDEMDecoder(1, 1, [Fault(.2, 8)])
    with pytest.raises(ValueError):
        BeliefPropagationDEMDecoder(1, 1, [Fault(.2, 1)], damping=1)
    with pytest.raises(ValueError):
        BeliefPropagationDEMDecoder(1, 1, [Fault(.2, 1)]).decode_batch(np.array([[2]]))
    with pytest.raises(ValueError, match='impossible'):
        BeliefPropagationDEMDecoder(1, 1, []).decode(np.array([1]))


def test_disjoint_detector_graph_large_no_exponential_states():
    n = 200
    faults = [Fault(.03, 1 << i) for i in range(n)] + [Fault(.02, 1 << n)]
    bp = BeliefPropagationDEMDecoder(n, 1, faults)
    y = np.zeros(n, dtype=int)
    y[12] = 1
    r = bp.decode(y)
    assert r.prediction.shape == (1,)
    assert r.logical_one_probabilities[0] == pytest.approx(.02, abs=1e-8)
    assert r.hard_fault_syndrome_matches


def test_progress_callback_counts_unique_syndromes():
    decoder = BeliefPropagationDEMDecoder(1, 1, [Fault(.1, 3), Fault(.05, 1)])
    events = []
    decoder.decode_batch(np.array([[0], [1], [0], [1]], dtype=np.uint8),
                         with_details=True,
                         progress_callback=lambda n, total: events.append((n, total)),
                         progress_every=1)
    assert events == [(1, 2), (2, 2)]
