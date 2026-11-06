import numpy as np
import pytest

from qec_decoder_lab.exact import ExactBayesianDEMDecoder, Fault
from qec_decoder_lab.recovery import SyndromeRecoveryDecoder
from qec_decoder_lab.archive_eval import parse_flattened_dem, projected_exact_check


def test_recovery_satisfies_detector_constraints():
    faults = [Fault(.04, 0b0011), Fault(.02, 0b0101),
              Fault(.06, 0b0110), Fault(.01, 0b1100)]
    decoder = SyndromeRecoveryDecoder(3, 1, faults, iterations=12)
    inputs = np.array([[(i >> j) & 1 for j in range(3)] for i in range(8)], dtype=np.uint8)
    predictions, details = decoder.decode_batch(inputs, with_details=True)
    assert predictions.shape == (8, 1)
    assert details['fraction_syndrome_consistent'] == 1.0


def test_single_fault_recovery_matches_exact():
    decoder = SyndromeRecoveryDecoder(1, 1, [Fault(.01, 0b11)], iterations=15)
    oracle = ExactBayesianDEMDecoder(1, 1, [Fault(.01, 0b11)])
    rows = np.array([[0], [1]], dtype=np.uint8)
    assert np.array_equal(decoder.decode_batch(rows), oracle.decode_batch(rows))


def test_impossible_syndrome_is_rejected():
    dec = SyndromeRecoveryDecoder(2, 1, [Fault(.05, 0b001)])
    with pytest.raises(ValueError):
        dec.decode(np.array([0, 1], dtype=np.uint8))


def test_proj_exact_on_real_dem_shape():
    projected = projected_exact_check([Fault(.01, 0b011), Fault(.02, 0b101)], 2, 1,
                                      projected_detectors=1)
    assert projected['exact_oracle_states'] == 4
    assert projected['reachable_syndromes'] > 0
    assert 0 <= projected['weighted_recovery_exact_map_agreement'] <= 1


def test_flattened_dem_separator_is_one_fault(tmp_path):
    p = tmp_path/'example.dem'
    p.write_text('error(0.01) D0 L0 ^ D1 L0\nerror(0.05) D2\n')
    faults = parse_flattened_dem(p, 3, 1)
    assert len(faults) == 2
    assert faults[0].mask == 0b011
    with pytest.raises(ValueError, match='unsupported'):
        p.write_text('unknown(1) D0\n')
        parse_flattened_dem(p, 3, 1)


def test_repeat_and_detector_shift_expansion(tmp_path):
    p = tmp_path / 'compressed.dem'
    p.write_text('repeat 2 {\n  error(0.1) D0 L0\n  shift_detectors 1\n}\n')
    faults = parse_flattened_dem(p, 2, 1)
    assert [f.mask for f in faults] == [0b101, 0b110]
