import hashlib
import json

import numpy as np
import pytest

from qec_decoder_lab.evidence import EvidenceError, verify_case
from qec_decoder_lab.paired_analysis import holm_adjust
from qec_decoder_lab.stats import summarize_errors
from qec_decoder_lab.studies import analyze_archived_case


def test_holm_adjustment_preserves_familywise_order():
    assert holm_adjust([.01, .04, .03]) == pytest.approx([.03, .06, .06])
    assert holm_adjust([]) == []
    with pytest.raises(ValueError):
        holm_adjust([float('nan')])


def test_bayesian_subset_uses_exactly_matched_shots(tmp_path):
    path = tmp_path / 'run.json'
    truth = np.zeros((6, 1), dtype=np.uint8)
    mwpm = truth.copy()
    mwpm[[0, 4], 0] = 1
    bp = np.array([[0], [1], [0]], dtype=np.uint8)
    np.savez_compressed(path.with_suffix('.npz'),
                        detection_events=np.zeros((6, 1), dtype=np.uint8),
                        actual_observables=truth, mwpm_predictions=mwpm,
                        bp_predictions=bp, bp_logical_one_probabilities=bp.astype(float))
    circuit, dem = b'circuit', b'error(0.1) D0 L0'
    path.with_suffix('.stim').write_bytes(circuit)
    path.with_suffix('.dem').write_bytes(dem)
    wrong_mwpm = np.any(mwpm != truth, axis=1)
    wrong_bp = np.any(bp != truth[:3], axis=1)
    case = {
        'parameters': {'shots': 6, 'rounds': 3, 'effective_noise_probabilities': {'a': .01}},
        'model': {'num_detectors': 1, 'num_observables': 1,
                  'circuit_sha256': hashlib.sha256(circuit).hexdigest(),
                  'dem_sha256': hashlib.sha256(dem).hexdigest(),
                  'raw_shots_sha256': hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()},
        'mwpm': summarize_errors(wrong_mwpm),
        'bayesian': {'status': 'not_requested'},
        'bp_approximate': {'status': 'completed', 'matched_shots': 3, **summarize_errors(wrong_bp)},
        'bp_comparisons': {'vs_mwpm': {
            'mwpm_only_wrong': 1, 'bp_only_wrong': 1,
            'both_wrong': 0, 'both_correct': 1,
        }},
    }
    path.write_text(json.dumps(case))
    report = verify_case(path, require_raw_hash=True)
    assert 'raw_shots_sha256' in report['gates']
    comparison = analyze_archived_case(path)
    assert comparison['rates']['mwpm']['shots'] == 6
    assert comparison['rates']['loopy_bp']['shots'] == 3
    pair = comparison['paired_comparisons']['mwpm_vs_loopy_bp']
    assert pair['shots'] == 3
    assert pair['first_only_wrong'] == 1
    assert pair['second_only_wrong'] == 1
    path.with_suffix('.npz').write_bytes(path.with_suffix('.npz').read_bytes() + b'tamper')
    with pytest.raises(EvidenceError, match='SHA256'):
        verify_case(path, require_raw_hash=True)
