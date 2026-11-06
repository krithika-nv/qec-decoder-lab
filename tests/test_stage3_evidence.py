import hashlib
import json

import numpy as np
import pytest

from qec_decoder_lab.evidence import EvidenceError, verify_case


def make_case(tmp_path):
    path = tmp_path / 'd3.json'
    circuit, dem = b'circuit\n', b'error(0.1) D0 L0\n'
    path.with_suffix('.stim').write_bytes(circuit)
    path.with_suffix('.dem').write_bytes(dem)
    syndrome = np.array([[0], [1], [0]], dtype=np.uint8)
    truth = np.array([[0], [1], [1]], dtype=np.uint8)
    mwpm = np.array([[0], [1], [0]], dtype=np.uint8)
    np.savez_compressed(path.with_suffix('.npz'), detection_events=syndrome,
                        actual_observables=truth, mwpm_predictions=mwpm)
    from qec_decoder_lab.stats import summarize_errors
    case = {
        'parameters': {'shots': 3, 'effective_noise_probabilities': {'a': 0.1}},
        'model': {'num_detectors': 1, 'num_observables': 1,
                  'circuit_sha256': hashlib.sha256(circuit).hexdigest(),
                  'dem_sha256': hashlib.sha256(dem).hexdigest()},
        'mwpm': summarize_errors(np.array([False, False, True])),
        'bayesian': {'status': 'not_requested'},
        'bp_approximate': {'status': 'not_requested'},
    }
    path.write_text(json.dumps(case))
    return path


def test_archived_evidence_passes_without_stim(tmp_path):
    path = make_case(tmp_path)
    result = verify_case(path)
    assert result['status'] == 'PASS'
    assert 'raw_shots' in result['gates']
    assert result['replay'] == 'NOT_REQUESTED'


def test_detects_hash_tampering(tmp_path):
    path = make_case(tmp_path)
    path.with_suffix('.dem').write_text('tampered')
    with pytest.raises(EvidenceError, match='SHA256'):
        verify_case(path)


def test_detects_failure_rate_tampering(tmp_path):
    path = make_case(tmp_path)
    data = json.loads(path.read_text())
    data['mwpm']['logical_failures'] = 0
    path.write_text(json.dumps(data))
    with pytest.raises(EvidenceError, match='MWPM summary'):
        verify_case(path)


def test_detects_missing_shots(tmp_path):
    path = make_case(tmp_path)
    path.with_suffix('.npz').unlink()
    with pytest.raises(FileNotFoundError):
        verify_case(path)
