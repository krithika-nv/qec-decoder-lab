"""Regression tests for archived-data validation and explicit paired labels."""
import hashlib
import json

import numpy as np
import pytest

from qec_decoder_lab.archive_eval import evaluate_case
from qec_decoder_lab.studies import analyze_archived_case


def _saved_case(tmp_path):
    file = tmp_path / 'case.json'
    syndrome = np.array([[0], [1], [0], [1]], dtype=np.uint8)
    truth = syndrome.copy()
    mwpm = truth.copy()
    original_bp = np.zeros_like(truth)
    np.savez_compressed(file.with_suffix('.npz'), detection_events=syndrome,
                        actual_observables=truth, mwpm_predictions=mwpm,
                        bp_predictions=original_bp)
    file.with_suffix('.dem').write_text('error(0.05) D0 L0\n')
    checksum = hashlib.sha256(file.with_suffix('.npz').read_bytes()).hexdigest()
    file.write_text(json.dumps({'parameters': {'rounds': 1},
                                'model': {'raw_shots_sha256': checksum}}))
    return file


def test_saved_shots_recovery_and_explicit_pair_order(tmp_path):
    saved = _saved_case(tmp_path)
    result = evaluate_case(saved, shots=4)
    assert result['hashes']['matches_recorded_raw_hash']
    assert result['diagnostics']['fraction_syndrome_consistent'] == 1.0
    assert result['counts'] == {'mwpm_wrong': 0, 'old_bp_wrong': 2, 'recovery_wrong': 0}
    assert result['paired_recovery_vs_mwpm']['shots'] == 4
    archived = analyze_archived_case(saved)
    assert archived['paired_comparisons']['mwpm_vs_loopy_bp']['first_decoder'] == 'mwpm'
    assert archived['paired_comparisons']['mwpm_vs_loopy_bp']['second_decoder'] == 'loopy_bp'


def test_saved_shot_hash_tamper_fails(tmp_path):
    saved = _saved_case(tmp_path)
    saved.with_suffix('.npz').write_bytes(b'modified')
    with pytest.raises(ValueError, match='hash mismatch'):
        evaluate_case(saved)


def test_missing_raw_evidence_fails(tmp_path):
    saved = _saved_case(tmp_path)
    saved.with_suffix('.dem').unlink()
    with pytest.raises(FileNotFoundError):
        evaluate_case(saved)
