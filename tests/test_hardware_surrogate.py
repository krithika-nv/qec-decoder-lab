import json
from pathlib import Path

import pytest
from qec_decoder_lab.hardware_surrogate import surrogate_from_calibration


FIXTURE = Path(__file__).resolve().parents[1] / 'data/examples/synthetic_calibration.json'


def test_explicit_calibration_mapping():
    cfg, provenance = surrogate_from_calibration(FIXTURE, distance=3, rounds=3)
    p = cfg.noise_parameters()
    assert p == {
        'after_clifford_depolarization': .01,
        'before_round_data_depolarization': .002,
        'before_measure_flip_probability': .03,
        'after_reset_flip_probability': .015,
    }
    assert provenance['status'] == 'surrogate_only_not_hardware_validated'
    assert provenance['source_kind'] == 'synthetic_example'
    assert len(provenance['input_sha256']) == 64


def test_missing_assumptions_not_guessed(tmp_path):
    data = json.loads(FIXTURE.read_text())
    data['assumptions'].pop('after_reset_flip_probability')
    f = tmp_path / 'calibration.json'
    f.write_text(json.dumps(data))
    with pytest.raises(KeyError):
        surrogate_from_calibration(f, distance=3, rounds=3)


def test_non_utc_timestamp_rejected(tmp_path):
    data = json.loads(FIXTURE.read_text())
    data['calibration_timestamp_utc'] = '2026-01-01T00:00:00-07:00'
    f = tmp_path / 'calibration.json'
    f.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='UTC'):
        surrogate_from_calibration(f, distance=3, rounds=3)
