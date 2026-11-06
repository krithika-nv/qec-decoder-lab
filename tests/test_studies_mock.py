"""Software-level orchestration test using explicitly mocked quantum backend."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from qec_decoder_lab import studies
from qec_decoder_lab.stats import summarize_errors


def test_matrix_orchestration_and_case_audit_with_mock_backend(monkeypatch, tmp_path):
    def fake_run(settings, *, shots_path):
        shots_path.parent.mkdir(parents=True, exist_ok=True)
        truth = np.zeros((settings.shots, 1), dtype=np.uint8)
        mwpm = truth.copy()
        mwpm[0, 0] = 1
        syndromes = np.zeros((settings.shots, 2), dtype=np.uint8)
        syndromes[0, 0] = 1
        np.savez_compressed(shots_path, detection_events=syndromes,
                            actual_observables=truth, mwpm_predictions=mwpm)
        return {
            'parameters': {'shots': settings.shots, 'effective_noise_probabilities':
                           settings.config.noise_parameters(), 'rounds': settings.config.rounds},
            'model': {'num_detectors': 2, 'num_observables': 1,
                      'circuit_sha256': hashlib.sha256(b'fake_circuit').hexdigest(),
                      'dem_sha256': hashlib.sha256(b'fake_dem').hexdigest()},
            'mwpm': summarize_errors(np.any(mwpm != truth, axis=1)),
            'bayesian': {'status': 'not_requested'},
            'bp_approximate': {'status': 'not_requested'},
        }

    def fake_files(dest, cfg, model):
        dest.with_suffix('.stim').write_bytes(b'fake_circuit')
        dest.with_suffix('.dem').write_bytes(b'fake_dem')
        return {'stim': str(dest.with_suffix('.stim'))}

    monkeypatch.setattr(studies, 'run_experiment', fake_run)
    monkeypatch.setattr(studies, 'save_model_files', fake_files)
    monkeypatch.setattr(studies, 'time_decoders', lambda *a, **k: {'mwpm': {'median_us': 4}})
    report = studies.run_benchmark(
        tmp_path, distances=[3], ps=[.001], bases=['x'],
        profiles=['circuit_level', 'data_only'], shots=5, seed=111,
        approximate=False, timing_shots=2, timing_repeats=2, bp_iterations=5,
        replay=False, plot=False,
    )
    assert report['status'] == 'PASS'
    assert len(report['cases']) == 2
    assert all(x['audit']['status'] == 'PASS' for x in report['cases'])
    with (tmp_path / 'summary.csv').open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 2
    assert {r['noise_profile'] for r in rows} == {'data_only', 'circuit_level'}
    assert json.loads((tmp_path/'study_manifest.json').read_text())['status'] == 'PASS'


def test_resume_uses_verified_checkpoint_without_re_running_backend(monkeypatch, tmp_path):
    # Reuse the test-specific backend above by running the orchestration test's
    # fixture separately would duplicate its setup; use a minimal direct fake.
    from qec_decoder_lab import studies

    def fake_run(settings, *, shots_path):
        shots_path.parent.mkdir(parents=True, exist_ok=True)
        syndromes = np.zeros((settings.shots, 2), dtype=np.uint8)
        truth = np.zeros((settings.shots, 1), dtype=np.uint8)
        mwpm = truth.copy()
        np.savez_compressed(shots_path, detection_events=syndromes,
                            actual_observables=truth, mwpm_predictions=mwpm)
        return {
            'parameters': {'shots': settings.shots, 'rounds': settings.config.rounds,
                           'effective_noise_probabilities': settings.config.noise_parameters()},
            'model': {'num_detectors': 2, 'num_observables': 1,
                      'circuit_sha256': hashlib.sha256(b'fake_circuit').hexdigest(),
                      'dem_sha256': hashlib.sha256(b'fake_dem').hexdigest()},
            'mwpm': summarize_errors(np.zeros(settings.shots, dtype=bool)),
            'bayesian': {'status': 'not_requested'},
            'bp_approximate': {'status': 'not_requested'},
        }

    def fake_files(dest, cfg, model):
        dest.with_suffix('.stim').write_bytes(b'fake_circuit')
        dest.with_suffix('.dem').write_bytes(b'fake_dem')
        return {'stim': str(dest.with_suffix('.stim'))}

    monkeypatch.setattr(studies, 'run_experiment', fake_run)
    monkeypatch.setattr(studies, 'save_model_files', fake_files)
    monkeypatch.setattr(studies, 'time_decoders', lambda *a, **k: {'mwpm': {'median_us': 4}})
    params = dict(distances=[3], ps=[.001], bases=['x'], profiles=['data_only'],
                  shots=8, seed=123, approximate=False, timing_shots=2,
                  timing_repeats=1, bp_iterations=2, replay=False, plot=False)
    first = studies.run_benchmark(tmp_path, **params)
    assert first['status'] == 'PASS'
    assert len(list((tmp_path / 'checkpoints').glob('*.json'))) == 1
    def forbidden(*args, **kwargs):
        raise AssertionError('backend must not execute on validated resume')
    monkeypatch.setattr(studies, 'run_experiment', forbidden)
    second = studies.run_benchmark(tmp_path, **params, resume=True)
    assert second['status'] == 'PASS'
    assert second['resumed_cases'] == 1
    progress = json.loads((tmp_path / 'study_progress.json').read_text())
    assert progress['status'] == 'PASS'
    assert progress['completed_cases'] == 1
    # Distinct shot counts must not reuse a checkpoint from another protocol.
    third = studies.run_benchmark(tmp_path, **{**params, 'shots': 9}, resume=True)
    assert third['status'] == 'INCOMPLETE'
    assert third['errors'] and 'AssertionError' in third['errors'][0]['error']
