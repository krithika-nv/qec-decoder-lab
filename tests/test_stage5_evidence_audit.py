"""Synthetic I/O audit tests; not a real quantum benchmark."""
import csv
import json
import sys
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from qec_decoder_lab import literature
from qec_decoder_lab.stage5_audit import audit_stage5, Stage5EvidenceError


@pytest.fixture
def saved_paper_case(tmp_path, monkeypatch):
    class Sampler:
        def sample(self, shots, separate_observables):
            assert separate_observables
            detector = np.zeros((shots, 9), dtype=np.uint8)
            detector[0, -1] = 1
            truth = np.zeros((shots, 1), dtype=np.uint8)
            truth[0, 0] = 1
            return detector, truth

    class Circuit:
        def detector_error_model(self, **kwargs):
            assert kwargs == {'decompose_errors': True}
            return 'error(0.01) D8 L0'

        def compile_detector_sampler(self, seed):
            return Sampler()

        def __str__(self):
            return 'H 0\nM 0'

    class Matching:
        @staticmethod
        def from_detector_error_model(dem):
            return Matching()

        def decode_batch(self, shots):
            return np.zeros((len(shots), 1), dtype=np.uint8)

    monkeypatch.setitem(sys.modules, 'stim', SimpleNamespace(
        Circuit=SimpleNamespace(generated=lambda *a, **k: Circuit()), __version__='TEST_ONLY'))
    monkeypatch.setitem(sys.modules, 'pymatching', SimpleNamespace(Matching=Matching))
    monkeypatch.setattr(literature, 'version', lambda _: 'TEST_ONLY')
    source = Path(__file__).resolve().parents[1] / 'data/literature/pymatching_v2_p001.csv'
    out = tmp_path / 'paper'
    literature.run_matched_benchmark(source, out, distances=(5, 7), shots=10, repeats=2)
    return out


def test_stage5_audit_recomputes_shots_errors_timings_and_hashes(saved_paper_case):
    report = audit_stage5(saved_paper_case)
    assert report['status'] == 'PASS'
    assert report['distance_count'] == 2
    assert not report['all_nine_distances']
    assert report['verified'][0]['logical_failures_verified'] == 1
    with np.load(saved_paper_case / 'd5_raw_shots.npz', allow_pickle=False) as archive:
        bits = np.unpackbits(archive['detectors_packed'], axis=1, bitorder='little')[:, :9]
        assert bits[0, -1] == 1
        assert bits.sum() == 1


def test_stage5_full_coverage_fails_closed(saved_paper_case):
    with pytest.raises(Stage5EvidenceError, match='Incomplete'):
        audit_stage5(saved_paper_case, require_all_nine=True)


def test_stage5_rejects_tampered_raw_shot_file(saved_paper_case):
    path = saved_paper_case / 'd5_raw_shots.npz'
    path.write_bytes(path.read_bytes() + b'MUTATION')
    with pytest.raises(Stage5EvidenceError, match='SHA256'):
        audit_stage5(saved_paper_case)


def test_stage5_rejects_altered_timing_trial(saved_paper_case):
    path = saved_paper_case / 'evidence.json'
    data = json.loads(path.read_text())
    data['local'][0]['timing_trials_wall_seconds'][0] += 10
    path.write_text(json.dumps(data))
    with pytest.raises(Stage5EvidenceError, match='median'):
        audit_stage5(saved_paper_case)


def test_stage5_rejects_extra_csv_distance(saved_paper_case):
    path = saved_paper_case / 'timing_summary.csv'
    with path.open('a', newline='') as handle:
        handle.write('50,0.001,10,2,1,1,1,1,1,1,0\n')
    with pytest.raises(Stage5EvidenceError, match='CSV|SHA256'):
        audit_stage5(saved_paper_case)


def test_stage5_resume_reuses_verified_distances(saved_paper_case):
    source = Path(__file__).resolve().parents[1] / 'data/literature/pymatching_v2_p001.csv'
    outcome = literature.run_matched_benchmark(
        source, saved_paper_case, distances=(5, 7, 9), shots=10, repeats=2,
        resume=True,
    )
    assert outcome['resumed_distances_from_checkpoint'] == 2
    assert [r['distance'] for r in outcome['local']] == [5, 7, 9]
    assert audit_stage5(saved_paper_case)['distance_count'] == 3


def test_stage5_resume_rejects_changed_seed(saved_paper_case):
    source = Path(__file__).resolve().parents[1] / 'data/literature/pymatching_v2_p001.csv'
    with pytest.raises(ValueError, match='Cannot resume'):
        literature.run_matched_benchmark(
            source, saved_paper_case, distances=(5, 7), shots=10, repeats=2,
            seed=987, resume=True,
        )
