"""Deterministic audit/freeze tests that do not pretend to execute Stim."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from qec_decoder_lab import frozen_validation as fv
from qec_decoder_lab.paired_analysis import analyze_pair


def make_mock_case(root, protocol, spec, seed, progress):
    cid = fv._case_id(spec)
    paths = fv._case_paths(root, cid)
    paths['npz'].parent.mkdir(parents=True, exist_ok=True)
    n, paired = protocol['shots_per_case_mwpm'], protocol['paired_shots_per_case']
    rng = np.random.default_rng(seed)
    syndromes = rng.integers(0,2,(n,3),dtype=np.uint8)
    truth = syndromes[:, :1].copy()
    mwpm = truth.copy()
    mwpm[0] ^= 1
    recovery = truth[:paired].copy()
    recovery[1] ^= 1
    np.savez_compressed(paths['npz'], detection_events=syndromes,
                        actual_observables=truth, mwpm_predictions=mwpm,
                        recovery_predictions=recovery)
    paths['stim'].write_text('synthetic circuit fixture\n')
    paths['dem'].write_text('synthetic detector model fixture\n')
    summary = fv.summarize_case(syndromes, truth, mwpm, recovery, seed=seed,
                                bootstrap_repeats=protocol['bootstrap_repeats'])
    report = {'schema': fv.SCHEMA, 'case_id': cid, 'spec': spec,
              'protocol_sha256': protocol['protocol_sha256'], 'seed': seed,
              'n_shots': n, 'n_paired': paired, 'detectors': 3, 'observables': 1,
              'metrics': summary, 'status': 'COMPLETE',
              'artifact_sha256': {k: fv._digest(paths[k]) for k in ('npz','stim','dem')}}
    paths['json'].write_bytes(fv._canonical_bytes(report))
    return report


def test_plan_freezes_source_and_existing_plan_cannot_be_overwritten(tmp_path, monkeypatch):
    root = tmp_path/'run'
    plan = fv.create_protocol(root, profile='smoke', shots=12, paired_shots=5,
                              bootstrap_repeats=100)
    assert len(plan['case_specs']) == 2
    assert fv.load_locked_protocol(root) == plan
    with pytest.raises(FileExistsError):
        fv.create_protocol(root, profile='smoke', shots=12, paired_shots=5)
    monkeypatch.setattr(fv, '_source_manifest', lambda: {'tampered':'source'})
    with pytest.raises(ValueError, match='source hash'):
        fv.load_locked_protocol(root)


def test_tampered_locked_protocol_fails(tmp_path):
    root = tmp_path/'frozen'
    fv.create_protocol(root, profile='smoke', shots=10, paired_shots=4)
    path = root/fv.PROTOCOL_FILE
    data = json.loads(path.read_text())
    data['damping'] = 0.95
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='altered'):
        fv.load_locked_protocol(root)


def test_audit_recomputes_metrics_and_detects_corruption(tmp_path, monkeypatch):
    root = tmp_path/'run'
    fv.create_protocol(root, profile='smoke', shots=12, paired_shots=5,
                       bootstrap_repeats=100)
    monkeypatch.setattr(fv, '_capture_case', make_mock_case)
    completed = fv.run_study(root, progress=False)
    assert completed['status'] == 'PASS'
    assert completed['cases_verified'] == 2
    assert fv.run_study(root, progress=False)['status'] == 'PASS'  # verified resume
    p = fv._case_paths(root, fv._case_id(fv.load_locked_protocol(root)['case_specs'][0]))
    p['npz'].write_bytes(p['npz'].read_bytes() + b'corruption')
    report = fv.audit_study(root)
    assert report['status'] == 'INCOMPLETE'
    assert any('checksum' in row['error'] for row in report['failures'])


def test_audit_rejects_falsified_error_counts_and_missing_case(tmp_path, monkeypatch):
    root = tmp_path/'run'
    fv.create_protocol(root, profile='smoke', shots=12, paired_shots=5,
                       bootstrap_repeats=100)
    monkeypatch.setattr(fv, '_capture_case', make_mock_case)
    fv.run_study(root, progress=False)
    spec = fv.load_locked_protocol(root)['case_specs'][0]
    paths = fv._case_paths(root, fv._case_id(spec))
    item = json.loads(paths['json'].read_text())
    item['metrics']['recovery_paired_shots']['logical_failures'] = 99
    paths['json'].write_bytes(fv._canonical_bytes(item))
    report = fv.audit_study(root)
    assert report['status'] == 'INCOMPLETE'
    assert 'statistics' in report['failures'][0]['error']
    paths['json'].unlink()
    assert fv.audit_study(root)['status'] == 'INCOMPLETE'


def test_paired_statistics_use_identical_subsamples_and_wilson():
    syndromes = np.zeros((100,2),dtype=np.uint8)
    truth = np.zeros((100,1),dtype=np.uint8)
    mwpm = truth.copy(); mwpm[[0,1,60]] = 1
    recovery = truth[:40].copy(); recovery[[0,2]] = 1
    result = fv.summarize_case(syndromes, truth, mwpm, recovery,
                               seed=7, bootstrap_repeats=100)
    assert result['mwpm_all_shots']['logical_failures'] == 3
    assert result['mwpm_paired_shots']['logical_failures'] == 2
    assert result['recovery_paired_shots']['logical_failures'] == 2
    pair = result['recovery_minus_mwpm']
    assert pair['first_only_wrong'] == 1
    assert pair['second_only_wrong'] == 1
    assert pair['both_wrong'] == 1
    assert pair['shots'] == 40
    assert result['mwpm_all_shots']['wilson_95_high'] > 0.03


def test_protocol_rejects_illegal_sizes(tmp_path):
    with pytest.raises(ValueError, match='paired_shots'):
        fv.create_protocol(tmp_path/'bad', shots=5, paired_shots=6)


def test_distinct_predeclared_case_ids():
    for profile, expected in [('smoke',2),('targeted',8),('focused',12),('expanded',18)]:
        ids = [fv._case_id(x) for x in fv._case_grid(profile)]
        assert len(ids) == expected == len(set(ids))
