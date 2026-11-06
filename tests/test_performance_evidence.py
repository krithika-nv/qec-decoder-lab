import json
import statistics

import pytest

from qec_decoder_lab import performance
from qec_decoder_lab import frozen_validation as fv


def stub_performance(tmp_path, monkeypatch):
    root = tmp_path/'heldout'
    plan = fv.create_protocol(root, profile='smoke', shots=8, paired_shots=4)
    out = root/'performance'
    out.mkdir()
    records = []
    for spec in plan['case_specs']:
        cid = fv._case_id(spec)
        for dec in ('mwpm','recovery'):
            times = [.00002,.00003,.00004]
            data = {
                'decoder': dec, 'case_id': cid,
                'protocol_sha256': plan['protocol_sha256'], 'shots':4,
                'repetitions':3, 'trial_wall_seconds':times,
                'median_microseconds_per_shot': statistics.median(times)*1e6/4,
                'raw_shots_sha256':'known_hash',
                'peak_process_rss_bytes':1200000,
                'predictions_replayed_against_saved_evidence': True,
                'machine':{'platform':'test','python':'3.12','processor':'test'},
            }
            (out/f'{cid}_{dec}.json').write_text(json.dumps(data))
            records.append(data)
    (out/'performance_results.json').write_text(json.dumps({
        'status':'PASS','protocol_sha256':plan['protocol_sha256'],'records':records}))
    monkeypatch.setattr(performance, '_digest', lambda path: 'known_hash')
    return root, out


def test_perf_audit_checks_fresh_worker_record_reconciliation(tmp_path, monkeypatch):
    root, out = stub_performance(tmp_path, monkeypatch)
    assert performance.audit_performance(root, out)['decoder_workers'] == 4
    p = next(out.glob('*_mwpm.json'))
    x = json.loads(p.read_text())
    x['median_microseconds_per_shot'] = 8
    p.write_text(json.dumps(x))
    with pytest.raises(ValueError, match='timing record'):
        performance.audit_performance(root, out)


def test_perf_audit_rejects_changed_machine(tmp_path, monkeypatch):
    root, out = stub_performance(tmp_path, monkeypatch)
    manifest = out/'performance_results.json'
    data = json.loads(manifest.read_text())
    data['records'][0]['machine']['platform'] = 'another-host'
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='different machines'):
        performance.audit_performance(root, out)


def test_bundle_requires_complete_validated_outputs(tmp_path):
    from qec_decoder_lab.validation_bundle import pack
    root = tmp_path/'heldout'
    fv.create_protocol(root,profile='smoke',shots=8,paired_shots=4)
    with pytest.raises(RuntimeError, match='incomplete'):
        pack(root, tmp_path/'bundle.zip',require_performance=False)
