"""Real simulator tests executed on CI when Stim/PyMatching are installed."""
import pytest

stim = pytest.importorskip('stim')
pymatching = pytest.importorskip('pymatching')

from qec_decoder_lab.frozen_validation import create_protocol, run_study, audit_study


def test_fresh_real_stim_heldout_and_replay(tmp_path):
    root = tmp_path/'heldout'
    create_protocol(root, profile='smoke', shots=16, paired_shots=8,
                    bootstrap_repeats=100, bp_iterations=8)
    result = run_study(root, progress=False)
    assert result['status'] == 'PASS'
    audit = audit_study(root, redo_mwpm=True, redo_recovery=True, replay_seed=True)
    assert audit['status'] == 'PASS'
    assert audit['cases_verified'] == 2
    from qec_decoder_lab.performance import benchmark_all, audit_performance
    from qec_decoder_lab.validation_bundle import pack
    perf = benchmark_all(root, timing_shots=4, repetitions=2)
    assert perf['status'] == 'PASS'
    assert audit_performance(root)['decoder_workers'] == 4
    bundle_path = tmp_path/'bundle.zip'
    assert pack(root, bundle_path)['heldout_cases_verified'] == 2
    assert bundle_path.exists()
