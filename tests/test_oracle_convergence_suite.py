import numpy as np
import pytest

from qec_decoder_lab.exact import Fault
from qec_decoder_lab.exact_oracle_suite import evaluate_small_model, run as run_oracle
from qec_decoder_lab.convergence_study import evaluate_variants


def test_exact_oracle_expected_risk_is_optimal():
    cases = [Fault(.11, 1), Fault(.15, 2), Fault(.02, 1 | 4), Fault(.04, 1 | 2 | 4)]
    record = evaluate_small_model(2, 1, cases)
    exact = record['exact_logical_MAP']['exact_expected_logical_failure']
    assert exact <= record['original_loopy_bp']['exact_expected_logical_failure'] + 1e-10
    assert exact <= record['syndrome_recovery']['exact_expected_logical_failure'] + 1e-10
    assert record['syndrome_recovery']['excess_expected_failure_over_oracle'] >= 0


def test_oracle_suite_is_seed_reproducible(tmp_path):
    x = run_oracle(tmp_path/'a', seed=37, count=6, iterations=7, damping=.3)
    y = run_oracle(tmp_path/'b', seed=37, count=6, iterations=7, damping=.3)
    assert x['cases'] == y['cases']
    assert len(x['cases']) == 10
    assert (tmp_path/'a'/'exact_oracle_summary.csv').is_file()


def test_exact_oracle_rejects_large_state():
    with pytest.raises(ValueError, match='12'):
        evaluate_small_model(12, 1, [Fault(.1,1)])


def test_sensitivity_assesses_recovery_and_bp_on_same_shots():
    faults = [Fault(.1,1),Fault(.2,2),Fault(.08,1|2|4)]
    syn = np.array([[0,0],[1,0],[0,1],[1,1]], dtype=np.uint8)
    truth = np.zeros((4,1),dtype=np.uint8)
    rows = evaluate_variants(2,1,faults,syn,truth,dampings=(0.,.4),iterations=(4,8))
    assert len(rows) == 4
    assert all(0 <= x['fraction_converged'] <= 1 for x in rows)
    assert all(x['fraction_recovery_syndrome_consistent'] == 1 for x in rows)
    assert all('recovery_logical_errors' in x for x in rows)
