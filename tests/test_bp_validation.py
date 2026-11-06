import pytest
from qec_decoder_lab.validate_bp import validation_report


def test_synthetic_bp_report_all_syndromes():
    result = validation_report()
    tree = result['metrics']['tree_single_check']
    assert tree['num_syndromes_with_evidence'] == 2
    assert tree['weighted_mean_absolute_posterior_error'] < 1e-8
    assert tree['weighted_exact_map_agreement'] == pytest.approx(1.)
    loop = result['metrics']['loopy_three_check']
    assert loop['num_syndromes_with_evidence'] == 8
    assert 0 <= loop['weighted_exact_map_agreement'] <= 1
    assert 0 <= loop['weighted_mean_absolute_posterior_error'] <= 1
    assert loop['exact_evidence_normalized']
