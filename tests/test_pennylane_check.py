import pytest

pytest.importorskip('pennylane')
pytest.importorskip('stim')

from qec_decoder_lab.pennylane_check import validate_zz_check


def test_independent_parity_check_full_basis():
    result = validate_zz_check(p=.12, shots=3000, seed=17)
    assert result['status'] == 'passed'
    assert len(result['cases']) == 4
