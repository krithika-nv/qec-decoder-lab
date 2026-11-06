import numpy as np
import pytest

from qec_decoder_lab.literature import compare_scaling_shapes


def test_identical_shape_has_zero_error():
    source = [
        {'distance': d, 'p': 0.001, 'detectors_per_round': d*d-1,
         'microseconds_per_round': y}
        for d, y in ((5, .2), (7, .5), (9, 1.0))
    ]
    rerun = [{**r, 'microseconds_per_round': 3*r['microseconds_per_round']}
             for r in source]
    result = compare_scaling_shapes(source, rerun)
    assert result['normalized_shape_log_rmse'] == pytest.approx(0, abs=1e-12)
    assert result['local_loglog_slope'] == pytest.approx(result['reference_loglog_slope'])


def test_unmatched_paper_protocol_cannot_pass():
    ref = [{'distance': d, 'p': .001, 'detectors_per_round': d*d-1,
            'microseconds_per_round': x} for d, x in ((5, .2), (7, .4))]
    with pytest.raises(ValueError, match='unmatched'):
        compare_scaling_shapes(ref, [{**r, 'p': .02} for r in ref])
