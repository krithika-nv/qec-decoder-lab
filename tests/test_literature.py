from pathlib import Path

import pytest
from qec_decoder_lab.literature import read_reference_csv


DATA = Path(__file__).resolve().parents[1] / 'data/literature/pymatching_v2_p001.csv'


def test_upstream_v2_series_integrity():
    rows, digest = read_reference_csv(DATA)
    assert len(rows) == 9
    assert rows[0]['microseconds_per_shot'] == pytest.approx(.388141)
    assert rows[-1]['distance'] == 50
    assert rows[-1]['microseconds_per_round'] == pytest.approx(778.247/50)
    assert rows[1]['detectors_per_round'] == 48
    assert len(digest) == 64


def test_complete_transcribed_series_matches_upstream_visible_values():
    values = [0.388141, 1.05477, 2.31456, 6.96616, 16.0062,
              45.1341, 102.495, 310.649, 778.247]
    rows, _ = read_reference_csv(DATA)
    assert [r['microseconds_per_shot'] for r in rows] == values
