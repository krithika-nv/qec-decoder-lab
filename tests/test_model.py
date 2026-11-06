import pytest

from qec_decoder_lab.model import MemoryConfig


def test_default_distance_rounds_and_noise():
    c = MemoryConfig()
    assert c.rounds == c.distance
    assert all(v == c.p for v in c.noise_parameters().values())


def test_noise_profiles_are_distinct():
    assert MemoryConfig(noise_profile="data_only").noise_parameters()["before_measure_flip_probability"] == 0
    assert MemoryConfig(noise_profile="measurement_only").noise_parameters()["before_round_data_depolarization"] == 0


def test_custom_requires_complete_explicit_probabilities():
    with pytest.raises(ValueError, match="requires"):
        MemoryConfig(noise_profile="custom")


def test_rejects_even_distance_and_invalid_p():
    with pytest.raises(ValueError):
        MemoryConfig(distance=4)
    with pytest.raises(ValueError):
        MemoryConfig(p=1.5)
