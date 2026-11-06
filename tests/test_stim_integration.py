"""These integration tests run when stim and pymatching are installed."""
import numpy as np
import pytest

stim = pytest.importorskip("stim")
pymatching = pytest.importorskip("pymatching")

from qec_decoder_lab.exact import ExactBayesianDEMDecoder, faults_from_dem
from qec_decoder_lab.experiment import ExperimentSettings, run_experiment
from qec_decoder_lab.model import MemoryConfig


@pytest.mark.parametrize("basis", ["x", "z"])
def test_zero_noise_mwpm_has_no_logical_failures(basis):
    result = run_experiment(ExperimentSettings(
        MemoryConfig(distance=3, rounds=3, basis=basis, p=0),
        shots=100, seed=100, bayesian="off",
    ))
    assert result["mwpm"]["logical_failures"] == 0
    assert result["model"]["num_observables"] == 1


def test_bayes_and_matching_on_identical_surface_code_shots():
    result = run_experiment(ExperimentSettings(
        MemoryConfig(distance=3, rounds=1, p=.01),
        shots=400, seed=42, bayesian="on", max_bayes_bits=16,
    ))
    assert result["bayesian"]["status"] == "completed"
    assert sum(result["paired_comparison"][k] for k in (
        "mwpm_only_wrong", "bayesian_only_wrong", "both_wrong", "both_correct"
    )) == 400


def test_seed_reproducibility():
    settings = ExperimentSettings(MemoryConfig(distance=3, rounds=3, p=.02),
                                  shots=512, seed=2026, bayesian="off")
    a = run_experiment(settings)
    b = run_experiment(settings)
    assert a["mwpm"] == b["mwpm"]


def test_correlated_dem_separator_is_one_fault_mask():
    dem = stim.DetectorErrorModel("error(0.2) D0 ^ D1 L0")
    detectors, obs, faults = faults_from_dem(dem)
    assert (detectors, obs) == (2, 1)
    assert faults[0].mask == (1 | 2 | 4)


def test_bayes_skips_large_models_safely():
    result = run_experiment(ExperimentSettings(
        MemoryConfig(distance=5, rounds=5, p=.01),
        shots=10, seed=111, bayesian="auto",
    ))
    assert result["bayesian"]["status"] == "skipped_state_limit"


@pytest.mark.parametrize("rounds,distance", [(1,3), (3,3), (5,5)])
def test_approximate_bp_uses_identical_stim_shots(rounds, distance):
    result = run_experiment(ExperimentSettings(
        MemoryConfig(distance=distance, rounds=rounds, p=.01),
        shots=60, seed=44, bayesian="auto", approximate="on", bp_iterations=15,
    ))
    assert result["bp_approximate"]["status"] == "completed"
    assert result["bp_approximate"]["shots"] == 60
    assert result["bp_approximate"]["graph_edges"] > 0
    assert len(result["scientific_limits"]) >= 4
    assert sum(result["bp_comparisons"]["vs_mwpm"][k] for k in (
        "mwpm_only_wrong", "bp_only_wrong", "both_wrong", "both_correct"
    )) == 60
    if result["bayesian"]["status"] == "completed":
        assert 0 <= result["bp_comparisons"]["vs_exact"]["logical_prediction_agreement"] <= 1
        assert 0 <= result["bp_comparisons"]["vs_exact"]["mean_absolute_logical_marginal_error"] <= 1
