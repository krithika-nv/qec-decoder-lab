"""Scope of paired comparisons when BP is capped for large MWPM studies."""
import sys
from types import SimpleNamespace

import numpy as np

from qec_decoder_lab import experiment


def test_capped_bp_uses_first_n_identical_stim_shots(monkeypatch, tmp_path):
    syndromes = np.zeros((10, 2), dtype=np.uint8)
    truth = np.zeros((10, 1), dtype=np.uint8)
    truth[[0, 7], 0] = 1

    class FakeSampler:
        def sample(self, shots, separate_observables):
            assert shots == 10 and separate_observables
            return syndromes.copy(), truth.copy()

    class FakeCircuit:
        num_qubits = 3
        def detector_error_model(self, decompose_errors):
            assert decompose_errors
            return SimpleNamespace(num_detectors=2, num_observables=1)
        def compile_detector_sampler(self, seed):
            return FakeSampler()
        def __str__(self):
            return 'H 0\nM 0'

    class FakeConfig:
        rounds = 3
        distance = 3
        def circuit(self):
            return FakeCircuit()
        def noise_parameters(self):
            return {'p': .01}
        def to_dict(self):
            return {'rounds': 3, 'distance': 3}

    class FakeMatching:
        @staticmethod
        def from_detector_error_model(dem):
            return FakeMatching()
        def decode_batch(self, rows):
            return np.zeros((len(rows), 1), dtype=np.uint8)

    class FakeBP:
        @staticmethod
        def from_dem(dem, **opts):
            return FakeBP()
        def decode_batch(self, rows, with_details):
            assert len(rows) == 4 and with_details
            return np.zeros((4, 1), np.uint8), np.zeros((4, 1)), {'converged': 1}

    monkeypatch.setitem(sys.modules, 'stim', SimpleNamespace(__version__='MOCK'))
    monkeypatch.setitem(sys.modules, 'pymatching', SimpleNamespace(Matching=FakeMatching))
    monkeypatch.setattr(experiment, 'version', lambda package: 'MOCK')
    monkeypatch.setattr(experiment, 'BeliefPropagationDEMDecoder', FakeBP)
    output = tmp_path/'case.npz'
    result = experiment.run_experiment(
        experiment.ExperimentSettings(config=FakeConfig(), shots=10,
                                      approximate='on', bayesian='off', bp_shots=4),
        shots_path=output,
    )
    assert result['mwpm']['shots'] == 10
    assert result['bp_approximate']['shots'] == 4
    assert result['bp_comparisons']['vs_mwpm']['mwpm_only_wrong'] == 0
    with np.load(output, allow_pickle=False) as archive:
        assert archive['detection_events'].shape == (10, 2)
        assert archive['bp_predictions'].shape == (4, 1)
