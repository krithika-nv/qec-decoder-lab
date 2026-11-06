from pathlib import Path


def test_dedicated_workflow_is_not_blocked_on_pennylane():
    root = Path(__file__).resolve().parents[1]
    workflow = (root / '.github/workflows/pymatching-timing.yml').read_text()
    assert 'RUN_STAGE5_PAPER.sh' in workflow
    assert 'needs: [stage3]' not in workflow
    assert 'stim==1.16.0' in workflow
    assert 'pymatching==2.4.0' in workflow
    assert 'upload-artifact' in workflow


def test_runner_accepts_custom_distances_and_shot_count():
    root = Path(__file__).resolve().parents[1]
    script = (root / 'scripts/shell/RUN_STAGE5_PAPER.sh').read_text()
    assert 'DISTANCES' in script
    assert 'SHOTS' in script
    assert 'REPEATS' in script
    assert '--distances "${DISTANCE_ARRAY[@]}"' in script


def test_benchmark_writes_new_local_csv_and_evidence_with_fake_quantum_modules(tmp_path, monkeypatch):
    """An I/O integration test, not a scientific timing benchmark."""
    import csv
    import json
    import sys
    from types import SimpleNamespace
    import numpy as np
    import qec_decoder_lab.literature as literature

    class FakeSampler:
        def sample(self, shots, separate_observables):
            assert separate_observables
            return np.zeros((shots, 2), dtype=np.uint8), np.zeros((shots, 1), dtype=np.uint8)

    class FakeCircuit:
        def detector_error_model(self, **kwargs):
            assert kwargs == {'decompose_errors': True}
            return 'error(0.001) D0'

        def compile_detector_sampler(self, seed):
            return FakeSampler()

        def __str__(self):
            return 'H 0\nM 0'

    class FakeMatching:
        @classmethod
        def from_detector_error_model(cls, dem):
            return cls()

        def decode_batch(self, syndromes):
            return np.zeros((len(syndromes), 1), dtype=np.uint8)

    def make_circuit(kind, **kwargs):
        assert kind == 'surface_code:rotated_memory_x'
        assert kwargs['rounds'] == kwargs['distance']
        assert set(kwargs.values()) >= {0.001}
        return FakeCircuit()

    monkeypatch.setitem(sys.modules, 'stim', SimpleNamespace(
        Circuit=SimpleNamespace(generated=make_circuit), __version__='FAKE'))
    monkeypatch.setitem(sys.modules, 'pymatching', SimpleNamespace(
        Matching=FakeMatching))
    monkeypatch.setattr(literature, 'version', lambda _: 'FAKE')
    reference = Path(__file__).resolve().parents[1] / 'data/literature/pymatching_v2_p001.csv'
    output = tmp_path / 'timing'
    result = literature.run_matched_benchmark(reference, output,
                                               distances=(5, 7), shots=10000, repeats=2)
    rows = list(csv.DictReader((output / 'timing_summary.csv').open()))
    assert len(rows) == 2
    assert all(float(r['local_us_per_shot_median']) > 0 for r in rows)
    assert result['host']['logical_cpus'] >= 1
    assert result['status'] == 'protocol_matched_cross_machine_comparison_not_exact_timing_reproduction'
    assert (output / 'reference_vs_local.png').exists()
    assert json.loads((output / 'evidence.json').read_text())['versions']['stim'] == 'FAKE'
