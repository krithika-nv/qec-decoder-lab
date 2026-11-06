"""Stage 3: independently verify archived circuit-level experiment evidence.

Hashes, shots, predictions, bit dimensions, error rates, paired accounting,
configurations and optionally deterministic Stim re-sampling are checked.
Failures are explicit: a JSON report is not treated as scientific evidence just
because the originating command returned zero.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from .model import MemoryConfig
from .stats import summarize_errors


class EvidenceError(ValueError):
    """Raised if an archived result fails a required reproducibility gate."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _binary_array(arr: np.ndarray, *, name: str, n: int, width: int):
    if arr.shape != (n, width) or not np.all((arr == 0) | (arr == 1)):
        raise EvidenceError(f"{name}: expected binary ({n}, {width}) array")


def _summary_matches(reference: dict, predicted: np.ndarray, truth: np.ndarray) -> bool:
    actual = summarize_errors(np.any(predicted != truth, axis=1))
    return all(np.isclose(actual[key], reference[key], rtol=1e-11, atol=1e-12)
               for key in actual)


def verify_case(report_path: str | Path, *, replay: bool = False,
                require_raw_hash: bool = False, redo_mwpm: bool = False) -> dict:
    """Verify case; raises EvidenceError on mismatch. Replay requires Stim.

    We derive paths from the report path rather than trusting serialized paths.
    Replay resamples the circuit with the recorded seed and checks exact bits;
    it does not merely compare aggregate rates.
    """
    path = Path(report_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data["parameters"]
    model = data["model"]
    n, d, o = int(params["shots"]), int(model["num_detectors"]), int(model["num_observables"])
    if n < 1 or o < 1 or d < 0:
        raise EvidenceError("Invalid shot or bit dimensions")
    for suffix, key in ((".stim", "circuit_sha256"), (".dem", "dem_sha256")):
        artifact = path.with_suffix(suffix)
        if not artifact.is_file() or _sha(artifact) != model[key]:
            raise EvidenceError(f"Missing or SHA256-mismatched {artifact}")
    raw_path = path.with_suffix(".npz")
    saved_sha = model.get("raw_shots_sha256")
    if require_raw_hash and not saved_sha:
        raise EvidenceError("Missing mandatory Stage 4 raw archive SHA256")
    if saved_sha and _sha(raw_path) != saved_sha:
        raise EvidenceError("Raw shot archive SHA256 mismatch")
    with np.load(raw_path, allow_pickle=False) as archive:
        required = ("detection_events", "actual_observables", "mwpm_predictions")
        if any(key not in archive.files for key in required):
            raise EvidenceError("Missing raw evidence arrays")
        syndromes, truth, mwpm = (archive[key] for key in required)
        _binary_array(syndromes, name="syndromes", n=n, width=d)
        _binary_array(truth, name="actual_observables", n=n, width=o)
        _binary_array(mwpm, name="mwpm_predictions", n=n, width=o)
        if not _summary_matches(data["mwpm"], mwpm, truth):
            raise EvidenceError("MWPM summary does not match archived shots")
        checked = ["circuit_sha256", "dem_sha256", "raw_shots", "mwpm_failure_accounting"]
        if saved_sha:
            checked.append("raw_shots_sha256")
        if redo_mwpm:
            import pymatching
            import stim
            matching = pymatching.Matching.from_detector_error_model(
                stim.DetectorErrorModel.from_file(str(path.with_suffix(".dem")))
            )
            independent = matching.decode_batch(syndromes)
            if not np.array_equal(independent, mwpm):
                raise EvidenceError("Archived MWPM predictions differ from independently decoded syndromes")
            checked.append("independent_mwpm_redecode")
        if data["bayesian"]["status"] == "completed":
            if "bayesian_predictions" not in archive:
                raise EvidenceError("Missing exact Bayesian prediction array")
            exact = archive["bayesian_predictions"]
            _binary_array(exact, name="bayesian", n=n, width=o)
            if not _summary_matches(data["bayesian"], exact, truth):
                raise EvidenceError("Exact Bayesian summary differs from shot archive")
            checked.append("exact_bayesian_failure_accounting")
        if data["bp_approximate"]["status"] == "completed":
            if not {"bp_predictions", "bp_logical_one_probabilities"}.issubset(archive.files):
                raise EvidenceError("Missing BP arrays")
            approx = archive["bp_predictions"]
            probs = archive["bp_logical_one_probabilities"]
            bp_n = int(data["bp_approximate"].get("matched_shots", n))
            if not 1 <= bp_n <= n:
                raise EvidenceError("Invalid BP matched sample size")
            _binary_array(approx, name="bp", n=bp_n, width=o)
            if probs.shape != (bp_n, o) or not np.all(np.isfinite(probs)) or np.any(probs < 0) or np.any(probs > 1):
                raise EvidenceError("BP probability array invalid")
            if not _summary_matches(data["bp_approximate"], approx, truth[:bp_n]):
                raise EvidenceError("BP failure summary differs from shots")
            comparison = data["bp_comparisons"]["vs_mwpm"]
            mw = np.any(mwpm[:bp_n] != truth[:bp_n], axis=1)
            bp = np.any(approx != truth[:bp_n], axis=1)
            derived = {
                "mwpm_only_wrong": int(np.sum(mw & ~bp)),
                "bp_only_wrong": int(np.sum(~mw & bp)),
                "both_wrong": int(np.sum(mw & bp)),
                "both_correct": int(np.sum(~mw & ~bp)),
            }
            if any(comparison[k] != v for k, v in derived.items()):
                raise EvidenceError("Paired MWPM/BP count mismatch")
            checked.extend(["bp_failure_accounting", "bp_probabilities", "matched_shot_accounting"])
        if data.get('recovery', {}).get('status') == 'completed':
            if 'recovery_predictions' not in archive.files:
                raise EvidenceError('Missing recovery prediction array')
            recovery = archive['recovery_predictions']
            recovery_n = int(data['recovery']['matched_shots'])
            if not 1 <= recovery_n <= n:
                raise EvidenceError('Invalid recovery matched sample')
            _binary_array(recovery, name='recovery', n=recovery_n, width=o)
            if not _summary_matches(data['recovery'], recovery, truth[:recovery_n]):
                raise EvidenceError('Recovery failure summary disagrees with archived truth')
            checked.append('recovery_failure_accounting')
        if all(v == 0 for v in params["effective_noise_probabilities"].values()):
            if np.any(syndromes) or np.any(truth) or np.any(mwpm):
                raise EvidenceError("Noiseless model contains nonzero events")
            checked.append("noiseless_invariant")
        if replay:
            config_keys = (
                "distance", "rounds", "basis", "p", "noise_profile",
                "after_clifford_depolarization", "before_round_data_depolarization",
                "before_measure_flip_probability", "after_reset_flip_probability",
            )
            config = MemoryConfig(**{key: params[key] for key in config_keys})
            circuit = config.circuit()
            if hashlib.sha256(str(circuit).encode("utf8")).hexdigest() != model["circuit_sha256"]:
                raise EvidenceError("Regenerated circuit hash mismatch")
            dem = circuit.detector_error_model(decompose_errors=True)
            if hashlib.sha256(str(dem).encode("utf8")).hexdigest() != model["dem_sha256"]:
                raise EvidenceError("Regenerated DEM hash mismatch")
            rerun_d, rerun_o = circuit.compile_detector_sampler(seed=int(params["seed"])).sample(
                shots=n, separate_observables=True,
            )
            if not np.array_equal(rerun_d, syndromes) or not np.array_equal(rerun_o, truth):
                raise EvidenceError("Seeded replay did not reproduce the archived bit arrays")
            checked.append("seeded_bitwise_replay")
    return {"status": "PASS", "report": str(path), "gates": checked,
            "replay": "PASS" if replay else "NOT_REQUESTED", "shots": n}
