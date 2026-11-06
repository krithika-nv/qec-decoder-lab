"""Circuit generation, matched-shot decoder comparison, and provenance."""
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import os
import subprocess
from time import perf_counter
from pathlib import Path

import numpy as np

from .exact import ExactBayesianDEMDecoder
from .belief_propagation import BeliefPropagationDEMDecoder
from .recovery import SyndromeRecoveryDecoder
from .model import MemoryConfig
from .stats import compare_paired, summarize_errors


@dataclass(frozen=True)
class ExperimentSettings:
    config: MemoryConfig
    shots: int = 10000
    seed: int = 20261101
    bayesian: str = "auto"  # auto | on | off
    max_bayes_bits: int = 16
    approximate: str = "off"  # off | on
    bp_iterations: int = 30
    bp_damping: float = 0.3
    bp_tolerance: float = 1e-6
    bp_shots: int | None = None  # cap the matched BP study without reducing MWPM shot count
    recovery: str = "off"  # new experimental decoder, opt-in
    recovery_shots: int | None = None

    def __post_init__(self):
        if self.shots < 1:
            raise ValueError("shots must be > 0")
        if not 0 <= self.seed < (1 << 63):
            raise ValueError("seed must be in range [0, 2^63)")
        if self.bayesian not in ("auto", "on", "off"):
            raise ValueError("bayesian must be auto, on, or off")
        if not 1 <= self.max_bayes_bits <= 20:
            raise ValueError("max_bayes_bits must be in [1, 20] to bound memory")
        if self.approximate not in ("on", "off"):
            raise ValueError("approximate must be on or off")
        if self.bp_shots is not None and self.bp_shots < 1:
            raise ValueError("bp_shots must be positive")
        if self.recovery not in ("off", "on") or (self.recovery_shots is not None and self.recovery_shots < 1):
            raise ValueError("invalid recovery settings")
        if self.bp_iterations < 1 or not 0 <= self.bp_damping < 1 or self.bp_tolerance <= 0:
            raise ValueError("invalid BP iteration, damping, or tolerance")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True,
            check=True, timeout=2, cwd=Path(__file__).resolve().parent,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _named_comparison(first_wrong: np.ndarray, second_wrong: np.ndarray,
                      first_name: str, second_name: str) -> dict:
    # Keep the established paired arithmetic without mislabeling BP as Bayesian MAP.
    old = compare_paired(first_wrong, second_wrong)
    return {
        f"{first_name}_only_wrong": old["mwpm_only_wrong"],
        f"{second_name}_only_wrong": old["bayesian_only_wrong"],
        "both_wrong": old["both_wrong"],
        "both_correct": old["both_correct"],
        f"{second_name}_minus_{first_name}_rate": old["bayesian_minus_mwpm_rate"],
        "paired_difference_standard_error": old["paired_difference_standard_error"],
        "interpretation": f"Negative difference favors {second_name}; descriptive, not a significance test",
    }


def run_experiment(settings: ExperimentSettings, *, shots_path: Path | None = None) -> dict:
    import pymatching
    import stim

    circuit = settings.config.circuit()
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=settings.seed)
    syndromes, truth = sampler.sample(shots=settings.shots, separate_observables=True)
    noiseless = all(v == 0 for v in settings.config.noise_parameters().values())
    if noiseless:
        if np.any(syndromes) or np.any(truth):
            raise RuntimeError("Noiseless circuit produced a detector or observable event")
        mwpm_prediction = np.zeros_like(truth)
        mwpm_mode = "trivial_noiseless_frame"
    else:
        mwpm_prediction = matching.decode_batch(syndromes)
        mwpm_mode = "PyMatching_MWPM"
    if mwpm_prediction.shape != truth.shape:
        raise RuntimeError("MWPM predictions do not match observable array dimensions")
    mwpm_wrong = np.any(mwpm_prediction != truth, axis=1)
    bayesian_summary = {"status": "not_requested"}
    paired = None
    bayes_prediction = None
    exact_decoder = None

    if settings.bayesian != "off":
        bits = dem.num_detectors + dem.num_observables
        if bits > settings.max_bayes_bits:
            if settings.bayesian == "on":
                raise ValueError(
                    f"Bayesian decoder requires 2^{bits} states; max_bits={settings.max_bayes_bits}. "
                    "Try --rounds 1, --distance 3, or --bayesian off."
                )
            bayesian_summary = {
                "status": "skipped_state_limit",
                "requested_bits": bits,
                "max_bits": settings.max_bayes_bits,
            }
        else:
            bayesian = ExactBayesianDEMDecoder.from_dem(dem, max_bits=settings.max_bayes_bits)
            exact_decoder = bayesian
            bayes_prediction = bayesian.decode_batch(syndromes)
            if bayes_prediction.shape != truth.shape:
                raise RuntimeError("Bayesian predictions do not match observable dimensions")
            bayes_wrong = np.any(bayes_prediction != truth, axis=1)
            bayesian_summary = {
                "status": "completed",
                **summarize_errors(bayes_wrong),
                "states": int(1 << bits),
                "reachable_syndrome_patterns": int(np.count_nonzero(bayesian.evidence > 0)),
                "inference_model": "independent_error_mechanisms_in_Stim_DEM",
            }
            paired = compare_paired(mwpm_wrong, bayes_wrong)
    approx_prediction = None
    approx_probabilities = None
    approx_summary = {"status": "not_requested"}
    approximate_comparison = None
    if settings.approximate == "on":
        verbose = os.environ.get('QEC_PROGRESS') == '1'
        if verbose:
            print(f'  MWPM decoded {settings.shots} shots. Preparing Bayesian '
                  f'approximation for {min(settings.shots, settings.bp_shots or settings.shots)} shots.',
                  flush=True)
        bp_n = min(settings.shots, settings.bp_shots or settings.shots)
        bp_input = syndromes[:bp_n]
        bp_truth = truth[:bp_n]
        started = perf_counter()
        bp = BeliefPropagationDEMDecoder.from_dem(
            dem, iterations=settings.bp_iterations, damping=settings.bp_damping,
            tolerance=settings.bp_tolerance,
        )
        # Emit live progress during the slow pure-Python BP loop, not only at
        # the end of an entire Stage 4 experiment case.
        progress = (lambda done, total: print(
            f'  BP unique syndromes decoded: {done}/{total}', flush=True
        )) if verbose else None
        bp_kwargs = {'with_details': True}
        if verbose:
            bp_kwargs.update(progress_callback=progress,
                             progress_every=max(1, bp_n // 10))
        approx_prediction, approx_probabilities, diagnostics = bp.decode_batch(
            bp_input, **bp_kwargs
        )
        decoding_seconds = perf_counter() - started
        if approx_prediction.shape != bp_truth.shape:
            raise RuntimeError("BP predictions do not match observable dimensions")
        bp_wrong = np.any(approx_prediction != bp_truth, axis=1)
        approx_summary = {
            "status": "completed", **summarize_errors(bp_wrong), **diagnostics,
            "matched_shots": bp_n, "selection": "first-N-of-seeded-MWPM-sample",
            "model": "loopy_sum_product_and_factorized_logical_marginals",
            "wall_seconds_including_decoder_construction": float(decoding_seconds),
            "microseconds_per_shot_including_construction": float(decoding_seconds * 1e6 / bp_n),
            "mean_logical_probability_brier": float(
                np.mean((approx_probabilities - bp_truth.astype(float))**2)
            ),
        }
        approximate_comparison = {"vs_mwpm": _named_comparison(mwpm_wrong[:bp_n], bp_wrong, "mwpm", "bp")}
        if exact_decoder is not None:
            indices = np.zeros(bp_n, dtype=np.int64)
            for j in range(exact_decoder.detectors):
                indices |= bp_input[:, j].astype(np.int64) << j
            exact_probabilities = np.empty_like(approx_probabilities)
            for k in range(exact_decoder.observables):
                logical_indices = [i for i in range(1 << exact_decoder.observables) if i & (1 << k)]
                exact_probabilities[:, k] = (
                    exact_decoder.joint[logical_indices][:, indices].sum(axis=0)
                    / exact_decoder.evidence[indices]
                )
            approximate_comparison["vs_exact"] = {
                "logical_prediction_agreement": float(np.mean(
                    np.all(approx_prediction == bayes_prediction[:bp_n], axis=1)
                )),
                "mean_absolute_logical_marginal_error": float(np.mean(
                    np.abs(approx_probabilities - exact_probabilities)
                )),
                "max_absolute_logical_marginal_error": float(np.max(
                    np.abs(approx_probabilities - exact_probabilities)
                )),
                "exact_map_vs_bp": _named_comparison(bayes_wrong[:bp_n], bp_wrong, "exact", "bp"),
            }
    recovery_predictions = None
    recovery_summary = {"status": "not_requested"}
    if settings.recovery == 'on':
        recovery_n = min(settings.shots, settings.recovery_shots or settings.bp_shots or settings.shots)
        if os.environ.get('QEC_PROGRESS') == '1':
            print(f'  Syndrome-consistent fault recovery for {recovery_n} matched shots...', flush=True)
        started = perf_counter()
        decoder = SyndromeRecoveryDecoder.from_dem(
            dem, iterations=settings.bp_iterations, damping=settings.bp_damping
        )
        progress = (lambda done, total: print(f'  Recovery unique syndromes: {done}/{total}', flush=True)) \
            if os.environ.get('QEC_PROGRESS') == '1' else None
        recovery_predictions, recovery_diagnostics = decoder.decode_batch(
            syndromes[:recovery_n], with_details=True,
            progress_callback=progress, progress_every=max(1, recovery_n // 10)
        )
        recovery_wrong = np.any(recovery_predictions != truth[:recovery_n], axis=1)
        recovery_summary = {
            'status': 'completed', **summarize_errors(recovery_wrong), **recovery_diagnostics,
            'matched_shots': recovery_n,
            'method': 'bp_guided_prior_scored_syndrome_consistent_fault_recovery',
            'wall_seconds_including_decoder_construction': perf_counter() - started,
            'scope': 'fault-configuration heuristic; not exact Bayesian MAP or MWPM',
        }
    if shots_path is not None:
        shots_path.parent.mkdir(parents=True, exist_ok=True)
        archive = {
            "detection_events": syndromes,
            "actual_observables": truth,
            "mwpm_predictions": mwpm_prediction,
        }
        if bayes_prediction is not None:
            archive["bayesian_predictions"] = bayes_prediction
        if recovery_predictions is not None:
            archive["recovery_predictions"] = recovery_predictions
        if approx_prediction is not None:
            archive["bp_predictions"] = approx_prediction
            archive["bp_logical_one_probabilities"] = approx_probabilities
        np.savez_compressed(shots_path, **archive)
    return {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "stim": stim.__version__, "pymatching": version("pymatching"),
            "numpy": np.__version__, "git_commit": git_commit(),
            "sampling_source": "Stim circuit detector sampler; identical shots for both decoders",
        },
        "parameters": {
            **settings.config.to_dict(), "shots": settings.shots,
            "seed": settings.seed, "bayesian": settings.bayesian,
            "max_bayes_bits": settings.max_bayes_bits,
            "approximate": settings.approximate,
            "bp_iterations": settings.bp_iterations,
            "bp_damping": settings.bp_damping,
            "bp_tolerance": settings.bp_tolerance, "bp_shots": settings.bp_shots,
            "recovery": settings.recovery, "recovery_shots": settings.recovery_shots,
        },
        "model": {
            "num_qubits": circuit.num_qubits,
            "num_detectors": dem.num_detectors,
            "num_observables": dem.num_observables,
            "rounds_scale_with_distance": settings.config.rounds == settings.config.distance,
            "reference_architecture": "Stim-generated rotated surface-code memory",
            "circuit_sha256": hashlib.sha256(str(circuit).encode("utf8")).hexdigest(),
            "dem_sha256": hashlib.sha256(str(dem).encode("utf8")).hexdigest(),
            "mwpm_mode": mwpm_mode,
        },
        "mwpm": summarize_errors(mwpm_wrong),
        "bayesian": bayesian_summary,
        "paired_comparison": paired,
        "bp_approximate": approx_summary,
        "bp_comparisons": approximate_comparison,
        "recovery": recovery_summary,
        "scientific_limits": [
            "Simulated circuit-level memory experiment; not measured hardware performance",
            "Stim detector error model can approximate correlations in the original circuit",
            "Exact Bayesian inference is exact only for the supplied independent DEM",
            "No surface-code threshold inferred from a small finite-shot sweep",
            "Loopy BP and its factorized logical marginals are heuristic; decoding may be inaccurate",
            "BP hard fault estimates need not satisfy all detection-event parity constraints",
            "Syndrome-consistent recovery selects a plausible fault configuration, not the logical MAP frame",
        ],
    }
