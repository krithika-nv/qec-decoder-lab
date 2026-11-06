"""Transparent binomial and matched-shot estimators; no hidden fitting."""
from math import sqrt

import numpy as np


def wilson_interval(errors: int, shots: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Two-sided approximately 95% Wilson score interval (z=1.96 by default)."""
    if shots <= 0 or not 0 <= errors <= shots:
        raise ValueError("require shots > 0 and 0 <= errors <= shots")
    phat = errors / shots
    denom = 1 + z * z / shots
    center = (phat + z * z / (2 * shots)) / denom
    half = z * sqrt((phat * (1 - phat) + z * z / (4 * shots)) / shots) / denom
    return (0.0 if errors == 0 else max(0.0, center - half)), (1.0 if errors == shots else min(1.0, center + half))


def summarize_errors(wrong: np.ndarray) -> dict:
    wrong = np.asarray(wrong, dtype=bool)
    if wrong.ndim != 1 or wrong.size == 0:
        raise ValueError("wrong must be a nonempty 1D array")
    n = int(wrong.size)
    k = int(np.count_nonzero(wrong))
    lo, hi = wilson_interval(k, n)
    return {"shots": n, "logical_failures": k, "logical_failure_rate": k / n,
            "wilson_95_low": lo, "wilson_95_high": hi}


def compare_paired(mwpm_wrong: np.ndarray, bayes_wrong: np.ndarray) -> dict:
    """Paired comparison; no claim of statistical significance from a point estimate."""
    mwpm_wrong = np.asarray(mwpm_wrong, dtype=bool)
    bayes_wrong = np.asarray(bayes_wrong, dtype=bool)
    if mwpm_wrong.ndim != 1 or mwpm_wrong.shape != bayes_wrong.shape or not mwpm_wrong.size:
        raise ValueError("error arrays must have matching nonempty 1D shape")
    n = int(mwpm_wrong.size)
    a = int(np.count_nonzero(mwpm_wrong & ~bayes_wrong))
    b = int(np.count_nonzero(~mwpm_wrong & bayes_wrong))
    delta = (b - a) / n  # Bayesian - MWPM; negative favors Bayesian.
    differences = bayes_wrong.astype(float) - mwpm_wrong.astype(float)
    standard_error = float(np.std(differences, ddof=1) / np.sqrt(n)) if n > 1 else None
    return {
        "mwpm_only_wrong": a,
        "bayesian_only_wrong": b,
        "both_wrong": int(np.count_nonzero(mwpm_wrong & bayes_wrong)),
        "both_correct": int(np.count_nonzero(~mwpm_wrong & ~bayes_wrong)),
        "bayesian_minus_mwpm_rate": delta,
        "paired_difference_standard_error": standard_error,
        "interpretation": "Negative difference favors Bayesian; descriptive, not a significance test",
    }
