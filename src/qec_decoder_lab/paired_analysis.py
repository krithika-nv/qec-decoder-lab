"""Stage 4: interpretable matched-shot statistical decoder comparisons."""
from __future__ import annotations

from math import exp, lgamma, log

import numpy as np


def mcnemar_exact_p(a: int, b: int) -> float:
    """Exact two-sided binomial/McNemar test for discordant pairs.

    a, b: cases where only decoder A or only decoder B makes an error.
    A p-value is not an effect-size estimate; no advantage is asserted by default.
    """
    if min(a, b) < 0 or int(a) != a or int(b) != b:
        raise ValueError("Discordant counts must be nonnegative integers")
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    # logsumexp of binomial probabilities; safe for large n without SciPy.
    logterms = [lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) - n * log(2)
                for i in range(k + 1)]
    vmax = max(logterms)
    return min(1.0, 2 * exp(vmax) * sum(exp(x - vmax) for x in logterms))


def logical_error_per_round(memory_failure_rate: float, rounds: int) -> float:
    """Common independent-round-equivalent mapping, not an observed per-round rate."""
    if not (0 <= memory_failure_rate <= 0.5) or rounds < 1:
        raise ValueError("rate must be in [0,0.5], rounds >= 1")
    if memory_failure_rate == 0.5:
        return 0.5  # continuous limit; avoids log(0) numerical warning
    return float(-np.expm1(np.log1p(-2 * memory_failure_rate) / rounds) / 2)


def analyze_pair(first_wrong: np.ndarray, second_wrong: np.ndarray, *,
                 seed: int = 412, bootstrap_repeats: int = 2000) -> dict:
    a = np.asarray(first_wrong, dtype=bool)
    b = np.asarray(second_wrong, dtype=bool)
    if a.ndim != 1 or a.shape != b.shape or a.size == 0:
        raise ValueError("error arrays must be nonempty and identically shaped")
    if bootstrap_repeats < 10:
        raise ValueError("at least 10 bootstrap repeats required")
    n = len(a)
    only_first = int(np.sum(a & ~b))
    only_second = int(np.sum(~a & b))
    both_wrong = int(np.sum(a & b))
    both_correct = n - only_first - only_second - both_wrong
    # Resample the *paired outcomes*, not the decoders independently.
    rng = np.random.default_rng(seed)
    probs = np.array([only_first, only_second, both_wrong, both_correct], dtype=float) / n
    draws = rng.multinomial(n, probs, size=bootstrap_repeats)
    sample_effects = (draws[:, 1] - draws[:, 0]) / n
    lo, hi = np.quantile(sample_effects, [0.025, 0.975])
    return {
        "shots": n, "first_only_wrong": only_first, "second_only_wrong": only_second,
        "both_wrong": both_wrong, "both_correct": both_correct,
        "first_failure_rate": float(np.mean(a)), "second_failure_rate": float(np.mean(b)),
        "second_minus_first_failure_rate": (only_second - only_first) / n,
        "mcnemar_exact_two_sided_p": mcnemar_exact_p(only_first, only_second),
        "paired_bootstrap_95_percentile_interval": [float(lo), float(hi)],
        "bootstrap_repeats": bootstrap_repeats, "bootstrap_seed": seed,
        "interpretation": "Negative difference favors second decoder. No superiority is inferred from a single test or exploratory sweep. Bootstrap interval is descriptive, especially with rare errors.",
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down FWER correction, independent of test ordering.

    All hypotheses in a single declared family must be submitted together.
    Empty families return an empty list.
    """
    if any(not np.isfinite(p) or p < 0 or p > 1 for p in p_values):
        raise ValueError('p-values must be in [0,1] and finite')
    order = sorted(range(len(p_values)), key=lambda i: p_values[i])
    result = [0.0] * len(p_values)
    maximum = 0.0
    for rank, idx in enumerate(order):
        maximum = max(maximum, min(1.0, (len(p_values) - rank) * p_values[idx]))
        result[idx] = maximum
    return result
