"""Optional Numba JIT implementation of the *same* sum-product BP equations.

`numba` is intentionally optional: some supported Python releases do not have a
compatible wheel. Compilation must be warmed up before latency measurements.
This engine does not fix poor convergence or improve posterior correctness.
"""
from __future__ import annotations

import numpy as np

try:
    import numba
except ImportError as exc:
    raise ImportError('compiled BP requires a compatible optional numba installation: pip install numba') from exc


@numba.njit(cache=True)
def _compute(prior: np.ndarray, edge_vars: np.ndarray, indptr: np.ndarray,
             syndrome: np.ndarray, iterations: int, damping: float, tolerance: float):
    nfaults = len(prior)
    nedges = len(edge_vars)
    v2c = np.empty(nedges, dtype=np.float64)
    c2v = np.zeros(nedges, dtype=np.float64)
    for e in range(nedges):
        v2c[e] = prior[edge_vars[e]]
    converged = False
    used = 0
    t = np.empty(nedges, dtype=np.float64)
    prefix = np.empty(nedges, dtype=np.float64)
    total = np.empty(nfaults, dtype=np.float64)
    for iteration in range(1, iterations + 1):
        used = iteration
        # Prefix/suffix parity products, O(edges) and zero-safe by construction.
        # This is algebraically the same sum-product check-node update, but
        # avoids transcendental log/exp operations for each individual edge.
        for row in range(len(syndrome)):
            start = indptr[row]
            end = indptr[row + 1]
            running = 1.0
            for e in range(start, end):
                value = np.tanh(v2c[e] / 2.0)
                t[e] = value
                prefix[e] = running
                running *= value
            suffix = 1.0
            for e in range(end - 1, start - 1, -1):
                prod = prefix[e] * suffix
                if syndrome[row]:
                    prod = -prod
                if prod > 1.0 - 1e-12:
                    prod = 1.0 - 1e-12
                if prod < -1.0 + 1e-12:
                    prod = -1.0 + 1e-12
                c2v[e] = 2.0 * np.arctanh(prod)
                suffix *= t[e]
        total[:] = 0.0
        for e in range(nedges):
            total[edge_vars[e]] += c2v[e]
        change = 0.0
        for e in range(nedges):
            candidate = prior[edge_vars[e]] + total[edge_vars[e]] - c2v[e]
            updated = damping * v2c[e] + (1.0 - damping) * candidate
            delta = abs(updated - v2c[e])
            if delta > change:
                change = delta
            if updated < -60.0:
                updated = -60.0
            if updated > 60.0:
                updated = 60.0
            v2c[e] = updated
        if change < tolerance:
            converged = True
            break
    total[:] = 0.0
    for e in range(nedges):
        total[edge_vars[e]] += c2v[e]
    marginal = np.empty(nfaults, dtype=np.float64)
    for j in range(nfaults):
        value = prior[j] + total[j]
        if value < -60.0:
            value = -60.0
        if value > 60.0:
            value = 60.0
        marginal[j] = 1.0 / (1.0 + np.exp(value))
    return marginal, used, converged


def compiled_marginals(prior_llr: np.ndarray, edge_vars: np.ndarray,
                       indptr: np.ndarray, syndrome: np.ndarray,
                       iterations: int, damping: float, tolerance: float):
    return _compute(prior_llr, edge_vars, indptr, syndrome.astype(np.uint8),
                    iterations, damping, tolerance)

@numba.njit(cache=True)
def _minsum(prior: np.ndarray, edge_vars: np.ndarray, indptr: np.ndarray,
            syndrome: np.ndarray, iterations: int, damping: float, tolerance: float,
            normalization: float):
    """EXPERIMENTAL normalized min-sum approximation (not sum-product BP)."""
    nfaults = len(prior)
    nedges = len(edge_vars)
    v2c = np.empty(nedges, dtype=np.float64)
    c2v = np.zeros(nedges, dtype=np.float64)
    total = np.empty(nfaults, dtype=np.float64)
    for e in range(nedges):
        v2c[e] = prior[edge_vars[e]]
    used = 0
    converged = False
    for iteration in range(1, iterations+1):
        used=iteration
        for row in range(len(syndrome)):
            start=indptr[row]; end=indptr[row+1]
            if start == end:
                continue
            min1=1e300; min2=1e300; min_id=-1
            neg=int(syndrome[row])
            for e in range(start,end):
                v=v2c[e]; mag=abs(v)
                if v < 0:
                    neg ^= 1
                if mag < min1:
                    min2=min1; min1=mag; min_id=e
                elif mag < min2:
                    min2=mag
            for e in range(start,end):
                # degree-one parity check is deterministic (infinite LLR),
                # capped to match the sum-product implementation.
                if end-start==1:
                    magnitude=28.324168296488494
                else:
                    magnitude=min2 if e==min_id else min1
                    magnitude *= normalization
                sign=neg ^ int(v2c[e] < 0)
                c2v[e] = -magnitude if sign else magnitude
        total[:] = 0.0
        for e in range(nedges):
            total[edge_vars[e]] += c2v[e]
        change=0.0
        for e in range(nedges):
            v = prior[edge_vars[e]] + total[edge_vars[e]] - c2v[e]
            updated = damping*v2c[e] + (1-damping)*v
            diff=abs(updated-v2c[e])
            if diff>change: change=diff
            v2c[e] = min(60.0, max(-60.0, updated))
        if change < tolerance:
            converged = True
            break
    total[:] = 0.0
    for e in range(nedges):
        total[edge_vars[e]] += c2v[e]
    q = np.empty(nfaults, dtype=np.float64)
    for j in range(nfaults):
        x = max(-60.0, min(60.0, prior[j]+total[j]))
        q[j] = 1.0/(1.0+np.exp(x))
    return q, used, converged


def minsum_marginals(prior_llr: np.ndarray, edge_vars: np.ndarray,
                     indptr: np.ndarray, syndrome: np.ndarray,
                     iterations: int, damping: float, tolerance: float,
                     normalization: float = 0.8):
    return _minsum(prior_llr, edge_vars, indptr, syndrome.astype(np.uint8),
                   iterations, damping, tolerance, normalization)
