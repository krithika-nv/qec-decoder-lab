"""Experimental sum-product BP decoder for independent Stim DEM mechanisms.

The Tanner graph has one binary variable per independent DEM ``error`` term and
one hard parity-check factor per detector.  Sum-product in the log-likelihood
ratio (LLR) domain approximates P(fault_j=1 | detectors). The final logical
marginal is computed by assuming the *posterior* fault marginals are independent:

    P(L_k=1 | D) ~= (1 - prod_{j: L_k in fault_j}(1-2 q_j)) / 2.

That second factorization is not exact, even on a tree where BP fault marginals
are exact. No claim of syndrome-consistent fault recovery, exact logical MAP,
or competitive logical error rates is made. Unlike the 2**detector exact decoder,
space scales linearly with DEM graph edges. Each *shot* still costs O(edges *
iterations), and loopy BP may fail to converge or give poor marginals.
"""
from dataclasses import dataclass

import numpy as np

from .exact import Fault, faults_from_dem


@dataclass(frozen=True)
class BPResult:
    logical_one_probabilities: np.ndarray
    prediction: np.ndarray
    iterations: int
    converged: bool
    hard_fault_syndrome_matches: bool
    fault_marginals: np.ndarray


class BeliefPropagationDEMDecoder:
    def __init__(
        self, detectors: int, observables: int, faults: list[Fault], *,
        iterations: int = 30, damping: float = 0.3, tolerance: float = 1e-6,
        max_edges: int = 300000, max_faults: int = 300000,
        check_update: str = "legacy",
    ):
        if check_update not in ("legacy", "vectorized", "compiled", "minsum"):
            raise ValueError("check_update must be legacy, vectorized, compiled, or minsum")
        self.check_update = check_update
        if detectors < 0 or observables < 1:
            raise ValueError('require detectors >= 0 and observables >= 1')
        if (iterations < 1 or not 0 <= damping < 1 or tolerance <= 0 or
                max_edges < 1 or max_faults < 1):
            raise ValueError('invalid BP iteration, damping, tolerance or graph caps')
        if len(faults) > max_faults:
            raise ValueError('DEM fault count exceeds BP max_faults safety limit')
        self.detectors = detectors
        self.observables = observables
        self.iterations = iterations
        self.damping = damping
        self.tolerance = tolerance
        self.faults = tuple(faults)
        self.probabilities = np.array([f.probability for f in faults], dtype=float)
        if not np.all(np.isfinite(self.probabilities)) or np.any(self.probabilities < 0) or np.any(self.probabilities > 1):
            raise ValueError('fault probabilities must be in [0, 1]')
        if any(f.mask < 0 or f.mask.bit_length() > detectors + observables for f in faults):
            raise ValueError('invalid fault mask')
        eps = 1e-13
        clipped = np.clip(self.probabilities, eps, 1 - eps)
        self.prior_llr = np.log1p(-clipped) - np.log(clipped)
        self.det_masks = [f.mask & ((1 << detectors) - 1) for f in faults]
        self.logical_faults = [
            np.array([j for j, f in enumerate(faults) if (f.mask >> (detectors + k)) & 1], dtype=np.int64)
            for k in range(observables)
        ]
        edges = []
        for j, mask in enumerate(self.det_masks):
            while mask:
                lowest = mask & -mask
                edges.append((lowest.bit_length() - 1, j))
                mask ^= lowest
                if len(edges) > max_edges:
                    raise ValueError('DEM Tanner graph exceeds BP max_edges safety limit')
        edges.sort()
        self.edge_vars = np.array([j for _, j in edges], dtype=np.int64)
        row_degrees = np.bincount([i for i, _ in edges], minlength=detectors)
        self.indptr = np.concatenate([[0], np.cumsum(row_degrees)])
        self.edges = len(edges)
        # One-time Tanner graph layout, reused for every BP iteration and shot.
        self.edge_rows = np.repeat(np.arange(detectors, dtype=np.int64), row_degrees)
        self._nonzero_degree_rows = row_degrees > 0

    def _vectorized_check_messages(self, variable_to_check: np.ndarray,
                                   syndrome: np.ndarray) -> np.ndarray:
        """Leave-one-out parity messages computed in O(edges + detectors).

        Zero tanh messages need special handling: division by a zero incoming
        message is invalid, while the outgoing message at the unique zero edge
        may be nonzero. Log products and zero counts implement the exact
        exclusion rule (modulo floating-point roundoff). For an empty graph
        this returns an empty message vector.
        """
        if not self.edges:
            return np.empty(0, dtype=float)
        t = np.tanh(variable_to_check / 2.0)
        mag = np.abs(t)
        zero = mag == 0
        safe_log = np.zeros_like(mag)
        np.log(mag, out=safe_log, where=~zero)
        by_row_log = np.bincount(self.edge_rows, weights=safe_log, minlength=self.detectors)
        by_row_zeros = np.bincount(self.edge_rows, weights=zero.astype(np.int8), minlength=self.detectors)
        negative = t < 0
        by_row_negative = np.bincount(self.edge_rows, weights=negative.astype(np.int8), minlength=self.detectors)
        zero_count = by_row_zeros[self.edge_rows]
        log_except = by_row_log[self.edge_rows] - safe_log
        # Exactly one zero: only its own outgoing check message survives.
        usable = (zero_count == 0) | ((zero_count == 1) & zero)
        product_magnitude = np.zeros(self.edges, dtype=float)
        np.exp(log_except, out=product_magnitude, where=usable)
        is_negative = ((by_row_negative[self.edge_rows] - negative.astype(np.int8)) % 2 != 0)
        sign = np.where(is_negative, -1., 1.) * np.where(syndrome[self.edge_rows], -1., 1.)
        signed_product = np.clip(sign * product_magnitude, -1 + 1e-12, 1 - 1e-12)
        return 2 * np.arctanh(signed_product)

    @classmethod
    def from_dem(cls, dem, **kwargs) -> 'BeliefPropagationDEMDecoder':
        return cls(*faults_from_dem(dem), **kwargs)

    def decode(self, syndrome: np.ndarray) -> BPResult:
        y = np.asarray(syndrome)
        if y.shape != (self.detectors,) or np.any((y != 0) & (y != 1)):
            raise ValueError(f'expected binary syndrome of shape ({self.detectors},)')
        if any(self.indptr[i] == self.indptr[i+1] and y[i] for i in range(self.detectors)):
            raise ValueError('syndrome impossible: event on detector with no mechanisms')

        if self.check_update == "minsum":
            from .compiled_bp import minsum_marginals
            q, used, converged = minsum_marginals(
                self.prior_llr, self.edge_vars, self.indptr, y,
                self.iterations, self.damping, self.tolerance)
        elif self.check_update == "compiled":
            # Numba is imported only for the explicitly requested optional
            # engine. The first call compiles; warm before benchmarking.
            from .compiled_bp import compiled_marginals
            q, used, converged = compiled_marginals(
                self.prior_llr, self.edge_vars, self.indptr, y,
                self.iterations, self.damping, self.tolerance)
        else:
            q, used, converged = self._decode_numpy_messages(y)
        logical_p = np.array([
            (1.0 - np.prod(1.0 - 2.0 * q[ids])) / 2.0
            for ids in self.logical_faults
        ])
        prediction = (logical_p > 0.5).astype(np.uint8)
        hard_syndrome = 0
        for j in np.flatnonzero(q > 0.5):
            hard_syndrome ^= self.det_masks[j]
        actual_syndrome = sum(int(bit) << i for i, bit in enumerate(y))
        return BPResult(
            logical_one_probabilities=logical_p, prediction=prediction,
            iterations=used, converged=converged,
            hard_fault_syndrome_matches=hard_syndrome == actual_syndrome,
            fault_marginals=q.copy(),
        )

    def _decode_numpy_messages(self, y: np.ndarray):
        variable_to_check = self.prior_llr[self.edge_vars].copy()
        check_to_variable = np.zeros(self.edges, dtype=float)
        converged = False
        used = 0
        for iteration in range(1, self.iterations + 1):
            used = iteration
            # The check-node update uses leave-one-out products of tanh(LLR/2).
            # Avoids dividing by zero for uncertain messages.
            if self.check_update == "vectorized":
                check_to_variable = self._vectorized_check_messages(variable_to_check, y)
            else:
                for i in range(self.detectors):
                    start, end = int(self.indptr[i]), int(self.indptr[i + 1])
                    if start == end:
                        continue
                    t = np.tanh(variable_to_check[start:end] / 2)
                    left = np.concatenate([[1.0], np.cumprod(t[:-1])])
                    right = np.concatenate([np.cumprod(t[:0:-1])[::-1], [1.0]])
                    sign = 1 if not y[i] else -1
                    prod = np.clip(sign * left * right, -1 + 1e-12, 1 - 1e-12)
                    check_to_variable[start:end] = 2 * np.arctanh(prod)
            total = np.bincount(
                self.edge_vars, weights=check_to_variable, minlength=len(self.faults)
            )
            candidate = self.prior_llr[self.edge_vars] + total[self.edge_vars] - check_to_variable
            updated = self.damping * variable_to_check + (1 - self.damping) * candidate
            change = float(np.max(np.abs(updated - variable_to_check))) if self.edges else 0.0
            variable_to_check = np.clip(updated, -60.0, 60.0)
            if change < self.tolerance:
                converged = True
                break

        final_llr = self.prior_llr + np.bincount(
            self.edge_vars, weights=check_to_variable, minlength=len(self.faults)
        )
        q = 1.0 / (1.0 + np.exp(np.clip(final_llr, -60, 60)))
        return q, used, converged

    def fault_marginals(self, syndrome: np.ndarray) -> np.ndarray:
        """Approximate fault posteriors; not a joint posterior or fault recovery."""
        return self.decode(syndrome).fault_marginals

    def decode_batch(self, syndromes: np.ndarray, *, with_details: bool = False,
                     progress_callback=None, progress_every: int = 0):
        y = np.asarray(syndromes)
        if y.ndim != 2 or y.shape[1] != self.detectors or np.any((y != 0) & (y != 1)):
            raise ValueError(f'expected binary array with shape (shots, {self.detectors})')
        unique, inv = np.unique(y, axis=0, return_inverse=True)
        results = []
        total = len(unique)
        for index, row in enumerate(unique, start=1):
            results.append(self.decode(row))
            if progress_callback is not None and progress_every > 0 and (
                index % progress_every == 0 or index == total
            ):
                progress_callback(index, total)
        prediction = np.array([r.prediction for r in results], dtype=np.uint8)[inv]
        if not with_details:
            return prediction
        probabilities = np.array([r.logical_one_probabilities for r in results])[inv]
        diagnostics = {
            'unique_syndromes': int(len(unique)),
            'mean_iterations': float(np.mean(np.array([r.iterations for r in results])[inv])),
            'fraction_converged': float(np.mean(np.array([r.converged for r in results])[inv])),
            'fraction_hard_fault_syndrome_consistent': float(
                np.mean(np.array([r.hard_fault_syndrome_matches for r in results])[inv])
            ),
            'graph_edges': int(self.edges),
            'fault_mechanisms': int(len(self.faults)),
        }
        return prediction, probabilities, diagnostics
