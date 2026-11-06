"""Experimental syndrome-consistent fault recovery on an independent-error DEM.

This is not exact Bayesian logical MAP, not a fault-tolerant decoder, and does not
claim superiority over MWPM. A BP posterior guides a binary elimination basis;
trial recoveries are scored with the independent fault *prior*. The output is a
syndrome-consistent fault configuration whose logical parity is then computed
without factorizing correlated posterior logical marginals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .belief_propagation import BeliefPropagationDEMDecoder
from .exact import Fault, faults_from_dem


@dataclass(frozen=True)
class RecoveryResult:
    prediction: np.ndarray
    syndrome_consistent: bool
    prior_score: float
    bp_converged: bool
    bp_iterations: int
    candidate_method: str


def _bits(x: int):
    x = int(x)
    while x:
        bit = x & -x
        yield bit.bit_length() - 1
        x ^= bit


def _basis(det_masks: list[int], order: np.ndarray, *, target_rank: int | None = None) -> dict[int, tuple[int, int]]:
    """Binary Gaussian elimination; values carry their original-fault XOR masks."""
    pivots: dict[int, tuple[int, int]] = {}
    for original_id in order:
        j = int(original_id)
        v = int(det_masks[j])
        represent = 1 << j
        while v:
            bit = (v & -v).bit_length() - 1
            if bit not in pivots:
                pivots[bit] = (v, represent)
                break
            basis_vector, basis_faults = pivots[bit]
            v ^= basis_vector
            represent ^= basis_faults
        # Once the full known model rank is reached, remaining columns cannot
        # add an independent pivot. Avoid scanning the entire fault list.
        if target_rank is not None and len(pivots) == target_rank:
            break
    return pivots


def _solve(pivots: dict[int, tuple[int, int]], syndrome: int) -> int:
    solution = 0
    remainder = syndrome
    while remainder:
        bit = (remainder & -remainder).bit_length() - 1
        if bit not in pivots:
            raise ValueError('syndrome cannot be represented by the fault model')
        vector, fault_ids = pivots[bit]
        remainder ^= vector
        solution ^= fault_ids
    return solution


def _score(frame: int, weights: np.ndarray) -> float:
    return float(sum(float(weights[j]) for j in _bits(frame)))


def _refine(frame: int, pivots: dict[int, tuple[int, int]], det_masks: list[int],
            weights: np.ndarray, order: np.ndarray, budget: int) -> int:
    """Local nullspace moves preserve the syndrome and improve prior likelihood."""
    best_score = _score(frame, weights)
    for original_id in order[:budget]:
        j = int(original_id)
        try:
            cycle = (1 << j) ^ _solve(pivots, int(det_masks[j]))
        except ValueError:
            continue
        if cycle == 0:
            continue
        candidate = frame ^ cycle
        score = _score(candidate, weights)
        if score + 1e-11 < best_score:
            frame, best_score = candidate, score
    return frame


class SyndromeRecoveryDecoder:
    """BP-guided prior-scored fault recovery with an independently derived fallback.

    The BP-guided recovery and prior-only recovery are *both* solved directly
    from the detector model. This does not invoke MWPM. The lower-prior-energy
    result is selected; a lower prior energy does not imply logical MAP.
    """

    def __init__(self, detectors: int, observables: int, faults: list[Fault], *,
                 iterations: int = 20, damping: float = 0.3,
                 refinement_budget: int = 64, check_update: str = "vectorized",
                 basis_update: str = "python"):
        if refinement_budget < 0:
            raise ValueError('refinement_budget must be nonnegative')
        if basis_update not in ('python', 'compiled'):
            raise ValueError('basis_update must be python or compiled')
        self.basis_update = basis_update
        self.bp = BeliefPropagationDEMDecoder(detectors, observables, faults,
                                              iterations=iterations, damping=damping,
                                              check_update=check_update)
        self.detectors = detectors
        self.observables = observables
        self.det_masks = self.bp.det_masks
        self.logical_masks = [int(f.mask >> detectors) for f in faults]
        self.prior_weights = self.bp.prior_llr
        self.refinement_budget = refinement_budget
        self._prior_order = np.argsort(self.prior_weights, kind='stable')
        self._prior_pivots = _basis(self.det_masks, self._prior_order)
        if basis_update == 'compiled':
            from .compiled_basis import pack_detector_masks
            self._packed_masks = pack_detector_masks(self.det_masks, detectors)

    @classmethod
    def from_dem(cls, dem, **kwargs) -> 'SyndromeRecoveryDecoder':
        return cls(*faults_from_dem(dem), **kwargs)

    def decode(self, syndrome: np.ndarray) -> RecoveryResult:
        y = np.asarray(syndrome)
        if y.shape != (self.detectors,) or np.any((y != 0) & (y != 1)):
            raise ValueError('expected binary detector syndrome')
        target = sum(int(b) << i for i, b in enumerate(y))
        # Run BP for guided reliabilities and hard decision, as in the baseline.
        bp_result = self.bp.decode(y)
        q = bp_result.fault_marginals
        bp_hard = sum(1 << int(j) for j in np.flatnonzero(q > 0.5))
        bp_hard_det = 0
        for j in _bits(bp_hard):
            bp_hard_det ^= self.det_masks[j]
        reliability = np.abs(np.log1p(-np.clip(q, 1e-12, 1 - 1e-12)) -
                             np.log(np.clip(q, 1e-12, 1 - 1e-12)))
        guided_order = np.argsort(reliability, kind='stable')
        if self.basis_update == 'compiled':
            from .compiled_basis import packed_basis
            guided_pivots = packed_basis(self._packed_masks, guided_order, self.detectors,
                                         len(self._prior_pivots))
        else:
            guided_pivots = _basis(self.det_masks, guided_order,
                                   target_rank=len(self._prior_pivots))
        guided_frame = bp_hard ^ _solve(guided_pivots, target ^ bp_hard_det)
        guided_frame = _refine(guided_frame, guided_pivots, self.det_masks,
                               self.prior_weights, guided_order,
                               self.refinement_budget)
        prior_frame = _solve(self._prior_pivots, target)
        prior_frame = _refine(prior_frame, self._prior_pivots, self.det_masks,
                              self.prior_weights, self._prior_order,
                              self.refinement_budget)
        if _score(guided_frame, self.prior_weights) < _score(prior_frame, self.prior_weights):
            best, method = guided_frame, 'bp_guided_recovery'
        else:
            best, method = prior_frame, 'prior_basis_recovery'
        recovered_det = 0
        recovered_logical = 0
        for j in _bits(best):
            recovered_det ^= self.det_masks[j]
            recovered_logical ^= self.logical_masks[j]
        if recovered_det != target:
            raise ArithmeticError('internal fault-recovery consistency failure')
        bits = np.array([(recovered_logical >> k) & 1
                         for k in range(self.observables)], dtype=np.uint8)
        return RecoveryResult(bits, True, _score(best, self.prior_weights),
                              bp_result.converged, bp_result.iterations, method)

    def decode_batch(self, syndromes: np.ndarray, *, progress_callback=None,
                     progress_every: int = 0, with_details: bool = False):
        data = np.asarray(syndromes)
        if data.ndim != 2 or data.shape[1] != self.detectors or np.any((data != 0) & (data != 1)):
            raise ValueError('expected binary matrix of syndromes')
        unique, inverse = np.unique(data, axis=0, return_inverse=True)
        results = []
        for i, row in enumerate(unique, start=1):
            results.append(self.decode(row))
            if progress_callback is not None and progress_every and (i % progress_every == 0 or i == len(unique)):
                progress_callback(i, len(unique))
        decoded = np.asarray([r.prediction for r in results], dtype=np.uint8)[inverse]
        if not with_details:
            return decoded
        return decoded, {
            'unique_syndromes': len(unique),
            'fraction_syndrome_consistent': float(np.mean([r.syndrome_consistent for r in results])) if results else 1.0,
            'fraction_bp_converged': float(np.mean([r.bp_converged for r in results])) if results else 1.0,
            'prior_basis_recovery_count': sum(r.candidate_method == 'prior_basis_recovery' for r in results),
            'bp_guided_recovery_count': sum(r.candidate_method == 'bp_guided_recovery' for r in results),
        }
