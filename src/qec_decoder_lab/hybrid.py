"""Selective MWPM / fault-recovery hybrid for prospective validation.

A fixed fraction of active detectors is used as a CHEAP workload-complexity
proxy, NOT a calibrated decoder-confidence estimate. The gate was explored
on previously inspected samples, so any selected threshold requires a fresh
pre-registered study before accuracy conclusions are reported.
"""
from __future__ import annotations

import numpy as np


def select_complex_syndromes(syndromes: np.ndarray, threshold: float) -> np.ndarray:
    """Use recovery only when active-detector fraction >= fixed threshold."""
    s = np.asarray(syndromes)
    if s.ndim != 2 or not np.all((s == 0) | (s == 1)):
        raise ValueError('expected a binary (shots, detectors) array')
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and in [0, 1]')
    if s.shape[1] == 0:
        return np.zeros(s.shape[0], dtype=bool)
    # Avoid floating-point rounding of a large matrix of ratios.
    return np.count_nonzero(s, axis=1) >= threshold * s.shape[1]


class SelectiveHybridDecoder:
    """MWPM on all shots; recovery only for a pre-specified detector fraction."""

    def __init__(self, matching, recovery_decoder, *, threshold: float = 0.10):
        if not 0 <= threshold <= 1:
            raise ValueError('invalid threshold')
        self.matching = matching
        self.recovery = recovery_decoder
        self.threshold = float(threshold)

    def decode_batch(self, syndromes: np.ndarray, *, with_details: bool = False):
        s = np.asarray(syndromes)
        selected = select_complex_syndromes(s, self.threshold)
        output = np.asarray(self.matching.decode_batch(s), dtype=np.uint8).copy()
        if (output.ndim != 2 or output.shape[0] != s.shape[0] or
                output.shape[1] != self.recovery.observables):
            raise ValueError('MWPM prediction shape mismatch')
        if np.any(selected):
            replacement = np.asarray(self.recovery.decode_batch(s[selected]), dtype=np.uint8)
            if replacement.shape != output[selected].shape:
                raise ValueError('recovery prediction shape mismatch')
            output[selected] = replacement
        if with_details:
            return output, {'selected_shots':int(selected.sum()),
                            'total_shots':len(selected),
                            'selected_fraction':float(np.mean(selected)) if len(selected) else 0.0,
                            'threshold':self.threshold,
                            'gate_basis':'detector_hamming_weight_fraction',
                            'inference_scope':'MWPM plus optional Python recovery, not real-time validated'}
        return output
