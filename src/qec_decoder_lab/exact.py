"""Exact maximum-a-posteriori *logical* inference for small independent-error DEMs.

An error mechanism is represented by (probability, XOR mask on detector and
logical-observable bits). Iterated convolution computes the joint distribution
P(detection pattern, logical frame). MAP chooses the most likely *logical
frame after marginalizing over all compatible fault combinations*; this is not
minimum-weight fault decoding. Its cost grows exponentially in detector count.

Important: correctness is exact *relative to the supplied independent DEM*.
Stim's reduction of physical noise to a DEM need not preserve every physical
Pauli correlation. This is not a hardware-calibrated Bayesian decoder.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Fault:
    probability: float
    mask: int


def exact_joint_distribution(
    detectors: int, observables: int, faults: list[Fault], *, max_bits: int = 16
) -> np.ndarray:
    bits = detectors + observables
    if detectors < 0 or observables < 1:
        raise ValueError("require detectors >= 0 and observables >= 1")
    if bits > max_bits:
        raise ValueError(
            f"Exact posterior requires 2^{bits} states; exceeds max_bits={max_bits}. "
            "Use fewer syndrome rounds or MWPM for this experiment."
        )
    n = 1 << bits
    idx = np.arange(n, dtype=np.int64)
    probs = np.zeros(n, dtype=float)
    probs[0] = 1.0
    for fault in faults:
        if not 0 <= fault.probability <= 1:
            raise ValueError("error mechanism probability must be in [0, 1]")
        if not 0 <= fault.mask < n:
            raise ValueError("error mask exceeds model bit width")
        if fault.probability:
            probs = (1 - fault.probability) * probs + fault.probability * probs[idx ^ fault.mask]
    if not np.isclose(float(np.sum(probs)), 1.0, atol=1e-10):
        raise ArithmeticError("posterior probability lost normalization")
    return probs.reshape((1 << observables, 1 << detectors))


def faults_from_dem(dem) -> tuple[int, int, list[Fault]]:
    """Read a Stim DEM using documented public target/instruction APIs.

    Iteration is over a flattened DEM (no repeat blocks/index shifts).
    The separator '^' is a suggested decomposition of ONE error mechanism,
    not an independent Bernoulli event. All its target bits XOR into one mask.
    """
    flattened = dem.flattened()
    detectors = flattened.num_detectors
    observables = flattened.num_observables
    faults: list[Fault] = []
    for instruction in flattened:
        if instruction.type != "error":
            continue
        p = float(instruction.args_copy()[0])
        mask = 0
        for target in instruction.targets_copy():
            if target.is_relative_detector_id():
                mask ^= 1 << int(target.val)
            elif target.is_logical_observable_id():
                mask ^= 1 << (detectors + int(target.val))
            elif target.is_separator():
                continue
            else:
                raise ValueError(f"Unexpected DEM error target: {target}")
        faults.append(Fault(p, mask))
    return detectors, observables, faults


class ExactBayesianDEMDecoder:
    """Precompute MAP logical frame for every detector pattern in a small DEM."""

    def __init__(self, detectors: int, observables: int, faults: list[Fault], max_bits: int = 16):
        self.detectors = detectors
        self.observables = observables
        self.joint = exact_joint_distribution(detectors, observables, faults, max_bits=max_bits)
        self.evidence = self.joint.sum(axis=0)
        self.map_logicals = np.argmax(self.joint, axis=0)
        self.map_confidence = np.divide(
            self.joint.max(axis=0), self.evidence,
            out=np.zeros_like(self.evidence), where=self.evidence > 0,
        )

    @classmethod
    def from_dem(cls, dem, *, max_bits: int = 16) -> "ExactBayesianDEMDecoder":
        return cls(*faults_from_dem(dem), max_bits=max_bits)

    def decode_batch(self, syndromes: np.ndarray) -> np.ndarray:
        arr = np.asarray(syndromes)
        if arr.ndim != 2 or arr.shape[1] != self.detectors:
            raise ValueError(f"Expected shape (shots, {self.detectors})")
        if np.any((arr != 0) & (arr != 1)):
            raise ValueError("syndrome entries must be binary")
        indices = np.zeros(arr.shape[0], dtype=np.int64)
        for j in range(self.detectors):
            indices |= arr[:, j].astype(np.int64) << j
        if np.any(self.evidence[indices] <= 0):
            raise ValueError("Observed syndrome has zero probability under DEM")
        frames = self.map_logicals[indices]
        predictions = np.empty((arr.shape[0], self.observables), dtype=np.uint8)
        for j in range(self.observables):
            predictions[:, j] = (frames >> j) & 1
        return predictions
