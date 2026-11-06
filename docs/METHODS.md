# Methods

## Simulation model

Stim generates rotated surface-code X/Z memory circuits. Unless explicitly overridden, syndrome rounds scale with distance (`r=d`). Noise is specified using circuit-level, data-only, measurement-only, or explicit custom parameters. Decoder observations come from the same sampled detector event matrix and true logical observables.

The detector error model (DEM) is treated as a list of Bernoulli fault mechanisms with detector and logical flip masks. This abstraction is not a full hardware noise model. DEM decomposition and correlations require careful interpretation.

## Decoders

**MWPM reference:** PyMatching's matching decoder is the baseline for all sampled-shot comparisons. Its compiled batch implementation and the Python experimental decoders have different performance characteristics.

**Exact Bayesian:** Enumerates the joint probability of detector and logical frames for tractably small independent-error DEMs. It computes logical MAP after marginalizing over compatible faults. Complexity grows as `2^(detectors + logicals)`. A configured cap prevents an unreasonable exact computation. Stage 4's full models exceeded this cap.

**Original loopy BP:** Sum-product message passing estimates independent fault marginals. The initial logical prediction factorizes those marginals, an invalid general assumption because conditioning on syndrome parity creates correlations. It may return an inconsistent hard fault configuration and is retained as a documented negative baseline.

**Experimental syndrome-consistent recovery:** BP fault beliefs order candidate corrections, and a binary elimination basis solves the observed detector parity exactly. A prior-weighted alternative and limited nullspace moves are evaluated as well. The chosen configuration's logical parity is read directly from its fault mask. This guarantees syndrome consistency *for a syndrome representable by the DEM*. It does **not** guarantee logical MAP, beat MWPM, or preserve posterior logical correlations. Its Python implementation remains slower than native MWPM, even after numerical optimization. The check-node message update can use a vectorized log-product with explicit zero handling, and the guided Gaussian-elimination basis stops once it reaches the precomputed model rank. The old engine remains available for prediction-preservation tests. These are implementation changes, not a new decoder algorithm.

## Exact validation on projected models

The offline evaluator takes a saved full DEM, marginalizes unselected detector channels, retains all fault mechanisms and logical bits, and compares BP and recovery with exact logical MAP on a small projected syndrome space. This is a useful computational cross-check, **not** full-code Bayesian validation and **not** a physical claim about the discarded measurements.

## Fair performance comparisons

- Evaluate all candidate decoders on identical shot indices and identical logical truth.
- Report separate denominators when MWPM uses more shots than the experimental decoder.
- Define failure as any logical observable mismatch for a memory shot.
- Use Wilson intervals for individual rates, exact two-sided McNemar tests for paired discordance, and paired bootstrap intervals for differences.
- Apply Holm correction to a declared exploratory family of multiple comparisons.
- Name the comparison order explicitly: `first_decoder` and `second_decoder`; a negative `second_minus_first_failure_rate` favors the second decoder.
- Keep algorithm construction time separate from warmed decoding time; do not compare C++ batch timings directly with unoptimized Python prototypes as if matched implementations.

## Artifacts and reproducibility

Each Stage 4 run saves a source/configuration fingerprint, circuit, DEM, sampled syndrome and truth arrays, predictions, summary counts, and timestamps. Checkpoints are reused only after an evidence re-audit. Stage 5 saves bit-packed raw syndromes and outcomes, predictions, individual timing trials, and SHA-256 hashes. Audit PASS means consistency of the **recorded evidence**, not independent recreation of source experiments or scientific correctness.

## Literature protocol

The timing study follows the circuit construction and warmed batch-decoding approach published with the PyMatching v2 sparse-blossom work at `p=0.001`. The nine distances are 5, 7, 9, 13, 17, 23, 29, 39, and 50. Local timing and the authors' numerical series are kept separate, because software releases, processor hardware, and timing environments differ. See [References](REFERENCES.md).

## Frozen confirmation versus development diagnostics

Archived Stage 4 development shots were inspected during decoder development. New claims must instead use the [frozen validation protocol](VALIDATION.md). The lock records the *entire* Python source digest set, cases, decoder settings and statistical plan before independent sampling begins. The previous pre-optimization frozen study evaluated eight X/Z circuit-level points with 10,000 MWPM shots per case and a 2,048-shot paired recovery/MWPM subset. Its single Holm-significant case applies to that older source only. The new expanded post-optimization study predeclares 18 circuit-level cases across X/Z bases, distances 3/5/7 and p=0.003/0.005/0.01. Its default sample sizes are 25,000 total MWPM shots and 4,096 matched shots per case. Case-wise McNemar and Holm correction cover the entire new 18-case family. No source or decoder changes are permitted after the new protocol is frozen.

A separate larger-sample study fixes the same cases before collecting independent data. Its outcomes must not be silently combined with the primary family.

## Small-model exact-oracle tests

The oracle suite enumerates detector/logical joint probabilities on small, hand-authored and seeded randomly generated independent-error DEMs (at most 12 total detector-plus-logical bits). It compares BP and syndrome recovery using **joint-probability-weighted logical expected error**, not only MAP-label agreement. The oracle's optimal expected error is no greater than any fixed decoder under the modeled distribution. These synthetic and projected tests do **not** demonstrate full surface-code or hardware optimality.

## Convergence sensitivity

The BP diagnostic varies damping and maximum message-passing iterations on the **same archived development syndromes**, and compares hard-message convergence, hard-decision syndrome consistency and observed logical failures with the recovery method. Increasing damping may slow convergence; fixed-point convergence alone is neither logical correctness nor calibrated posterior inference. BP convergence is not automatically repaired by syndrome-consistent recovery.

## Runtime and memory measurements

The held-out performance tool launches one fresh worker subprocess for each decoder/case. Timings are repeated after warmup, using exactly the same detector rows; decoder construction is excluded from elapsed timing and included in peak *process* RSS. Python `tracemalloc` counts only Python allocations and does not capture C++ allocations, so it is reported separately. Peak process RSS includes runtime and dependency overhead, not algorithm-only memory. Both decoders run on the same host but differ substantially in implementation (native C++ versus Python). No raw cross-machine latency speedup claim is justified.

## Performance-engineering variants and hybrid gate

Sum-product BP in `vectorized` and optional `compiled` modes performs the same
independent-fault DEM update with different numerical ordering. Optional
`minsum` uses normalized minimum-magnitude parity messages instead of exact
sum-product messages and is an **approximate, distinct algorithm**. Changing
iterations may also change logical predictions.

The syndrome-consistent recovery retains both BP-guided and prior-weighted
fault candidates and selects the lower independent-prior cost, which is not an
exact logical Bayesian MAP rule. Compiled GF(2) basis generation accelerates
pivot construction without changing the intended elimination algebra.

The selective hybrid first obtains MWPM predictions for the entire batch,
computes each syndrome's fraction of active detector bits, and runs recovery
only when that fraction reaches the fixed frozen gate. This gate is not a
posterior confidence score. Its threshold was tuned on development data and
must be independently validated before accuracy/cost claims are made.
