# Project roadmap

This roadmap tracks **software and validation milestones**, not a percentage of scientific knowledge completed. Each milestone has a deliverable and a measurable exit gate. All future performance targets are *engineering objectives*, not results that have already been achieved.

## Current baseline

**Status: implemented and locally tested; independent quantum replay and remote CI status depend on where the project is run.**

Completed:

- Stim surface-code simulations, PyMatching MWPM baseline, original BP, syndrome-consistent recovery and selective hybrid.
- A 72-case development comparison, exact logical-MAP oracle checks on small models and a nine-distance PyMatching timing study.
- Two separate 18-case frozen simulation studies: optimized recovery (**7/18** Holm-significant comparisons) and selective hybrid (**5/18**).
- Evidence locks, paired statistics, performance measurement, archived-shot checks, tests and manual CI study workflows.
- Documented negative results, generalization limits, and the unsolved latency gap.

**Quality checks:** applicable tests and CI should pass, README results should remain consistent with frozen records, evidence hashes should be preserved, and documentation links should resolve.

## Next: performance attribution (P0)

**Problem:** the reported hybrid median per-case slowdown is about **1,202×** versus native MWPM. The earlier **3,401×** figure for full recovery used different timing conditions and cannot be subtracted or divided into a controlled improvement claim.

**Deliverables**

1. Run MWPM-only, full recovery and hybrid on the **same archived input batches**, same machine, warmup policy, decoder-construction scope, and repeat count.
2. Include gate evaluation and MWPM-on-every-shot time in the hybrid measurement; report cold-start/JIT overhead separately.
3. Profile high-noise d=7 cases into BP updates, GF(2) basis/solve, refinement and Python/native crossing.
4. Archive raw timing trials, machine details, predictions, peak process RSS, and the input hashes.

**Exit gate:** deterministic result replay; identical logical predictions where algorithm-preserving changes are claimed; at least five timed repeats per case (preferably ten for detailed comparisons); transparent case-by-case timings including uncertainty; an independent artifact audit. Do not claim an aggregate speedup until this direct comparison exists.

## Native and algorithmic acceleration (P1)

**Problem:** expensive per-syndrome message passing and basis reconstruction dominate at d=7.

**Deliverables**

- Benchmark the existing compiled BP core and GF(2) kernels with profiling that separates compilation and steady-state time.
- Prototype cached or locally decomposed syndrome handling, bounded-cost fault recovery, and a compiled/native basis path if warranted.
- Compare exact-prediction-preserving optimizations separately from accuracy-changing five-iteration/min-sum algorithms.
- Add regression tests for parity recovery, zero messages, repeated detector-model blocks, rank degeneracy, and large syndromes.

**Exit gate:** no unexplained prediction changes for semantics-preserving variants, memory and time reproducible on the same workload, explicit fallback behaviour, and a separately frozen accuracy study for any algorithmic change. A tenfold reduction in recovery-path runtime is a **stretch objective**, not an achieved result.

## More selective decision-making (P2)

**Problem:** at p=0.01 the current gate sends roughly 72% of all pooled shots, and over 90% in some d=7 cases, to costly recovery.

**Deliverables**

- Test detector-local structure, MWPM ambiguity/confidence proxies, and cheaper prefilters on **development-only** data.
- Measure gate inference overhead, true recovery invocation fraction, missed beneficial recoveries and adverse overrides.
- Select and freeze a candidate gate *before* new sampling; run an independent 18-case matched-shot X/Z, d=3/5/7 validation.
- Analyze accuracy, timing and memory jointly instead of optimizing a single metric.

**Exit gate:** predeclared operating points, transparent selection trade-offs, two-sided paired tests with study-wide correction, new seeds/shots, complete raw evidence and a reviewed failure analysis. Do not compare only best-performing cases.

## Broader decoder and noise evaluation (P3)

**Deliverables**

- Compare against stronger published decoder baselines under matched circuits, noise parameters, sample sizes and time scopes.
- Extend to additional distances and correlated or biased noise models; distinguish circuit-level logical-memory performance from hardware predictions.
- Reproduce at least one specifically named published logical-error result using primary author inputs, not just a reference curve.
- Define a reusable hardware-calibration adapter and uncertainty reporting, without treating surrogate parameters as measurements.

**Exit gate:** independent holdouts, source-linked provenance, matched assumptions, uncertainty bounds, and explicit failures/limits. Hardware-validity requires actual device data and comparisons against measured observables.

## Backlog and non-goals

- Investigate confidence-guided resource allocation and fault-tolerant logical operation benchmarks after the decoding performance bottleneck is understood.
- Profile memory pressure and scaling at d=9/11 before claiming production scalability.
- **Not claimed:** real-time decoder, full-code exact Bayesian optimality, a code threshold estimate, hardware-calibrated QEC prediction, universal superiority to MWPM, or a novel published research result.

## Working rules

A completed **implementation** is not a completed **validation**. Each newly tuned algorithm receives a separate frozen protocol with a unique identity and independent samples. Keep the negative BP results and all archived prior studies unchanged; use new data for independent confirmations. Update this roadmap when an exit gate is achieved, not simply when a script has been written.
