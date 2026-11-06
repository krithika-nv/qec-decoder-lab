# Decoder validation protocol

This page explains the frozen-study and artifact-audit workflows. The original eight-case, optimized 18-case and selective-hybrid 18-case validations have **all completed**, each with a separate source lock and dataset. New optimized or gated candidates still require their own independent sampling; none of these runs establishes a universal decoder advantage or practical low latency.

## 1. Development diagnostics (not independent evidence)

The original Stage 4 archive was used while designing the recovery decoder. Its 72-case results, including any improvement in error count, **must never be called held-out or used as a confirmatory test**.

```bash
# Requires archived Stage 4 runs/*.dem/.npz/.json
ARCHIVE_ROOT=outputs/stage4 LIMIT=4 SHOTS=48 \
PYTHON=python3.14 bash scripts/shell/RUN_BP_SENSITIVITY.sh

# 4 hand-authored and 24 deterministic random small DEMs, no Stim required
PYTHON=python3.14 bash scripts/shell/RUN_EXACT_ORACLE.sh
```

- `outputs/bp_convergence/bp_convergence.csv`: original BP logical errors, hard-fault syndrome consistency, message convergence, and syndrome-consistent recovery under varied damping and iteration caps. Sensitivity analysis is **development-only**, not a solution to BP's convergence problem.
- `outputs/exact_oracle/exact_oracle_summary.csv`: full exact logical-MAP risk versus BP and recovery on **small** independent-error detector models. Records excess expected logical failure risk and probability-weighted MAP agreement. Exact inference does **not** scale to the entire Stage 4 surface-code detector model.

## 2. Completed pre-optimization held-out evidence

The previous frozen eight-case run used new samples distinct from Stage 4 development shots. Across 16,384 matched shots, MWPM recorded 734 errors and recovery 609; only Z-memory d=5 p=0.01 survived eight-case Holm correction. The earlier recovery implementation was about 6,000–14,000 times slower than PyMatching in small-batch timing. Preserve its original source archive, protocol lock and output separately. The current source intentionally cannot resume or re-audit that frozen run as an unchanged source version.

## 3. Freeze the optimized implementation before a new sample

For a new test, use a NEW output directory and do not modify the algorithm after planning. **Do not reuse any Stage 4 shot archive.**

```bash
# Complete expanded study, not a smoke test
PYTHON=python3.14 bash scripts/shell/RUN_OPTIMIZED_HELDOUT.sh
```

The script first creates `outputs/heldout/locked_protocol.json`. It freezes:

- Source SHA-256 hashes for all Python modules, BP check-update implementation, parameters, cases, shot counts, error metric and declared exact McNemar tests with Holm adjustment.
- **18 predeclared circuit-level cases:** X and Z memory, distances 3, 5 and 7, physical error rates 0.003, 0.005 and 0.01. The smaller `MODE=smoke` is solely an execution check.
- **25,000 new Stim shots per case** for MWPM. Only the **first 4,096 same shots** are decoded by recovery; both decoders are compared on those 4,096 matched observations.
- New cryptographically drawn sampling seeds generated **after** the protocol is written. Every seed is stored in the case record for reproducibility and independently checked for uniqueness.

The protocol will not be overwritten. Its source and configuration hashes are checked before each run and audit. Interrupted complete cases are resumable, but a partially written case is rejected rather than silently overwritten. Move any incomplete files to a separate quarantined directory before retrying, retaining a record of the interrupted attempt.

To audit the results independently from raw arrays:

```bash
PYTHONPATH=src python3.14 -m qec_decoder_lab.frozen_validation audit \
  --root outputs/optimized_heldout --redo-mwpm --redo-recovery --replay-seed
```

The optional decoder/seed replay checks are computationally expensive. The optimized study uses a separate directory and a versioned protocol schema; never copy old held-out data into it. The routine audit always verifies raw array hashes, circuit and DEM hashes, dimensions, logical failure counts, paired McNemar statistics, bootstrap intervals, case coverage, uniqueness of seeds, source hashes and Holm correction.

**Caution:** The first 2,048 shots are *not* additional samples. Zero observed failures or non-significant tests do not prove the decoders equivalent. This protocol prevents post-selection of cases and denominator switching but does not eliminate implementation bugs or guarantee adequate statistical power.

## 4. Higher-sample accuracy study

For a separate, predeclared sensitivity study, run **without selecting cases based on their observed results**:

```bash
MODE=expanded OUTPUT_DIR=outputs/optimized_large_sample_study \
SHOTS=25000 PAIRED_SHOTS=8192 \
PYTHON=python3.14 bash scripts/shell/RUN_TARGETED_ACCURACY.sh
```

This uses the same 18 predeclared operating points and a **new independent** sample. It is substantially slower. Record the whole matrix, including ties, failures and adverse results. Report all eighteen corrected case tests rather than only the best-looking one. If performed after inspecting the main held-out results, label it a follow-up confirmation/sensitivity study, not part of the originally declared main family.

## 5. Same-machine latency and memory

After the complete held-out study, run:

```bash
OUTPUT_DIR=outputs/optimized_heldout TIMING_SHOTS=64 REPEATS=5 PYTHON=python3.14 \
bash scripts/shell/RUN_PERFORMANCE.sh
```

Each decoder runs in its own fresh Python subprocess with the identical archived detector rows. Decoder construction is excluded from **timed warmed batch decode** but included in the fresh process's **peak RSS**. Each record has individual trial durations, median microseconds per shot, entire-process peak memory, Python traced warmup allocation and machine details. These are Python implementation versus optimized C++ implementation measurements; they are not an equal-optimization theoretical algorithm comparison. CPU load, thermal conditions and memory allocator effects are not controlled.

`outputs/optimized_heldout/performance/performance_results.json` is audited against individual subprocess records and the original raw-shot hashes.

## 6. Final audit and evidence bundle

```bash
OUTPUT_DIR=outputs/optimized_heldout ZIP_PATH=qec-optimized-heldout-evidence.zip \
PYTHON=python3.14 bash scripts/shell/PACK_VALIDATION_EVIDENCE.sh
```

Creates `qec-optimized-heldout-evidence.zip` with all raw held-out arrays, true outcomes, both decoders' predictions, circuit and DEM source, individual performance trials, case reports, audit outputs, source hashes and a file inventory. Packaging **fails** if a case or its audit is incomplete, or if the performance results are absent/unverified. Set `WITHOUT_PERFORMANCE=1` only to explicitly publish an accuracy-only bundle.

The ZIP is not automatically committed to Git. Archive it separately; commit only code and small explanatory reports.

## Interpreting the evidence

The primary effect is **recovery logical failure rate minus MWPM logical failure rate** on each pair of identically sampled shots. Negative favours recovery. The primary inferential family is *all predeclared cases*, with two-sided exact McNemar and Holm-adjusted p-values. A bootstrap interval provides descriptive effect-size uncertainty. Aggregate pooled error counts are heterogeneous descriptive numbers, not a valid standalone claim of general algorithm superiority. Benchmark time and memory are descriptive engineering metrics; they do not enter the prespecified accuracy hypothesis.

An **audit PASS** validates recorded artifacts and arithmetic, not independent experimental physics, superiority, a logical threshold, hardware performance, or exact full-code decoding.

## 7. Development-only implementation profiling

The vectorized update and rank-limited basis were tested against previously reviewed held-out detector rows. These are **development samples** after they are used for optimization. The reproducibility script checks archived SHA-256 hashes and verifies optimized predictions against historical outputs, then records trial-by-trial old-versus-new timing:

```bash
ARCHIVE_ROOT=/path/to/extracted/heldout PYTHON=python3.14 \
bash scripts/shell/RUN_OPTIMIZATION_DEVELOPMENT.sh
```

The observed 3.80× and 4.45× same-process speedups in two five-repeat cases, and the eight-case 32-shot prediction-preservation checks, are engineering diagnostics, not inferential claims or comparisons against PyMatching. The 18-case optimized-source performance experiment has now completed; this instruction remains relevant for each new altered algorithm and frozen study.

## Selective-hybrid validation

The **hybrid** is a different algorithmic configuration from full recovery.
Its protocol freezes `strategy=hybrid`, `gate_fraction` (default 0.10),
`check_update`, `basis_update`, iteration count, case family and full source
hashes. Every paired sample still contains the independent MWPM prediction,
actual logical truth and final hybrid prediction. A deterministic gate audit
verifies all gate-off shots exactly match MWPM.

The post-study performance worker uses the same saved shots in distinct
processes for MWPM and hybrid. Its timing checks **also compare every timed
prediction to frozen archived evidence**; altered predictions abort the
benchmark. The package gate rejects incomplete or inconsistent evidence.

An inspected archive used to select a gate is **development data**, even if it
was originally held out for a different decoder. Select a gate first and
sample entirely new shots afterward for an independent claim.


## Completed selective-hybrid 18-case study

The independent frozen hybrid run using the preselected 10% active-detector gate completed on 450,000 sampled shots, of which 73,728 were paired. It yielded **2,641 MWPM errors** and **2,208 hybrid errors** on the same paired subset, with **five of eighteen** corrected-significant case improvements. The reviewer verified 115 archived files, 32 frozen source hashes, the gate reconstruction, case statistics and 36 timing records. Independent Stim/PyMatching replay was **not** performed. The per-case median timing-ratio summary was about **1,202× slower** than MWPM; comparisons with the earlier 3,401× run are not controlled because batch conditions differ.

The current source tree was updated **after** that study was frozen. Do not regenerate or resume those historical results in this modified checkout. To repeat the experiment, recover its exact locked source and archive. For a new algorithm/gate, select a fresh output directory and lock a **new** study with no reused previously inspected outcomes. Review [Evidence](EVIDENCE.md) and [Results](RESULTS_AND_LIMITATIONS.md).

## Next independent check

The next task is a controlled performance attribution on identical stored inputs: MWPM, full recovery and hybrid, with identical warmup, batch sizing, per-shot prediction checks, timing scope and machine conditions. The repository includes general benchmarks and hybrid runners; a new dedicated three-way protocol, its lock and its independent results are **planned**, not completed. See [Roadmap](ROADMAP.md).
