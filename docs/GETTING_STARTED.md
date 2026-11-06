# Getting started

## Environment

- Python 3.11 or later for the core package. Python 3.12 is recommended for the full PennyLane check.
- `stim`, `pymatching`, `numpy` for surface-code experiments.
- `pytest` for tests, `matplotlib` for figures, and `pennylane` for the independent parity-check example.

Use an interpreter that already has the dependencies, or install with your preferred Python package manager. For Homebrew-managed Python, avoid bypassing PEP 668's externally managed environment safeguards. A project environment is an option, not a requirement if packages are installed through an approved method.

```bash
python3 -m pip install -e '.[dev,plots]'
# Optional PennyLane checks, if compatible with the selected interpreter:
python3 -m pip install -e '.[quantum]'
```

Run scripts from the repository root. Use `PYTHON=python3.14` to select an already configured Python 3.14 interpreter, or `PYTHON=python3.12` when PennyLane requires it.

## Checks and experiments

| Command | Purpose | Main output |
|---|---|---|
| `bash scripts/shell/RUN_TESTS.sh` | Unit tests | Terminal results |
| `bash scripts/shell/RUN_STAGE3_VERIFY.sh` | Integration, exact small models, PennyLane, replay | `outputs/stage3/` |
| `bash scripts/shell/RUN_STAGE4_COMPARISON.sh` | Quick MWPM/BP smoke test | `outputs/stage4_quick/` |
| `MODE=full bash scripts/shell/RUN_STAGE4_COMPARISON.sh` | Full 72-case comparison | `outputs/stage4/` |
| `RECOVERY=1 bash scripts/shell/RUN_STAGE4_COMPARISON.sh` | Opt-in new recovery on sampled shots | Selected Stage 4 output directory |
| `bash scripts/shell/RUN_STAGE5_PAPER.sh` | Nine-distance PyMatching study and audit | `outputs/stage5_paper/` |
| `bash scripts/shell/RUN_RECOVERY_EVALUATION.sh` | Development-only archived Stage 4 re-evaluation | `outputs/recovery_eval/` |
| `bash scripts/shell/RUN_EXACT_ORACLE.sh` | 28 small exact-Bayesian models | `outputs/exact_oracle/` |
| `bash scripts/shell/RUN_BP_SENSITIVITY.sh` | BP damping and iteration study on old data | `outputs/bp_convergence/` |
| `bash scripts/shell/RUN_HELDOUT_VALIDATION.sh` | Lock source, sample new shots, compare and audit | `outputs/heldout/` |
| `bash scripts/shell/RUN_TARGETED_ACCURACY.sh` | Separate larger-sample locked study | `outputs/large_sample_study/` |
| `bash scripts/shell/RUN_PERFORMANCE.sh` | Isolated-worker decoder timing and memory | Selected held-out directory `performance/` |
| `bash scripts/shell/RUN_OPTIMIZATION_DEVELOPMENT.sh` | Development-only old versus optimized regression and timings | `outputs/optimization_development/` |
| `bash scripts/shell/RUN_OPTIMIZED_HELDOUT.sh` | New 18-case locked experiment including d=7 | `outputs/optimized_heldout/` |
| `bash scripts/shell/PACK_VALIDATION_EVIDENCE.sh` | Gate and package held-out evidence | `qec-heldout-validation-evidence.zip` |

In Stage 4 full mode, `SHOTS=2000` samples 2,000 shots per configuration for MWPM; `BP_SHOTS=256` compares BP on the **same first 256** shots. `RESUME=1` only reuses verified checkpoints with matching source and parameter fingerprints. The new `RECOVERY=1` mode adds a computationally expensive prototype and must not be confused with the historical 72-case baseline.

## Offline recovery evaluation

Extract an older `stage4.zip` such that it contains `stage4/runs/*.json`, `*.dem`, and `*.npz`. Set `ARCHIVE_ROOT` to that `stage4/` directory:

```bash
ARCHIVE_ROOT=../archive/stage4 SHOTS=256 \
PYTHON=python3 bash scripts/shell/RUN_RECOVERY_EVALUATION.sh
```

The offline tool checks recorded raw-array hashes, parses compressed DEM repeat blocks, computes the new decoder's logical predictions, and compares the same shot indices with archived MWPM and original BP outputs. It also performs a small marginal-projection check against exact Bayesian inference; **this is not exact inference on the full surface code**.

The default evaluates the entire archive, which can be slow for larger detector models. For a diagnostic run use `LIMIT=8 SHOTS=64` and label it exploratory. Never describe a reduced run as the full 72-case study.

## Stage 5 evidence

```bash
STRICT_ALL_NINE=1 REDO_DECODE=1 RESUME=1 \
PYTHON=python3 bash scripts/shell/RUN_STAGE5_PAPER.sh

STAGE5_ONLY=1 PYTHON=python3 bash scripts/shell/PACK_EVIDENCE.sh
```

A complete Stage 5 run requires nine distances, at least 10,000 shots per distance, and at least seven timing repetitions. The audit checks archived arrays and checksums; `REDO_DECODE=1` additionally recomputes predictions. `REPLAY_SAMPLING=1` requests seeded replay when supported. The output cannot be interpreted as matching the original authors' CPU speed.

## Where to find the results

Local runs create files inside this checkout's ignored `outputs/` directory. GitHub Actions runs create them on temporary runners and upload selected directories as **downloadable run artifacts**; they do not commit them to the repository. To retrieve those files, open **GitHub → Actions → workflow → completed run → Artifacts**, then download and extract the relevant ZIP. Workflow/artifact names, CLI instructions, and retention guidance are in [Outputs and GitHub Actions artifacts](OUTPUTS_AND_ARTIFACTS.md).

## Reviewing results

For a report, preserve its `.json`, `.dem`, `.stim`, `.npz`, source revision, environment versions, and the terminal log. Run the independent auditors before relying on aggregate figures. If a quantum dependency is unavailable, the package's pure Python unit tests can run, but a skipped quantum integration test is **not** a verified simulator result.

## New post-optimization confirmation workflow

Read [Validation](VALIDATION.md) **before sampling**. Execute `RUN_OPTIMIZED_HELDOUT.sh` to lock 18 cases with distance 7 and three noise rates, then `OUTPUT_DIR=outputs/optimized_heldout bash scripts/shell/RUN_PERFORMANCE.sh`. Run any high-sample follow-up in a different directory. Never reuse the older held-out lock or the archived Stage 4 development arrays. The audit checks altered source, altered data, incomplete cases, and changed denominators. Package with `OUTPUT_DIR=outputs/optimized_heldout bash scripts/shell/PACK_VALIDATION_EVIDENCE.sh`. That 18-case optimized study has since completed; this section documents how it was run. Future modified decoder settings require a new frozen study.

## Performance-engineering experiments

Use a Python environment with the project's core dependencies. An optional
Numba build provides compiled BP and bit-packed basis kernels:

```bash
python -m pip install -e '.[dev,plots,accelerated]'
```

Numba wheels are Python-version-specific; select a compatible interpreter
(e.g. Python 3.12/3.13 where supported) for `CHECK_UPDATE=compiled`. The default
`vectorized` engine does not require Numba and can run with a supported
Python 3.14 installation.

To repeat *retrospective* development checks, extract a previously reviewed
`heldout/cases` archive (outside the Git source checkout) and set `ARCHIVE_ROOT`:

```bash
ARCHIVE_ROOT=/path/to/extracted/heldout bash scripts/shell/RUN_PERFORMANCE_ABLATION.sh
ARCHIVE_ROOT=/path/to/extracted/heldout bash scripts/shell/RUN_ENGINE_STUDY.sh
ARCHIVE_ROOT=/path/to/extracted/heldout bash scripts/shell/RUN_HYBRID_GATE_STUDY.sh
```

These results are **DEVELOPMENT_ONLY** because the archive has already been
inspected and used for tuning.

For a **new** frozen validation of a *new variant*, run from a clean source tree without modifying
Python modules after plan creation:

```bash
PYTHON=python3.14 bash scripts/shell/RUN_HYBRID_HELDOUT.sh
OUTPUT_DIR=outputs/hybrid_heldout TIMING_SHOTS=256 REPEATS=5 \
  PYTHON=python3.14 bash scripts/shell/RUN_HYBRID_PERFORMANCE.sh
PYTHON=python3.14 bash scripts/shell/PACK_HYBRID_EVIDENCE.sh
```

For a compiled *separate* protocol, use `CHECK_UPDATE=compiled` and
`BASIS_UPDATE=compiled` with a compatible Numba interpreter, plus a brand-new
`OUTPUT_DIR`. Do not treat a retrospective gate selection as held-out evidence.

See [Performance engineering](PERFORMANCE_ENGINEERING.md) for numerical
results, methodology and accuracy/latency limitations.


### Existing evidence versus new experiments

The previously reported frozen 18-case hybrid evaluation is **complete** and reviewed, including its latency study. The commands above describe how to **run a different, newly locked study**. Never relabel the historical data as fresh validation for modified source code. For the current measured results see [Results and limitations](RESULTS_AND_LIMITATIONS.md) and [Evidence and provenance](EVIDENCE.md); for the prioritized next measurement see the [roadmap](ROADMAP.md).
