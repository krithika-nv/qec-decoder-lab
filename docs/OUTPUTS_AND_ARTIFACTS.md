# Outputs and GitHub Actions artifacts

Experiments write their results to `outputs/` **while they run**. This is a working directory, not a public results database. The repository retains `outputs/.gitkeep` so the directory exists after checkout; `.gitignore` excludes generated files. Do not commit raw syndromes, large timing arrays, logs, or local experimental ZIPs.

## Where results live

| Where the experiment runs | Where results are written | How to access them afterward |
|---|---|---|
| Locally | `outputs/<study>/` in **your local checkout** | Open the files locally, or package them with the appropriate `PACK_*_EVIDENCE.sh` script |
| GitHub Actions | `outputs/<study>/` on an **ephemeral GitHub-hosted runner** | Open that workflow run's **Artifacts** section and download the named ZIP artifact |

**GitHub Actions does not commit its `outputs/` directory back to `main`.** Each run starts in a fresh checkout, so a new workflow cannot rely on artifacts or outputs from an earlier run unless you explicitly download and supply them. Uploaded artifacts can expire. Download and separately preserve any evidence that must remain available beyond the configured retention period.

## Download results from GitHub Actions

1. Open your GitHub repository and select the **Actions** tab.
2. In the left sidebar, select the workflow you ran, for example **QEC hybrid performance validation (manual)**.
3. Open the specific run (verify its branch, commit SHA, input settings and **conclusion**).
4. On the run summary, locate **Artifacts** and choose the relevant artifact name from the table below.
5. Download the artifact, unzip it, and inspect the included `outputs/` subdirectory or packaged evidence ZIP. Preserve the accompanying audit/protocol/versions information when sharing the results.

Artifacts are **not evidence of a successful study by themselves**: an upload step may run after an experiment fails, and the resulting artifact may be partial. Check the workflow conclusion and the experiment-specific audit status before treating results as complete. Missing artifacts can mean the run has not generated files, upload failed, or retention has expired.

To download from the command line with the [GitHub CLI](https://cli.github.com/), authenticate with `gh auth login`, find the numeric run ID using `gh run list` or the web UI, then run:

```bash
gh run download RUN_ID --dir downloaded-qec-artifacts
```

This retrieves artifacts for the selected run into `downloaded-qec-artifacts/`. Use a new directory per run to avoid mixing results. If an artifact contains a separate `qec-*-evidence.zip`, unzip that archive as well.

## Workflow-to-artifact reference

These mappings reflect the workflow configurations included in this repository. The displayed artifact name can contain the run ID, Python version, study profile, or selected engine.

| Workflow (Actions tab) | Trigger | Runner output path(s) | Artifact in the run | Configured retention |
|---|---|---|---|---|
| **CI - tests and numerical invariants** | Push to `main`, PR, or manual | `outputs/ci-smoke.*`, `outputs/stage4_quick/`, `outputs/pennylane/`, `outputs/heldout/`, `outputs/exact_oracle/` | `qec-smoke-py<version>`, `qec-pennylane-check`, `qec-frozen-smoke` | Smoke and PennyLane: repository default; frozen smoke: **14 days** |
| **Optimized decoder validation (manual)** | Manual | `outputs/optimized_heldout/` | `qec-optimized-validation-<profile>` | **14 days** |
| **QEC hybrid performance validation (manual)** | Manual | `outputs/hybrid_heldout/` | `qec-hybrid-<profile>-<engine>` | **14 days** |
| **Paper benchmark - nine-distance measured PyMatching evidence** | Manual | `outputs/stage5_paper/` | `paper-timing-<run-id>` | **30 days** |
| **Stages 3-5 - verified experimental evidence** | Manual | `outputs/stage3/`, `outputs/stage4/`, `outputs/stage5_paper/` | `stage3-reproducibility`, `stage4-paired-study`, `stage5-paper-matching` | Repository default |

The table describes **configured upload steps**, not a guarantee that every listed artifact exists for every execution. The default retention is governed by the repository/organization settings and GitHub limits; it is not a promise of indefinite storage. See the workflow files under `.github/workflows/` for definitive names and paths.

### Which artifact should I download?

- **Hybrid accuracy, gate selection, and timing:** `qec-hybrid-<profile>-<engine>` from the hybrid manual workflow. It contains `outputs/hybrid_heldout/` and, when packaging succeeds, `qec-hybrid-heldout-evidence.zip`.
- **Optimized full-recovery comparison:** `qec-optimized-validation-<profile>`, including `qec-optimized-heldout-evidence.zip` when packaging succeeds.
- **Nine-distance PyMatching timing:** `paper-timing-<run-id>`, including `outputs/stage5_paper/` with timing CSVs, archives, logs and audit files if those steps ran.
- **PR smoke and integration checks:** `qec-smoke-py<version>`, `qec-pennylane-check`, or `qec-frozen-smoke`. These are **smoke tests**, not the complete scientific study.

Do not assume a smoke-profile artifact reproduces the full 18-case held-out or nine-distance timing study. Always read the saved experiment configuration and audit result.

## Run and inspect results locally

From the project root, a low-cost example is:

```bash
OUTPUT_DIR=outputs/hybrid_smoke MODE=smoke SHOTS=64 PAIRED_SHOTS=16 \
  PYTHON=python3 bash scripts/shell/RUN_HYBRID_HELDOUT.sh
ls -lah outputs/hybrid_smoke/
```

The output directory is a local artifact of **that specific run**. It does not automatically appear in GitHub's source-code browser. For full studies, use the commands in [Getting started](GETTING_STARTED.md) and the evidence packagers described in [Validation](VALIDATION.md).

## Preserve a reviewable result

1. Confirm the workflow conclusion and the experiment-specific evidence audit. Retain the source revision, frozen protocol, package dependency details, sample counts, model files, seeds, predictions and timing trials as applicable.
2. Download the complete evidence archive before the GitHub Actions artifact expires. Store it separately with its SHA-256 checksum and avoid overwriting a frozen result.
3. Keep curated summary tables and attributed figures under `docs/`; avoid committing large raw shot files or sensitive local metadata. Clearly distinguish code changes from independent experimental validation.

See [Evidence and provenance](EVIDENCE.md) for archived-result hashes and [Results and limitations](RESULTS_AND_LIMITATIONS.md) for the supported scientific conclusions.
