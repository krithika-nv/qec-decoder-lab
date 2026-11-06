# Contributing

QEC Decoder Lab is a software project with a research direction. Contributions are welcome when they maintain reproducibility, preserve negative results and avoid unsupported decoder-superiority claims.

## Workflow

1. Work on a feature branch and open a pull request into protected `main`; do not push directly to `main`.
2. Run `PYTHON=python3 bash scripts/shell/RUN_TESTS.sh` and any relevant smoke test. Record skipped native integration modules rather than counting them as verified experiments.
3. For decoder changes, add numerical regression and syndrome-consistency tests. If the algorithm changes logical predictions, treat it as a **new algorithm candidate** and use new frozen sampling for accuracy claims.
4. Update [Methods](METHODS.md), [Results](RESULTS_AND_LIMITATIONS.md), [Evidence](EVIDENCE.md), or [Roadmap](ROADMAP.md) when code, findings, or scientific interpretation change.
5. Verify shell syntax, links, security-sensitive file additions and CI. Keep raw `outputs/` and private local paths out of commits.
6. Include the change, test commands, limitations, and any independent experiment/evidence link in the PR description.

## GitHub controls

Enable a branch ruleset requiring pull requests and successful CI checks before `main` can change. A public repository **does not** give strangers write permission, but branch protection is still valuable for preventing your own direct pushes. These settings must be enabled in the GitHub repository; they are not created by this ZIP.

## Evidence discipline

- A passing test verifies an implementation invariant, **not** a scientific superiority claim.
- An audit PASS verifies source and artifact consistency, **not** an independent hardware experiment.
- Treat all previously inspected/used data as development data for future tuning. Freeze code, protocol, hypotheses and sampling rules before new validation.
- Report all cases, including ties, convergence failures, adverse comparisons and non-significant outcomes.
- Store large raw evidence ZIPs separately with SHA-256 and source revision; commit only small attributed figures and summary tables.

