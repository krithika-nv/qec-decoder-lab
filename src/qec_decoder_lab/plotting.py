"""Optional static plot from actual saved experiment results (never illustrative)."""
import csv
from pathlib import Path


def plot_sweep(csv_file: str | Path, output_png: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Path(csv_file).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("No experimental rows found")
    grouped: dict[tuple[int, str], list[dict]] = {}
    for row in rows:
        if row["status"] != "completed":
            continue
        grouped.setdefault((int(row["distance"]), row["decoder"]), []).append(row)
    if not grouped:
        raise ValueError("No completed rows for plotting")
    fig, ax = plt.subplots(figsize=(8, 5))
    for (distance, decoder), points in sorted(grouped.items()):
        points.sort(key=lambda x: float(x["p"]))
        ps = [float(x["p"]) for x in points]
        rates = [float(x["logical_failure_rate"]) for x in points]
        low = [float(x["wilson_95_low"]) for x in points]
        high = [float(x["wilson_95_high"]) for x in points]
        # Linear axes preserve zero-failure point estimates and CI coverage.
        ax.errorbar(ps, rates, yerr=[
            [r - l for r, l in zip(rates, low)],
            [h - r for r, h in zip(rates, high)],
        ], marker="o", capsize=3, label=f"d={distance} · {decoder}")
    ax.set_xlabel("Configured physical noise probability per selected location")
    ax.set_ylabel("Logical memory failure probability (per experiment)")
    ax.set_title("Rotated surface-code memory: sampled logical failures")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160)
    plt.close(fig)
