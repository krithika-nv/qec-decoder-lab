"""CLI that saves machine-readable evidence and does not silently hide failures."""
import argparse
import csv
import hashlib
from datetime import datetime, timezone
import json
from pathlib import Path

from .experiment import ExperimentSettings, run_experiment
from .model import MemoryConfig, NOISE_PROFILES
from .hardware_surrogate import surrogate_from_calibration


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--basis", choices=["x", "z"], default="x")
    parser.add_argument("--profile", choices=NOISE_PROFILES, default="circuit_level")
    parser.add_argument("--shots", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261101)
    parser.add_argument("--bayesian", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--max-bayes-bits", type=int, default=16)
    parser.add_argument("--approximate", choices=["on", "off"], default="off")
    parser.add_argument("--bp-iterations", type=int, default=30)
    parser.add_argument("--bp-damping", type=float, default=0.3)
    parser.add_argument("--bp-tolerance", type=float, default=1e-6)
    for name in (
        "after_clifford_depolarization", "before_round_data_depolarization",
        "before_measure_flip_probability", "after_reset_flip_probability",
    ):
        parser.add_argument("--" + name.replace("_", "-"), type=float, default=None)


def config(args, distance: int, p: float, rounds: int) -> MemoryConfig:
    return MemoryConfig(
        distance=distance, rounds=rounds, basis=args.basis, p=p,
        noise_profile=args.profile,
        after_clifford_depolarization=args.after_clifford_depolarization,
        before_round_data_depolarization=args.before_round_data_depolarization,
        before_measure_flip_probability=args.before_measure_flip_probability,
        after_reset_flip_probability=args.after_reset_flip_probability,
    )


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf8")


def save_model_files(json_path: Path, c: MemoryConfig, expected: dict) -> dict:
    """Save exactly specified circuit/DEM alongside run and verify provenance hashes."""
    circuit = c.circuit()
    dem = circuit.detector_error_model(decompose_errors=True)
    for obj, extension, key in (
        (circuit, ".stim", "circuit_sha256"),
        (dem, ".dem", "dem_sha256"),
    ):
        rendered = str(obj)
        checksum = hashlib.sha256(rendered.encode("utf8")).hexdigest()
        if checksum != expected[key]:
            raise RuntimeError(f"Regenerated {extension} does not match run evidence SHA256")
        json_path.with_suffix(extension).write_text(rendered, encoding="utf8")
    return {
        "stim_circuit": str(json_path.with_suffix(".stim")),
        "detector_error_model": str(json_path.with_suffix(".dem")),
        "shot_archive": str(json_path.with_suffix(".npz")),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="qec-lab", description="Reproducible surface-code memory decoder comparisons"
    )
    subs = parser.add_subparsers(dest="command", required=True)
    run = subs.add_parser("run", help="Single matched-shot memory experiment")
    add_common(run)
    run.add_argument("--distance", type=int, default=3)
    run.add_argument("--rounds", type=int, default=None, help="Default: equal to distance")
    run.add_argument("--p", type=float, default=0.005)
    run.add_argument("--output", type=Path, default=Path("outputs/single.json"))

    sweep = subs.add_parser("sweep", help="Distance/noise sweep, with raw JSON evidence")
    add_common(sweep)
    sweep.add_argument("--distances", type=int, nargs="+", default=[3, 5, 7])
    sweep.add_argument("--ps", type=float, nargs="+", default=[0.001, 0.003, 0.005])
    sweep.add_argument("--rounds", type=int, default=None,
                       help="Explicit rounds; default rounds=distance separately for each distance")
    sweep.add_argument("--output-dir", type=Path, default=Path("outputs/sweep"))
    sweep.add_argument("--plot", action="store_true", help="Requires matplotlib")

    hardware = subs.add_parser(
        "hardware-run", help="Run surrogate noise model from explicitly mapped calibration evidence"
    )
    hardware.add_argument("--calibration", type=Path, required=True)
    hardware.add_argument("--distance", type=int, default=3)
    hardware.add_argument("--rounds", type=int, default=3)
    hardware.add_argument("--basis", choices=["x", "z"], default="x")
    hardware.add_argument("--shots", type=int, default=1000)
    hardware.add_argument("--seed", type=int, default=20261103)
    hardware.add_argument("--output", type=Path, default=Path("outputs/hardware_surrogate.json"))
    hardware.add_argument("--approximate", choices=["off", "on"], default="off")
    args = parser.parse_args(argv)
    if args.command == "hardware-run":
        cfg, calibration = surrogate_from_calibration(
            args.calibration, distance=args.distance, rounds=args.rounds, basis=args.basis
        )
        data = run_experiment(ExperimentSettings(
            cfg, shots=args.shots, seed=args.seed, bayesian="off",
            approximate=args.approximate,
        ), shots_path=args.output.with_suffix(".npz"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data["hardware_surrogate"] = calibration
        data["evidence_files"] = save_model_files(args.output, cfg, data["model"])
        save_json(args.output, data)
        print(f"Wrote surrogate-only evidence: {args.output}")
        return
    if args.command == "run":
        c = config(args, args.distance, args.p, args.rounds if args.rounds is not None else args.distance)
        data = run_experiment(ExperimentSettings(
            c, shots=args.shots, seed=args.seed,
            bayesian=args.bayesian, max_bayes_bits=args.max_bayes_bits,
            approximate=args.approximate, bp_iterations=args.bp_iterations,
            bp_damping=args.bp_damping, bp_tolerance=args.bp_tolerance,
        ), shots_path=args.output.with_suffix(".npz"))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data["evidence_files"] = save_model_files(args.output, c, data["model"])
        save_json(args.output, data)
        print(f"Wrote {args.output}")
        print(f"MWPM: {data['mwpm']['logical_failures']}/{args.shots} = "
              f"{data['mwpm']['logical_failure_rate']:.6f}")
        if data["bayesian"]["status"] == "completed":
            print(f"Bayesian: {data['bayesian']['logical_failures']}/{args.shots} = "
                  f"{data['bayesian']['logical_failure_rate']:.6f}")
        else:
            print(f"Bayesian status: {data['bayesian']['status']}")
        if data["bp_approximate"]["status"] == "completed":
            print(f"BP: {data['bp_approximate']['logical_failures']}/{args.shots} = "
                  f"{data['bp_approximate']['logical_failure_rate']:.6f}; "
                  f"convergence={data['bp_approximate']['fraction_converged']:.3f}")
        return

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    manifest = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiments": [],
        "errors": [],
    }
    for distance in args.distances:
        for p_index, prob in enumerate(args.ps):
            rounds = args.rounds if args.rounds is not None else distance
            case_id = f"d{distance}_r{rounds}_{args.basis}_{args.profile}_p{prob:.7g}"
            # Derived seeds differ per combination; results are reproducible
            # under identical argument ordering.
            case_seed = args.seed + 1000003 * distance + p_index
            dest = output / "runs" / f"{case_id}.json"
            try:
                c = config(args, distance, prob, rounds)
                result = run_experiment(ExperimentSettings(
                    c, shots=args.shots, seed=case_seed, bayesian=args.bayesian,
                    max_bayes_bits=args.max_bayes_bits,
                    approximate=args.approximate, bp_iterations=args.bp_iterations,
                    bp_damping=args.bp_damping, bp_tolerance=args.bp_tolerance,
                ), shots_path=dest.with_suffix(".npz"))
                result["evidence_files"] = save_model_files(dest, c, result["model"])
                save_json(dest, result)
                manifest["experiments"].append(str(dest))
                for decoder, field in (("mwpm", "mwpm"), ("bayesian", "bayesian"),
                                       ("bp_approximate", "bp_approximate")):
                    summary = result[field]
                    if decoder != "mwpm" and summary["status"] != "completed":
                        continue
                    rows.append({
                        "distance": distance, "rounds": rounds, "basis": args.basis,
                        "noise_profile": args.profile, "p": prob,
                        "seed": case_seed, "decoder": decoder, "status": "completed",
                        **{k: summary[k] for k in (
                            "shots", "logical_failures", "logical_failure_rate",
                            "wilson_95_low", "wilson_95_high")},
                    })
                print(f"OK {case_id}: MWPM {result['mwpm']['logical_failure_rate']:.5f}; "
                      f"Bayesian {result['bayesian']['status']}")
            except Exception as exc:
                manifest["errors"].append({"case": case_id, "error": repr(exc)})
                print(f"FAILED {case_id}: {exc}")
    fieldnames = (
        "distance", "rounds", "basis", "noise_profile", "p", "seed", "decoder",
        "status", "shots", "logical_failures", "logical_failure_rate",
        "wilson_95_low", "wilson_95_high",
    )
    with (output / "summary.csv").open("w", newline="", encoding="utf8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    if args.plot and rows:
        from .plotting import plot_sweep
        plot_sweep(output / "summary.csv", output / "logical_error_rates.png")
    save_json(output / "manifest.json", manifest)
    print(f"Wrote {output / 'summary.csv'} and {output / 'manifest.json'}")
    if manifest["errors"]:
        raise SystemExit(f"{len(manifest['errors'])} cases failed; see manifest.json")


if __name__ == "__main__":
    main()
