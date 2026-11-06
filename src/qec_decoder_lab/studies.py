"""Stage 3/4 reproducibility gates and controlled circuit-level decoder study.

Run with `python -m qec_decoder_lab.studies benchmark ...`, then `verify ...`.
All decoder predictions for a case use the *same* sampled detector events.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
from importlib.metadata import version
import json
import platform
from pathlib import Path
from statistics import median
import time
from time import perf_counter

import numpy as np

from .belief_propagation import BeliefPropagationDEMDecoder
from .recovery import SyndromeRecoveryDecoder
from .cli import save_json, save_model_files
from .evidence import EvidenceError, verify_case
from .experiment import ExperimentSettings, run_experiment
from .model import MemoryConfig
from .paired_analysis import analyze_pair, logical_error_per_round


def analyze_archived_case(path: Path) -> dict:
    """Recalculate comparisons using the saved shot arrays, not summary fields."""
    case = json.loads(path.read_text())
    with np.load(path.with_suffix('.npz'), allow_pickle=False) as loaded:
        arrays = {key: loaded[key] for key in loaded.files}
    truth = arrays['actual_observables']
    predictions = {"mwpm": arrays['mwpm_predictions']}
    if 'bayesian_predictions' in arrays:
        predictions['exact_bayesian'] = arrays['bayesian_predictions']
    if 'bp_predictions' in arrays:
        predictions['loopy_bp'] = arrays['bp_predictions']
    if 'recovery_predictions' in arrays:
        predictions['recovery'] = arrays['recovery_predictions']
    wrong = {}
    for name, pred in predictions.items():
        if pred.ndim != 2 or pred.shape[1] != truth.shape[1] or not 1 <= len(pred) <= len(truth):
            raise ValueError(f'Invalid saved prediction dimensions: {name}')
        wrong[name] = np.any(pred != truth[:len(pred)], axis=1)
    rounds = int(case['parameters']['rounds'])
    rates = {}
    for name, values in wrong.items():
        observed = float(np.mean(values))
        # Mapping is conventional under symmetric independent logical flips;
        # for rates > 0.5 it is not meaningful.
        per_round = logical_error_per_round(observed, rounds) if observed <= 0.5 else None
        rates[name] = {'shots': len(values), 'memory_failure_rate': observed,
                       'independent_round_equivalent': per_round,
                       'logical_failures': int(np.sum(values))}
    comparisons = {}
    for name in ("loopy_bp", "exact_bayesian", "recovery"):
        if name in wrong:
            comparisons[f'mwpm_vs_{name}'] = {
                'first_decoder': 'mwpm', 'second_decoder': name,
                **analyze_pair(wrong['mwpm'][:len(wrong[name])], wrong[name]),
            }
    if 'recovery' in wrong and 'loopy_bp' in wrong:
        comparisons['loopy_bp_vs_recovery'] = {
            'first_decoder': 'loopy_bp', 'second_decoder': 'recovery',
            **analyze_pair(wrong['loopy_bp'][:len(wrong['recovery'])], wrong['recovery']),
        }
    if 'loopy_bp' in wrong and 'exact_bayesian' in wrong:
        comparisons['exact_bayesian_vs_loopy_bp'] = {
            'first_decoder': 'exact_bayesian', 'second_decoder': 'loopy_bp',
            **analyze_pair(wrong['exact_bayesian'][:len(wrong['loopy_bp'])], wrong['loopy_bp']),
        }
    return {'rates': rates, 'paired_comparisons': comparisons}


def time_decoders(config: MemoryConfig, syndromes: np.ndarray, *,
                  use_bp: bool, iterations: int = 20, repetitions: int = 3,
                  use_recovery: bool = False) -> dict:
    """Latency per shot for an identical subset, not a hardware throughput claim.

    Decoder construction excluded. MWPM warmed and timed as a C++ batch call;
    BP warmed and timed as pure-Python batch including unique-row handling.
    These are fundamentally different computational implementations.
    """
    import pymatching
    model = config.circuit().detector_error_model(decompose_errors=True)
    decoders = {'mwpm': pymatching.Matching.from_detector_error_model(model)}
    if use_bp:
        decoders['loopy_bp'] = BeliefPropagationDEMDecoder.from_dem(model, iterations=iterations)
    if use_recovery:
        decoders['recovery'] = SyndromeRecoveryDecoder.from_dem(model, iterations=iterations)
    result = {}
    for name, decoder in decoders.items():
        if name == 'loopy_bp' and len(syndromes) > 8:
            print(f'  Warmup for {name}, {len(syndromes)} timing shots, '
                  f'{repetitions} timed repetitions', flush=True)
        decoder.decode_batch(syndromes[:min(2, len(syndromes))])
        elapsed = []
        for trial in range(1, repetitions + 1):
            if name == 'loopy_bp' and len(syndromes) > 8:
                print(f'  Timing loopy BP trial {trial}/{repetitions}...', flush=True)
            started = perf_counter()
            decoder.decode_batch(syndromes)
            elapsed.append(perf_counter() - started)
        result[name] = {
            'median_microseconds_per_shot': median(elapsed) * 1e6 / len(syndromes),
            'individual_wall_seconds': elapsed,
            'sample_count': len(syndromes),
            'decoder_construction_included': False,
            'execution_scope': 'warmed decode_batch on identical syndrome subset',
        }
    return result


def run_benchmark(output: Path, *, distances: list[int], ps: list[float],
                  bases: list[str], profiles: list[str], shots: int, seed: int, approximate: bool,
                  timing_shots: int, timing_repeats: int, bp_iterations: int,
                  replay: bool, plot: bool, bp_shots: int | None = None,
                  redo_mwpm: bool = False, resume: bool = False,
                  recovery: bool = False) -> dict:
    if shots < 1 or timing_shots < 1 or timing_repeats < 1:
        raise ValueError('shots, timing shots and repeats must be positive')
    if not distances or not ps or not bases or not profiles:
        raise ValueError('nonempty distances, rates and bases are required')
    output.mkdir(parents=True, exist_ok=True)
    cases = []
    errors = []
    summaries = []
    comparisons = []
    parameter_grid = list(product(bases, distances, ps, profiles))
    print(f'Stage 4: {len(parameter_grid)} cases; {shots} MWPM shots/case; '
          f'{min(shots, bp_shots or shots)} paired BP shots/case; '
          f'{bp_iterations} BP iterations; resume={resume}', flush=True)
    progress_file = output / 'study_progress.json'
    resumed_cases = 0
    for case_index, (basis, d, p, profile) in enumerate(parameter_grid, start=1):
        case_started = time.monotonic()
        case_id = f'{profile}_{basis}_d{d}_r{d}_p{p:.9g}'
        dest = output / 'runs' / (case_id + '.json')
        checkpoint = output / 'checkpoints' / (case_id + '.json')
        print(f'[Stage 4 {case_index}/{len(parameter_grid)}] START {case_id}', flush=True)
        # Stable, argument-order-invariant, well-within-range seed.
        offset = int(hashlib.sha256(case_id.encode()).hexdigest()[:12], 16)
        case_seed = (seed + offset) % (1 << 63)
        try:
            # Configuration fingerprint prevents inadvertently reusing outputs from
            # another run, another decoder implementation, or different parameters.
            config_spec = {
                'case': case_id, 'shots': shots, 'seed': case_seed,
                'approximate': approximate, 'recovery': recovery, 'bp_iterations': bp_iterations,
                'bp_shots': bp_shots, 'timing_shots': timing_shots,
                'timing_repeats': timing_repeats, 'replay': replay,
                'redo_mwpm': redo_mwpm,
                'source_sha256': {
                    module: hashlib.sha256((Path(__file__).parent / module).read_bytes()).hexdigest()
                    for module in ('experiment.py', 'belief_propagation.py', 'recovery.py', 'exact.py',
                                   'paired_analysis.py', 'studies.py', 'model.py',
                                   'evidence.py', 'stats.py')
                },
            }
            fingerprint = hashlib.sha256(json.dumps(config_spec, sort_keys=True).encode()).hexdigest()
            cached = None
            if resume and checkpoint.is_file():
                try:
                    cached = json.loads(checkpoint.read_text())
                    if cached.get('fingerprint') != fingerprint:
                        raise ValueError('protocol/source fingerprint differs')
                    audit_again = verify_case(dest, replay=replay, require_raw_hash=True,
                                              redo_mwpm=redo_mwpm)
                    if audit_again['status'] != 'PASS':
                        raise ValueError('case re-audit did not pass')
                    reanalysis = analyze_archived_case(dest)
                    if reanalysis != cached['case_result']['archive_reanalysis']:
                        raise ValueError('reanalysis differs from checkpoint')
                    print(f'[Stage 4 {case_index}/{len(parameter_grid)}] '
                          f'RESUMED after independent evidence checks', flush=True)
                except Exception as error:
                    print(f'[Stage 4 {case_index}/{len(parameter_grid)}] '
                          f'checkpoint not reusable: {error}; rerunning', flush=True)
                    cached = None
            if cached is not None:
                case_result = cached['case_result']
                case_result['audit'] = audit_again
                case_summaries = cached['summaries']
                case_comparisons = cached['comparisons']
                resumed_cases += 1
            else:
                config = MemoryConfig(distance=d, rounds=d, basis=basis, p=p, noise_profile=profile)
                settings = ExperimentSettings(
                    config, shots=shots, seed=case_seed, bayesian='auto',
                    approximate='on' if approximate else 'off',
                    bp_iterations=bp_iterations, bp_shots=bp_shots,
                    recovery='on' if recovery else 'off', recovery_shots=bp_shots,
                )
                result = run_experiment(settings, shots_path=dest.with_suffix('.npz'))
                dest.parent.mkdir(parents=True, exist_ok=True)
                result['evidence_files'] = save_model_files(dest, config, result['model'])
                result['model']['raw_shots_sha256'] = hashlib.sha256(
                    dest.with_suffix('.npz').read_bytes()
                ).hexdigest()
                save_json(dest, result)
                audit = verify_case(dest, replay=replay, require_raw_hash=True,
                                    redo_mwpm=redo_mwpm)

                reanalyzed = analyze_archived_case(dest)
                with np.load(dest.with_suffix('.npz'), allow_pickle=False) as archive:
                    syndrome_subset = archive['detection_events'][:timing_shots].copy()
                print(f'[Stage 4 {case_index}/{len(parameter_grid)}] '
                      f'shot archive audited; running decoder timing', flush=True)
                benchmark = time_decoders(config, syndrome_subset, use_bp=approximate,
                                          iterations=bp_iterations, repetitions=timing_repeats,
                                          use_recovery=recovery)
                case_result = {'case_id': case_id, 'report': str(dest),
                               'audit': audit, 'archive_reanalysis': reanalyzed,
                               'timings': benchmark}
                case_summaries = []
                case_comparisons = []
                for decoder, metrics in reanalyzed['rates'].items():
                    summary = result[{'loopy_bp': 'bp_approximate',
                                      'exact_bayesian': 'bayesian',
                                      'mwpm': 'mwpm', 'recovery': 'recovery'}[decoder]]
                    case_summaries.append({
                        'case_id': case_id, 'basis': basis, 'distance': d, 'rounds': d,
                        'p': p, 'decoder': decoder, 'seed': case_seed,
                        'shots': metrics['shots'], 'logical_failures': metrics['logical_failures'],
                        'logical_failure_rate': metrics['memory_failure_rate'],
                        'wilson_95_low': summary['wilson_95_low'],
                        'wilson_95_high': summary['wilson_95_high'],
                        'independent_round_equivalent': metrics['independent_round_equivalent'],
                        'status': 'completed', 'noise_profile': profile,
                    })
                for pair_name, metrics in reanalyzed['paired_comparisons'].items():
                    case_comparisons.append({'case_id': case_id, 'pair': pair_name, **metrics})
                checkpoint.parent.mkdir(parents=True, exist_ok=True)
                tmp = checkpoint.with_suffix('.tmp')
                save_json(tmp, {'fingerprint': fingerprint,
                                'case_result': case_result, 'summaries': case_summaries,
                                'comparisons': case_comparisons})
                tmp.replace(checkpoint)
            cases.append(case_result)
            summaries.extend(case_summaries)
            comparisons.extend(case_comparisons)
            print(f'[Stage 4 {case_index}/{len(parameter_grid)}] DONE {case_id} '
                  f'in {time.monotonic() - case_started:.1f}s', flush=True)
        except Exception as exc:
            errors.append({'case': case_id, 'error': f'{type(exc).__name__}: {exc}'})
            print(f'[Stage 4 {case_index}/{len(parameter_grid)}] ERROR '
                  f'{case_id}: {type(exc).__name__}: {exc}', flush=True)
        # Progress survives Ctrl+C even though the final comparative summary
        # deliberately does not claim completion until every case finishes.
        save_json(progress_file, {
            'expected_cases': len(parameter_grid), 'completed_cases': len(cases),
            'failed_cases': len(errors), 'resumed_cases': resumed_cases,
            'last_case': case_id, 'last_case_elapsed_seconds': time.monotonic() - case_started,
            'status': 'in_progress',
        })
    columns = ['case_id', 'basis', 'noise_profile', 'distance', 'rounds', 'p', 'decoder', 'seed',
               'shots', 'logical_failures', 'logical_failure_rate', 'wilson_95_low',
               'wilson_95_high', 'independent_round_equivalent', 'status']
    with (output / 'summary.csv').open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(summaries)
    if plot and summaries:
        from .plotting import plot_sweep
        for basis, profile in product(bases, profiles):
            subset = [row for row in summaries if row['basis'] == basis
                      and row['noise_profile'] == profile]
            if subset:
                subset_csv = output / f'plot_data_{basis}_{profile}.csv'
                with subset_csv.open('w', newline='') as file:
                    writer = csv.DictWriter(file, fieldnames=columns)
                    writer.writeheader()
                    writer.writerows(subset)
                plot_sweep(subset_csv, output / f'logical_rates_{basis}_{profile}.png')
    # Holm adjustment controls family-wise error within the prespecified
    # family of all Stage 4 paired tests in the full study matrix.
    from .paired_analysis import holm_adjust
    adjusted = holm_adjust([row['mcnemar_exact_two_sided_p'] for row in comparisons])
    for row, value in zip(comparisons, adjusted):
        row['mcnemar_holm_adjusted_p_study_family'] = value
        row['exploratory_after_multiplicity'] = True
    with (output / 'paired_comparisons.csv').open('w', newline='') as handle:
        fieldnames = ['case_id', 'pair', 'first_decoder', 'second_decoder', 'shots', 'first_only_wrong', 'second_only_wrong',
                      'both_wrong', 'both_correct', 'first_failure_rate', 'second_failure_rate',
                      'second_minus_first_failure_rate', 'mcnemar_exact_two_sided_p',
                      'mcnemar_holm_adjusted_p_study_family']
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(comparisons)
    manifest = {
        'stage': '3_and_4', 'schema_version': 1,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'environment': {'python': platform.python_version(), 'platform': platform.platform(),
                        'numpy': np.__version__,
                        'stim': _version_or_unavailable('stim'),
                        'pymatching': _version_or_unavailable('pymatching')},
        'protocol': {'shots_each': shots, 'distances': distances, 'ps': ps,
                     'bases': bases, 'noise_profiles': profiles, 'rounds': 'r=d', 'seed': seed,
                     'approximate_bp': approximate, 'experimental_recovery': recovery, 'bp_iterations': bp_iterations,
                     'timing_subset_shots': timing_shots, 'timing_repeats': timing_repeats,
                     'bp_shots_cap': bp_shots, 'independent_mwpm_redecode': redo_mwpm,
                     'seeded_replay': replay},
        'cases': cases, 'comparisons': comparisons, 'errors': errors,
        'resumed_cases': resumed_cases,
        'status': 'PASS' if cases and not errors else 'INCOMPLETE',
        'interpretation': [
            'Every paired comparison uses exactly the first N identical seeded Stim shots for both decoders.',
            'MWPM may use more shots than BP; denominators are recorded separately and never mixed.',
            'Comparison first/second decoder columns explicitly identify statistic order.',
            'Holm multiplicity adjustment covers all Stage 4 paired comparisons.',
            'Exact Bayesian only runs within its configured bit-state limit.',
            'BP logical marginals use a factorization approximation and are not optimal MAP.',
            'Recovery predictions enforce syndrome parity but are not exact Bayesian MAP.',
            'No decoder superiority or threshold is established solely by exploratory sweeps.',
            'Per-round equivalent assumes independent symmetric logical flips.',
            'Timing results compare different software implementations on one host; they are not hardware latency predictions.',
        ],
    }
    save_json(output / 'study_manifest.json', manifest)
    save_json(progress_file, {
        'expected_cases': len(parameter_grid), 'completed_cases': len(cases),
        'failed_cases': len(errors), 'resumed_cases': resumed_cases,
        'status': manifest['status'],
    })
    return manifest


def _version_or_unavailable(package: str) -> str:
    try:
        return version(package)
    except Exception:
        return 'not_available'


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='QEC Stages 3 and 4 scientific evidence')
    sub = parser.add_subparsers(dest='command', required=True)
    verify = sub.add_parser('verify', help='Re-audit saved experiment outputs')
    verify.add_argument('reports', nargs='+', type=Path, help='Run JSON files, or folders recursively')
    verify.add_argument('--replay', action='store_true', help='Resample bitwise using original Stim seed')
    verify.add_argument('--output', type=Path, default=Path('outputs/stage3/verification.json'))
    compare = sub.add_parser('benchmark', help='Execute paired Stage 4 distance/noise study')
    compare.add_argument('--output-dir', type=Path, default=Path('outputs/stage4'))
    compare.add_argument('--distances', nargs='+', type=int, default=[3, 5, 7])
    compare.add_argument('--ps', nargs='+', type=float, default=[.001, .003, .005, .01])
    compare.add_argument('--bases', nargs='+', choices=('x', 'z'), default=['x'])
    compare.add_argument('--profiles', nargs='+', choices=('circuit_level', 'data_only', 'measurement_only'), default=['circuit_level'])
    compare.add_argument('--shots', type=int, default=250)
    compare.add_argument('--seed', type=int, default=20261103)
    compare.add_argument('--approximate', choices=('on', 'off'), default='on')
    compare.add_argument('--bp-iterations', type=int, default=20)
    compare.add_argument('--bp-shots', type=int, default=256,
                         help='Matched first-N BP subset; MWPM uses all --shots')
    compare.add_argument('--redo-mwpm', action='store_true', help='Recompute MWPM from archived shots')
    compare.add_argument('--recovery', action='store_true',
                         help='Opt-in syndrome-consistent fault recovery on paired shots')
    compare.add_argument('--resume', action='store_true', help='Reuse completed, independently verified per-case checkpoints')
    compare.add_argument('--timing-shots', type=int, default=32)
    compare.add_argument('--timing-repeats', type=int, default=2)
    compare.add_argument('--replay', action='store_true')
    compare.add_argument('--no-plot', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'verify':
        reports = []
        for p in args.reports:
            reports.extend(sorted(p.rglob('*.json')) if p.is_dir() else [p])
        reports = [p for p in reports if p.name not in ('manifest.json', 'study_manifest.json', 'verification.json', 'evidence.json', 'study_progress.json') and p.parent.name != 'checkpoints']
        passed, failed = [], []
        for p in reports:
            try:
                passed.append(verify_case(p, replay=args.replay))
            except Exception as exc:
                failed.append({'file': str(p), 'error': f'{type(exc).__name__}: {exc}'})
        report = {'status': 'PASS' if passed and not failed else 'INCOMPLETE',
                  'passed': passed, 'failed': failed, 'requested_replay': args.replay}
        save_json(args.output, report)
        print(f'{len(passed)} passed, {len(failed)} failed; evidence: {args.output}')
        if report['status'] != 'PASS':
            raise SystemExit(1)
    else:
        manifest = run_benchmark(
            args.output_dir, distances=args.distances, ps=args.ps, bases=args.bases, profiles=args.profiles,
            shots=args.shots, seed=args.seed, approximate=args.approximate == 'on',
            timing_shots=args.timing_shots, timing_repeats=args.timing_repeats,
            bp_iterations=args.bp_iterations, bp_shots=args.bp_shots,
            redo_mwpm=args.redo_mwpm, replay=args.replay, plot=not args.no_plot,
            resume=args.resume, recovery=args.recovery,
        )
        print(f"{len(manifest['cases'])} passed; {len(manifest['errors'])} failed; "
              f"report: {args.output_dir / 'study_manifest.json'}")
        if manifest['status'] != 'PASS':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
