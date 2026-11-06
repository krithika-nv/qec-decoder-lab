"""Pre-registered, independently sampled MWPM-versus-recovery validation.

PLAN must precede sampling. A SHA-256 source manifest freezes all Python
modules; RUN refuses changed code or altered protocols. The sampled shots are
new, distinct from the historical Stage 4 development archive. AUDIT computes
paired statistics again from raw arrays. Passing it never proves superiority.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import secrets

import numpy as np

from .model import MemoryConfig
from .paired_analysis import analyze_pair, holm_adjust
from .stats import summarize_errors


PROTOCOL_FILE = 'locked_protocol.json'
SCHEMA = 'qec-frozen-validation-v2'


def _canonical_bytes(obj) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda: stream.read(1 << 20), b''):
            h.update(part)
    return h.hexdigest()


def _source_manifest() -> dict[str, str]:
    folder = Path(__file__).resolve().parent
    return {path.name: _digest(path) for path in sorted(folder.glob('*.py'))}


def _case_id(spec: dict) -> str:
    return (f'{spec["noise_profile"]}_{spec["basis"]}_d{spec["distance"]}'
            f'_r{spec["rounds"]}_p{spec["p"]:.9g}')


def _case_grid(profile: str) -> list[dict]:
    if profile == 'smoke':
        distances, rates = (3,), (0.01,)
    elif profile == 'targeted':
        distances, rates = (3, 5), (0.005, 0.01)
    elif profile == 'focused':
        distances, rates = (3, 5, 7), (0.003, 0.01)
    elif profile == 'expanded':
        distances, rates = (3, 5, 7), (0.003, 0.005, 0.01)
    else:
        raise ValueError('profile must be smoke, targeted, focused or expanded')
    return [{'distance': d, 'rounds': d, 'basis': basis, 'p': p,
             'noise_profile': 'circuit_level'}
            for basis in ('x', 'z') for d in distances for p in rates]


def create_protocol(root: Path, *, profile: str = 'targeted', shots: int = 10000,
                    paired_shots: int = 2048, bp_iterations: int = 20,
                    damping: float = 0.3, refinement_budget: int = 64,
                    bootstrap_repeats: int = 2000,
                    check_update: str = 'vectorized',
                    basis_update: str = 'python',
                    strategy: str = 'recovery',
                    gate_fraction: float | None = None) -> dict:
    if shots < 1 or not (1 <= paired_shots <= shots):
        raise ValueError('shots and paired_shots must be positive with paired_shots <= shots')
    if bp_iterations < 1 or refinement_budget < 0 or not 0 <= damping < 1:
        raise ValueError('invalid frozen decoder settings')
    if check_update not in ('vectorized', 'legacy', 'compiled', 'minsum'):
        raise ValueError('unsupported BP check-update engine')
    if basis_update not in ('python', 'compiled'):
        raise ValueError('unsupported basis update engine')
    if strategy not in ('recovery', 'hybrid'):
        raise ValueError('strategy must be recovery or hybrid')
    if strategy == 'hybrid':
        if gate_fraction is None or not np.isfinite(gate_fraction) or not 0 <= gate_fraction <= 1:
            raise ValueError('hybrid strategy requires a fixed detector gate fraction in [0,1]')
    elif gate_fraction is not None:
        raise ValueError('gate_fraction is only valid for the hybrid strategy')
    if bootstrap_repeats < 10:
        raise ValueError('bootstrap_repeats must be >= 10')
    root.mkdir(parents=True, exist_ok=True)
    path = root / PROTOCOL_FILE
    if path.exists():
        raise FileExistsError('protocol already locked; do not overwrite to change a held-out study')
    if any(root.iterdir()):
        raise FileExistsError('use an empty new output directory to prevent mixing prior results')
    source = _source_manifest()
    items = _case_grid(profile)
    config = {
        'schema': SCHEMA,
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'intent': 'NEW held-out simulation, not historical Stage 4 development shots',
        'profile': profile, 'case_specs': items, 'shots_per_case_mwpm': shots,
        'paired_shots_per_case': paired_shots,
        'algorithm': ('selective_mwpm_plus_syndrome_recovery' if strategy == 'hybrid'
                      else 'syndrome_consistent_prior_scored_bp_guided_recovery'),
        'strategy': strategy, 'gate_fraction': gate_fraction, 'basis_update': basis_update,
        'check_update': check_update, 'implementation': 'post_optimization',
        'iterations': bp_iterations, 'damping': damping,
        'refinement_budget': refinement_budget,
        'bootstrap_repeats': bootstrap_repeats,
        'primary_analysis': {
            'unit': 'independent circuit-level simulation shot',
            'paired_metric': 'recovery_minus_mwpm_logical_failure_rate',
            'per_case_test': 'two-sided exact McNemar',
            'multiple_testing': 'Holm correction across ALL predeclared cases',
            'alpha': 0.05,
            'missing_case_policy': 'incomplete, no replacement or result exclusion',
            'pooled_analysis': 'descriptive only; heterogeneous conditions cannot be pooled into a universal claim',
            'zero_failure_policy': 'report binomial uncertainty, never claim zero underlying error',
        },
        'randomization': 'new cryptographic PRNG seed generated per case after locking; never reuses Stage 4 seeds',
        'source_sha256': source,
        'dependency_versions_collected_at_run': True,
    }
    config['protocol_sha256'] = hashlib.sha256(_canonical_bytes(config)).hexdigest()
    path.write_bytes(_canonical_bytes(config))
    return config


def load_locked_protocol(root: Path, *, verify_source: bool = True) -> dict:
    path = root / PROTOCOL_FILE
    protocol = json.loads(path.read_text())
    digest = protocol.pop('protocol_sha256', None)
    actual = hashlib.sha256(_canonical_bytes(protocol)).hexdigest()
    protocol['protocol_sha256'] = digest
    if digest != actual or protocol.get('schema') != SCHEMA:
        raise ValueError('locked protocol was altered or has unexpected schema')
    if verify_source and protocol['source_sha256'] != _source_manifest():
        raise ValueError('frozen decoder source hash changed after protocol registration; start a distinct study')
    if protocol.get('strategy', 'recovery') not in ('recovery', 'hybrid'):
        raise ValueError('unrecognized frozen strategy')
    if protocol.get('strategy') == 'hybrid' and not 0 <= protocol['gate_fraction'] <= 1:
        raise ValueError('invalid frozen hybrid gate fraction')
    if protocol.get('basis_update', 'python') not in ('python', 'compiled'):
        raise ValueError('invalid frozen basis update')
    if len(set(_case_id(x) for x in protocol['case_specs'])) != len(protocol['case_specs']):
        raise ValueError('duplicate case IDs in protocol')
    for spec in protocol['case_specs']:
        MemoryConfig(**spec)
    return protocol


def summarize_case(syndromes: np.ndarray, truth: np.ndarray, mwpm: np.ndarray,
                   recovery: np.ndarray, *, seed: int, bootstrap_repeats: int) -> dict:
    if (syndromes.ndim != 2 or truth.ndim != 2 or mwpm.shape != truth.shape or
            recovery.ndim != 2 or recovery.shape[1:] != truth.shape[1:] or
            recovery.shape[0] < 1 or recovery.shape[0] > truth.shape[0] or
            len(syndromes) != len(truth)):
        raise ValueError('invalid matched-shot array shapes')
    for x in (syndromes, truth, mwpm, recovery):
        if np.any((x != 0) & (x != 1)):
            raise ValueError('nonbinary experiment evidence')
    n = recovery.shape[0]
    first = np.any(mwpm[:n] != truth[:n], axis=1)
    second = np.any(recovery != truth[:n], axis=1)
    return {
        'mwpm_all_shots': summarize_errors(np.any(mwpm != truth, axis=1)),
        'mwpm_paired_shots': summarize_errors(first),
        'recovery_paired_shots': summarize_errors(second),
        'recovery_minus_mwpm': analyze_pair(
            first, second, seed=seed % (1 << 32), bootstrap_repeats=bootstrap_repeats),
        'selection': 'first paired_shots of new independent, seeded detector sample; same shots for both decoders',
    }


def _case_paths(root: Path, case_id: str) -> dict[str, Path]:
    folder = root / 'cases'
    return {ext: folder / (case_id + '.' + ext) for ext in ('json', 'npz', 'stim', 'dem')}


def _decode_experimental(protocol: dict, dem, syndromes: np.ndarray,
                         mwpm: np.ndarray, *, progress: bool = False):
    """Run locked full recovery or gated hybrid on the same MWPM sample."""
    from .recovery import SyndromeRecoveryDecoder
    from .hybrid import select_complex_syndromes
    decoder = SyndromeRecoveryDecoder.from_dem(
        dem, iterations=protocol['iterations'], damping=protocol['damping'],
        refinement_budget=protocol['refinement_budget'],
        check_update=protocol['check_update'],
        basis_update=protocol.get('basis_update', 'python'))
    if protocol.get('strategy') == 'hybrid':
        selected = select_complex_syndromes(syndromes, protocol['gate_fraction'])
        recovered = mwpm.copy()
        if np.any(selected):
            new_pred, details = decoder.decode_batch(syndromes[selected], with_details=True,
                                                     progress_callback=None)
            recovered[selected] = new_pred
        else:
            details = {'unique_syndromes': 0, 'fraction_syndrome_consistent':1.0,
                       'fraction_bp_converged': None}
        details.update({'selected_shots':int(selected.sum()),
                        'selected_fraction':float(np.mean(selected)),
                        'selected_gate':'active_detector_fraction',
                        'gate_fraction':protocol['gate_fraction'],
                        'strategy':'hybrid'})
        return recovered, details
    if progress:
        def on_progress(done, total):
            print(f'    recovery unique syndromes {done}/{total}', flush=True)
    else:
        on_progress = None
    return decoder.decode_batch(
        syndromes, with_details=True, progress_callback=on_progress,
        progress_every=max(1, len(syndromes)//10) if progress else 0)


def _capture_case(root: Path, protocol: dict, spec: dict, seed: int, progress: bool) -> dict:
    import stim
    import pymatching
    from importlib.metadata import version
    import platform
    import sys

    config = MemoryConfig(**spec)
    circuit = config.circuit()
    dem = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(dem)
    sampler = circuit.compile_detector_sampler(seed=seed)
    shots = protocol['shots_per_case_mwpm']
    paired = protocol['paired_shots_per_case']
    syndromes, truth = sampler.sample(shots=shots, separate_observables=True)
    syndromes = np.asarray(syndromes, dtype=np.uint8)
    truth = np.asarray(truth, dtype=np.uint8)
    mwpm = np.asarray(matching.decode_batch(syndromes), dtype=np.uint8)
    recovered, details = _decode_experimental(
        protocol, dem, syndromes[:paired], mwpm[:paired], progress=progress)
    recovered = np.asarray(recovered, dtype=np.uint8)
    metrics = summarize_case(syndromes, truth, mwpm, recovered,
                             seed=seed, bootstrap_repeats=protocol['bootstrap_repeats'])
    case_id = _case_id(spec)
    paths = _case_paths(root, case_id)
    paths['npz'].parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(paths['npz'], detection_events=syndromes,
                        actual_observables=truth, mwpm_predictions=mwpm,
                        recovery_predictions=recovered)
    paths['stim'].write_text(str(circuit) + '\n')
    paths['dem'].write_text(str(dem) + '\n')
    record = {
        'schema': SCHEMA, 'case_id': case_id, 'spec': spec,
        'protocol_sha256': protocol['protocol_sha256'],
        'seed': seed, 'n_shots': shots, 'n_paired': paired,
        'detectors': int(dem.num_detectors), 'observables': int(dem.num_observables),
        'metrics': metrics,
        'recovery_diagnostics': details,
        'versions': {'stim': stim.__version__, 'pymatching': version('pymatching'),
                     'numpy': np.__version__, 'python': sys.version.split()[0],
                     'platform': platform.platform(),
                     'numba': version('numba') if protocol['check_update'] in ('compiled','minsum')
                              or protocol.get('basis_update') == 'compiled' else None},
        'artifact_sha256': {key: _digest(paths[key]) for key in ('npz', 'stim', 'dem')},
        'status': 'COMPLETE', 'scope': 'new independently seeded held-out simulated shots',
    }
    paths['json'].write_bytes(_canonical_bytes(record))
    return record


def audit_case(root: Path, protocol: dict, spec: dict, *, redo_mwpm: bool = False,
               redo_recovery: bool = False, replay_seed: bool = False) -> dict:
    case_id = _case_id(spec)
    paths = _case_paths(root, case_id)
    if any(not p.is_file() for p in paths.values()):
        raise FileNotFoundError(f'{case_id}: case evidence incomplete')
    item = json.loads(paths['json'].read_text())
    if (item.get('status') != 'COMPLETE' or item.get('schema') != SCHEMA or
            item.get('spec') != spec or item.get('case_id') != case_id or
            item.get('protocol_sha256') != protocol['protocol_sha256'] or
            item.get('n_shots') != protocol['shots_per_case_mwpm'] or
            item.get('n_paired') != protocol['paired_shots_per_case']):
        raise ValueError(f'{case_id}: case and locked protocol mismatch')
    for name in ('npz', 'stim', 'dem'):
        if item['artifact_sha256'].get(name) != _digest(paths[name]):
            raise ValueError(f'{case_id}: {name} checksum mismatch')
    with np.load(paths['npz'], allow_pickle=False) as raw:
        if set(raw.files) != {'detection_events','actual_observables','mwpm_predictions','recovery_predictions'}:
            raise ValueError(f'{case_id}: array set mismatch')
        syn, truth, mwpm, recovery = [raw[name].copy() for name in (
            'detection_events', 'actual_observables', 'mwpm_predictions', 'recovery_predictions')]
    if (syn.shape != (item['n_shots'], item['detectors']) or
            truth.shape != (item['n_shots'], item['observables']) or
            recovery.shape != (item['n_paired'], item['observables'])):
        raise ValueError(f'{case_id}: raw evidence dimensions mismatch')
    if not 0 <= int(item['seed']) < (1 << 63):
        raise ValueError('invalid recorded sampling seed')
    if protocol.get('strategy') == 'hybrid':
        from .hybrid import select_complex_syndromes
        selection = select_complex_syndromes(syn[:item['n_paired']], protocol['gate_fraction'])
        if not np.array_equal(recovery[~selection], mwpm[:item['n_paired']][~selection]):
            raise ValueError(f'{case_id}: gated-off shots differ from saved MWPM results')
        details = item.get('recovery_diagnostics', {})
        if details.get('selected_shots') != int(selection.sum()) or details.get('gate_fraction') != protocol['gate_fraction']:
            raise ValueError(f'{case_id}: hybrid gate metadata inconsistent')
    recalculated = summarize_case(syn, truth, mwpm, recovery,
                                  seed=int(item['seed']),
                                  bootstrap_repeats=protocol['bootstrap_repeats'])
    if _canonical_bytes(recalculated) != _canonical_bytes(item['metrics']):
        raise ValueError(f'{case_id}: recorded statistics do not reconcile with raw shots')
    if redo_mwpm or redo_recovery or replay_seed:
        import stim
        import pymatching
        circuit = stim.Circuit(paths['stim'].read_text())
        dem = circuit.detector_error_model(decompose_errors=True)
        if str(dem).strip() != paths['dem'].read_text().strip():
            raise ValueError(f'{case_id}: circuit and detector model are inconsistent')
        if redo_mwpm:
            actual = pymatching.Matching.from_detector_error_model(dem).decode_batch(syn)
            if not np.array_equal(actual, mwpm):
                raise ValueError(f'{case_id}: MWPM replay differs')
        if redo_recovery:
            from .recovery import SyndromeRecoveryDecoder
            predicted, _ = _decode_experimental(
                protocol, dem, syn[:item['n_paired']], mwpm[:item['n_paired']])
            if not np.array_equal(predicted, recovery):
                raise ValueError(f'{case_id}: experimental decoder replay differs')
        if replay_seed:
            regenerated = circuit.compile_detector_sampler(seed=int(item['seed'])).sample(
                shots=item['n_shots'], separate_observables=True)
            if not (np.array_equal(regenerated[0], syn) and np.array_equal(regenerated[1], truth)):
                raise ValueError(f'{case_id}: deterministic Stim sampling replay differs')
    return {'case_id': case_id, 'status': 'PASS', 'shots': item['n_shots'],
            'paired_shots': item['n_paired'], 'seed': item['seed'],
            'error_difference': recalculated['recovery_minus_mwpm']['second_minus_first_failure_rate'],
            'paired_p': recalculated['recovery_minus_mwpm']['mcnemar_exact_two_sided_p'],
            'redo_mwpm': redo_mwpm, 'redo_recovery': redo_recovery, 'replay_seed': replay_seed}


def audit_study(root: Path, *, require_source: bool = True, redo_mwpm: bool = False,
                redo_recovery: bool = False, replay_seed: bool = False,
                write_report: bool = True) -> dict:
    protocol = load_locked_protocol(root, verify_source=require_source)
    records, failures, seeds = [], [], []
    for spec in protocol['case_specs']:
        try:
            result = audit_case(root, protocol, spec, redo_mwpm=redo_mwpm,
                                redo_recovery=redo_recovery, replay_seed=replay_seed)
            records.append(result)
            seeds.append(result['seed'])
        except Exception as error:
            failures.append({'case': _case_id(spec), 'error': f'{type(error).__name__}: {error}'})
    if len(set(seeds)) != len(seeds):
        failures.append({'case': 'seed_uniqueness', 'error': 'duplicate sampling seed'})
    adjusted = holm_adjust([row['paired_p'] for row in records])
    for row, p in zip(records, adjusted):
        row['holm_adjusted_p'] = p
        row['recovery_significantly_better_than_mwpm'] = bool(p < .05 and row['error_difference'] < 0)
    report = {
        'status': 'PASS' if not failures and len(records) == len(protocol['case_specs']) else 'INCOMPLETE',
        'schema': SCHEMA, 'protocol_sha256': protocol['protocol_sha256'],
        'cases_expected': len(protocol['case_specs']), 'cases_verified': len(records),
        'shots_per_case': protocol['shots_per_case_mwpm'],
        'paired_shots_per_case': protocol['paired_shots_per_case'],
        'source_verified': require_source, 'records': records, 'failures': failures,
        'interpretation': ('Integrity/paired-analysis check only. Significance is case-specific with Holm correction. '
                           'A passed audit or pooled difference does not demonstrate a generally superior decoder.'),
    }
    if write_report:
        (root / 'heldout_audit.json').write_bytes(_canonical_bytes(report))
        with (root / 'heldout_case_comparison.csv').open('w', newline='') as file:
            fields = ['case_id', 'status', 'shots', 'paired_shots', 'error_difference',
                      'paired_p', 'holm_adjusted_p', 'recovery_significantly_better_than_mwpm']
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for row in records:
                writer.writerow({key: row[key] for key in fields})
    return report


def run_study(root: Path, *, resume: bool = True, progress: bool = True) -> dict:
    protocol = load_locked_protocol(root, verify_source=True)
    for index, spec in enumerate(protocol['case_specs'], start=1):
        case_id = _case_id(spec)
        case_file = _case_paths(root, case_id)['json']
        if case_file.exists():
            if not resume:
                raise FileExistsError(f'{case_id} already exists; do not overwrite or select convenient outcomes')
            audit_case(root, protocol, spec)
            if progress:
                print(f'[heldout {index}/{len(protocol["case_specs"])}] VERIFIED checkpoint {case_id}', flush=True)
            continue
        paths = _case_paths(root, case_id)
        if any(path.is_file() for path in paths.values()):
            raise FileExistsError(
                f'{case_id}: partial evidence exists without a verified checkpoint; '
                'quarantine incomplete files before restarting, never overwrite prior samples')
        # Entropy is drawn only after the immutable protocol and source are checked.
        seed = secrets.randbelow((1 << 63) - 1) + 1
        if progress:
            print(f'[heldout {index}/{len(protocol["case_specs"])}] NEW SAMPLE {case_id}', flush=True)
        _capture_case(root, protocol, spec, seed, progress)
        audit_case(root, protocol, spec)
        if progress:
            print(f'[heldout {index}/{len(protocol["case_specs"])}] CHECKPOINT VERIFIED', flush=True)
    result = audit_study(root)
    if result['status'] != 'PASS':
        raise RuntimeError('held-out protocol did not pass all evidence gates')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    create = commands.add_parser('plan', help='freeze source, hyperparameters, cases and statistical analysis before sampling')
    create.add_argument('--root', type=Path, default=Path('outputs/heldout'))
    create.add_argument('--profile', choices=['smoke','targeted','focused','expanded'], default='targeted')
    create.add_argument('--shots', type=int, default=10000)
    create.add_argument('--paired-shots', type=int, default=2048)
    create.add_argument('--iterations', type=int, default=20)
    create.add_argument('--damping', type=float, default=.3)
    create.add_argument('--refinement-budget', type=int, default=64)
    create.add_argument('--bootstrap-repeats', type=int, default=2000)
    create.add_argument('--check-update', choices=['vectorized', 'legacy', 'compiled', 'minsum'], default='vectorized')
    create.add_argument('--basis-update', choices=['python','compiled'], default='python')
    create.add_argument('--strategy', choices=['recovery','hybrid'], default='recovery')
    create.add_argument('--gate-fraction', type=float)
    run = commands.add_parser('run', help='sample NEW held-out shots under a frozen protocol')
    run.add_argument('--root', type=Path, default=Path('outputs/heldout'))
    run.add_argument('--no-resume', action='store_true')
    audit = commands.add_parser('audit', help='recompute every check from archived raw evidence')
    audit.add_argument('--root', type=Path, default=Path('outputs/heldout'))
    audit.add_argument('--redo-mwpm', action='store_true')
    audit.add_argument('--redo-recovery', action='store_true')
    audit.add_argument('--replay-seed', action='store_true')
    args = parser.parse_args(argv)
    if args.command == 'plan':
        result = create_protocol(args.root, profile=args.profile, shots=args.shots,
                                 paired_shots=args.paired_shots, bp_iterations=args.iterations,
                                 damping=args.damping, refinement_budget=args.refinement_budget,
                                 bootstrap_repeats=args.bootstrap_repeats,
                                 check_update=args.check_update, basis_update=args.basis_update,
                                 strategy=args.strategy, gate_fraction=args.gate_fraction)
        print(f'FROZEN: {len(result["case_specs"])} preregistered cases; '
              f'protocol sha256 {result["protocol_sha256"]}')
    elif args.command == 'run':
        result = run_study(args.root, resume=not args.no_resume)
        print(f'{result["status"]}: {result["cases_verified"]}/{result["cases_expected"]} held-out cases')
    else:
        result = audit_study(args.root, redo_mwpm=args.redo_mwpm,
                             redo_recovery=args.redo_recovery, replay_seed=args.replay_seed)
        print(json.dumps({'status': result['status'], 'cases_verified': result['cases_verified'],
                          'failures': result['failures']}, indent=2))
        if result['status'] != 'PASS':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
