"""Offline evaluation against separately archived Stage 4 detector shots.

No Stim/PyMatching runtime required: recorded detector models and original MWPM
predictions are loaded with hashes. Accuracy claims are *held out from code
execution*, not held out from parameter or architecture selection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from datetime import datetime, timezone

import numpy as np

from .belief_propagation import BeliefPropagationDEMDecoder
from .exact import ExactBayesianDEMDecoder, Fault
from .paired_analysis import analyze_pair, holm_adjust
from .recovery import SyndromeRecoveryDecoder


_ERROR = re.compile(r'^error\(([^)]+)\)\s+(.*?)\s*$')
_TOKEN = re.compile(r'^[DL](\d+)$')


def parse_flattened_dem(path: Path, detectors: int, observables: int) -> list[Fault]:
    """Expand Stim DEM repeats and detector-index shifts without executing code.

    Coordinates and coordinate-only shifts carry metadata and do not alter
    syndrome parity. ``^`` separates pieces of a single Bernoulli mechanism.
    Invalid or unsupported instructions fail closed instead of being ignored.
    """
    lines = [(i, x.strip()) for i, x in enumerate(path.read_text().splitlines(), 1)
             if x.strip() and not x.lstrip().startswith('#')]
    cursor = 0

    def read_block(is_nested=False):
        nonlocal cursor
        nodes = []
        while cursor < len(lines):
            lineno, line = lines[cursor]
            cursor += 1
            if line == '}':
                if not is_nested:
                    raise ValueError(f'{path.name}:{lineno}: unmatched closing brace')
                return nodes
            match = re.fullmatch(r'repeat (\d+) \{', line)
            if match:
                count = int(match.group(1))
                if count > 5000:
                    raise ValueError('repeat exceeds safe expansion limit')
                nodes.append(('repeat', count, read_block(True)))
            else:
                if line.endswith('{'):
                    raise ValueError(f'{path.name}:{lineno}: unsupported block')
                nodes.append(('instruction', lineno, line))
        if is_nested:
            raise ValueError(f'{path.name}: unterminated repeat block')
        return nodes

    instructions = read_block()
    offset = 0
    faults = []

    def execute(nodes):
        nonlocal offset
        for kind, a, b in nodes:
            if kind == 'repeat':
                for _ in range(a):
                    execute(b)
                continue
            lineno, line = a, b
            if re.fullmatch(r'detector\([^)]*\) D\d+', line):
                continue
            if re.fullmatch(r'shift_detectors\([^)]*\) 0', line):
                continue
            shift = re.fullmatch(r'shift_detectors (\d+)', line)
            if shift:
                offset += int(shift.group(1))
                if offset > detectors:
                    raise ValueError(f'{path.name}:{lineno}: detector shift out of bounds')
                continue
            match = _ERROR.fullmatch(line)
            if not match:
                raise ValueError(f'{path.name}:{lineno}: unsupported DEM instruction: {line[:80]}')
            p = float(match.group(1))
            if not np.isfinite(p) or not 0 <= p <= 1:
                raise ValueError(f'{path.name}:{lineno}: invalid probability')
            mask = 0
            for token in match.group(2).split():
                if token == '^':
                    continue
                t = _TOKEN.fullmatch(token)
                if t is None:
                    raise ValueError(f'{path.name}:{lineno}: invalid target {token!r}')
                index = int(t.group(1))
                if token.startswith('D'):
                    absolute = index + offset
                    if absolute >= detectors:
                        raise ValueError(f'{path.name}:{lineno}: detector index out of bounds')
                    mask ^= 1 << absolute
                else:
                    if index >= observables:
                        raise ValueError(f'{path.name}:{lineno}: observable index out of bounds')
                    mask ^= 1 << (detectors + index)
            faults.append(Fault(p, mask))
            if len(faults) > 200000:
                raise ValueError('DEM expansion exceeds safe fault limit')

    execute(instructions)
    if not faults:
        raise ValueError(f'{path.name}: empty detector-error model')
    return faults


def projected_exact_check(faults: list[Fault], detectors: int, observables: int, *,
                          projected_detectors: int = 4, iterations: int = 20) -> dict:
    """Exactly solvable *marginal projection*, not full physical-data validation.

    Unselected detectors are marginalized away by deleting them from event masks.
    The projected model retains all fault mechanisms and logical observables.
    The marginal predictions only condition on selected detectors, so they must
    never be evaluated against full-detector Stage 4 accuracy as if equivalent.
    """
    take = min(detectors, projected_detectors)
    if take + observables > 16:
        raise ValueError('projected state is too large')
    truncated = []
    for f in faults:
        mask = (f.mask & ((1 << take) - 1)) | ((f.mask >> detectors) << take)
        if mask:
            truncated.append(Fault(f.probability, mask))
    oracle = ExactBayesianDEMDecoder(take, observables, truncated, max_bits=16)
    bp = BeliefPropagationDEMDecoder(take, observables, truncated, iterations=iterations)
    recovery = SyndromeRecoveryDecoder(take, observables, truncated, iterations=iterations)
    patterns = np.array([[(i >> j) & 1 for j in range(take)] for i in range(1 << take)],
                        dtype=np.uint8)
    reachable = np.flatnonzero(oracle.evidence > 1e-14)
    if not len(reachable):
        raise ValueError('no reachable projected syndromes')
    weights = oracle.evidence[reachable]
    weights /= weights.sum()
    exact_preds = oracle.decode_batch(patterns[reachable])
    bp_preds = bp.decode_batch(patterns[reachable])
    recovered_preds = recovery.decode_batch(patterns[reachable])
    return {
        'selected_detectors': take,
        'total_detectors': detectors,
        'num_projected_faults': len(truncated),
        'exact_oracle_states': 1 << (take + observables),
        'reachable_syndromes': len(reachable),
        'weighted_bp_exact_map_agreement': float(np.sum(weights * np.all(bp_preds == exact_preds, axis=1))),
        'weighted_recovery_exact_map_agreement': float(np.sum(weights * np.all(recovered_preds == exact_preds, axis=1))),
        'scope': 'small marginal projection of actual archived DEM, not exact full-surface-code inference',
    }


def evaluate_case(case_json: Path, *, shots: int = 256, iterations: int = 20,
                  projected_detectors: int = 4) -> dict:
    if shots < 1:
        raise ValueError('shots must be >= 1')
    meta = json.loads(case_json.read_text())
    raw_path = case_json.with_suffix('.npz')
    dem_path = case_json.with_suffix('.dem')
    if not raw_path.is_file() or not dem_path.is_file():
        raise FileNotFoundError('expected matching .npz and .dem files')
    raw_hash = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    dem_hash = hashlib.sha256(dem_path.read_bytes()).hexdigest()
    saved_hash = meta.get('model', {}).get('raw_shots_sha256')
    if saved_hash is not None and saved_hash != raw_hash:
        raise ValueError('archived Stage 4 shot hash mismatch')
    with np.load(raw_path, allow_pickle=False) as data:
        required = ('detection_events', 'actual_observables', 'mwpm_predictions', 'bp_predictions')
        if any(k not in data.files for k in required):
            raise ValueError('incomplete Stage 4 shot archive')
        full_syn, truth, mwpm, old_bp = (data[k] for k in required)
    n = min(shots, len(old_bp), len(full_syn))
    if n == 0 or full_syn.shape[0] != len(truth) or full_syn.shape[0] != len(mwpm):
        raise ValueError('invalid input array shape')
    if (old_bp.shape[1:] != truth.shape[1:] or mwpm.shape[1:] != truth.shape[1:]):
        raise ValueError('incompatible predictions and truth')
    if any(np.any((arr != 0) & (arr != 1)) for arr in (full_syn, truth, old_bp, mwpm)):
        raise ValueError('nonbinary detector evidence')
    detectors, observables = full_syn.shape[1], truth.shape[1]
    faults = parse_flattened_dem(dem_path, detectors, observables)
    recovered = SyndromeRecoveryDecoder(detectors, observables, faults, iterations=iterations)
    prediction, diagnostics = recovered.decode_batch(full_syn[:n], with_details=True)
    wrong = lambda x: np.any(x != truth[:n], axis=1)
    mwpm_wrong, old_wrong, recovered_wrong = map(wrong, (mwpm[:n], old_bp[:n], prediction))
    against_mwpm = analyze_pair(mwpm_wrong, recovered_wrong, bootstrap_repeats=500)
    against_old = analyze_pair(old_wrong, recovered_wrong, bootstrap_repeats=500)
    exact_projection = projected_exact_check(faults, detectors, observables,
                                              projected_detectors=projected_detectors,
                                              iterations=iterations)
    return {
        'case_id': case_json.stem,
        'shots_compared': n,
        'hashes': {'npz_sha256': raw_hash, 'dem_sha256': dem_hash,
                   'matches_recorded_raw_hash': saved_hash is None or saved_hash == raw_hash},
        'prior_model': 'Stim flattened independent-error DEM',
        'algorithm': 'BP-guided syndrome-consistent recovery with prior-likelihood selection',
        'counts': {'mwpm_wrong': int(sum(mwpm_wrong)), 'old_bp_wrong': int(sum(old_wrong)),
                   'recovery_wrong': int(sum(recovered_wrong))},
        'paired_recovery_vs_mwpm': against_mwpm,
        'paired_recovery_vs_old_bp': against_old,
        'diagnostics': diagnostics,
        'projected_exact': exact_projection,
        'caution': 'Recovery posterior is not exact; prior-weighted fault MAP is not logical MAP.',
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description='Audit new decoder against preserved Stage 4 shots')
    p.add_argument('--archive-root', type=Path, required=True,
                   help='directory containing Stage 4 runs/*.json/.npz/.dem')
    p.add_argument('--output', type=Path, default=Path('outputs/recovery_eval'))
    p.add_argument('--limit', type=int, default=0, help='0 evaluates all cases')
    p.add_argument('--shots', type=int, default=256)
    p.add_argument('--iterations', type=int, default=20)
    p.add_argument('--projected-detectors', type=int, default=4)
    args = p.parse_args(argv)
    cases = sorted(args.archive_root.glob('runs/*.json'))
    if args.limit:
        cases = cases[:args.limit]
    if not cases:
        raise SystemExit('no Stage 4 archived run JSON files found')
    args.output.mkdir(parents=True, exist_ok=True)
    evaluated, failed = [], []
    for i, case in enumerate(cases, start=1):
        try:
            item = evaluate_case(case, shots=args.shots, iterations=args.iterations,
                                 projected_detectors=args.projected_detectors)
            evaluated.append(item)
            print(f'[{i}/{len(cases)}] {case.stem}: MWPM={item["counts"]["mwpm_wrong"]}, '
                  f'old BP={item["counts"]["old_bp_wrong"]}, '
                  f'recovery={item["counts"]["recovery_wrong"]}', flush=True)
        except Exception as error:
            failed.append({'case': case.stem, 'error': repr(error)})
            print(f'[{i}/{len(cases)}] FAILED {case.stem}: {error}', flush=True)
    adjusted = holm_adjust([item['paired_recovery_vs_mwpm']['mcnemar_exact_two_sided_p']
                            for item in evaluated])
    for item, p_adjusted in zip(evaluated, adjusted):
        item['holm_p_within_evaluated_cases'] = p_adjusted
    result = {
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if evaluated and not failed else 'INCOMPLETE',
        'n_cases': len(evaluated), 'requested_cases': len(cases),
        'shots_requested_per_case': args.shots,
        'cases': evaluated, 'failures': failed,
        'scope': 'Offline analysis of separately preserved Stage 4 model/shots; compare only the same archived shot indices',
        'limitations': [
            'Original Stage 4 data were used during development: not a held-out validation set.',
            'Projected exact checks validate only a marginal submodel, not the full DEM.',
            'Candidate recovery is not exact Bayesian logical MAP.',
            'Many cases and sparse errors need more shots before generalizing.',
        ],
    }
    (args.output/'evaluation.json').write_text(json.dumps(result, indent=2) + '\n')
    with (args.output/'case_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['case_id', 'shots', 'mwpm_errors',
                 'old_bp_errors', 'recovery_errors', 'recovery_minus_mwpm',
                 'paired_exact_p', 'holm_p', 'recovery_consistent'])
        writer.writeheader()
        for item in evaluated:
            writer.writerow({
                'case_id': item['case_id'], 'shots': item['shots_compared'],
                'mwpm_errors': item['counts']['mwpm_wrong'],
                'old_bp_errors': item['counts']['old_bp_wrong'],
                'recovery_errors': item['counts']['recovery_wrong'],
                'recovery_minus_mwpm': item['paired_recovery_vs_mwpm']['second_minus_first_failure_rate'],
                'paired_exact_p': item['paired_recovery_vs_mwpm']['mcnemar_exact_two_sided_p'],
                'holm_p': item['holm_p_within_evaluated_cases'],
                'recovery_consistent': item['diagnostics']['fraction_syndrome_consistent'],
            })
    if failed:
        raise SystemExit(f'{len(failed)} cases failed; see evaluation.json')


if __name__ == '__main__':
    main()
