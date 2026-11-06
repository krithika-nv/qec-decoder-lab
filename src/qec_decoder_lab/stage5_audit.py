"""Independent, fail-closed audit of published-protocol timing artifacts.

Uses only NumPy and the standard library for the default audit. Optional
--replay requires Stim and PyMatching and decodes the archived syndromes again.
The audit does not claim cross-machine speed equality or independent CPU timing.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median

import numpy as np

PAPER_DISTANCES = (5, 7, 9, 13, 17, 23, 29, 39, 50)


class Stage5EvidenceError(ValueError):
    pass


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(2**20), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not (np.isfinite(actual) and np.isfinite(expected) and
            np.isclose(actual, expected, rtol=1e-9, atol=1e-12)):
        raise Stage5EvidenceError(f'{label}: {actual!r} != {expected!r}')


def _require_file(directory: Path, filename: str, digest: str | None = None) -> Path:
    if Path(filename).name != filename:
        raise Stage5EvidenceError(f'Unsafe artifact path: {filename}')
    path = directory / filename
    if not path.is_file():
        raise Stage5EvidenceError(f'Missing artifact: {path}')
    if digest is not None and sha256(path) != digest:
        raise Stage5EvidenceError(f'SHA256 mismatch: {path}')
    return path


def _verify_raw(directory: Path, row: dict, *, redo_decode: bool = False,
                replay_sampling: bool = False, chunk: int = 256) -> dict:
    d, shots, detectors, obs = (
        int(row['distance']), int(row['shots']), int(row['num_detectors']),
        int(row['num_observables']),
    )
    raw = _require_file(directory, row['raw_shots_file'], row['raw_shots_sha256'])
    circuit = _require_file(directory, f'd{d}.stim', row['circuit_sha256'])
    dem = _require_file(directory, f'd{d}.dem', row['dem_sha256'])
    if detectors < 1 or obs < 1 or shots < 1:
        raise Stage5EvidenceError(f'd={d}: invalid detector, observable or shot count')
    with np.load(raw, allow_pickle=False) as archive:
        required = {'detectors_packed', 'actual_observables', 'mwpm_predictions',
                    'num_detectors', 'num_observables', 'shots', 'seed'}
        if not required.issubset(archive.files):
            raise Stage5EvidenceError(f'd={d}: raw archive missing {required - set(archive.files)}')
        packed = archive['detectors_packed']
        truth = archive['actual_observables']
        predictions = archive['mwpm_predictions']
        if (packed.dtype != np.uint8 or packed.shape != (shots, (detectors + 7)//8) or
            truth.shape != (shots, obs) or predictions.shape != (shots, obs) or
            not np.all((truth == 0) | (truth == 1)) or
            not np.all((predictions == 0) | (predictions == 1))):
            raise Stage5EvidenceError(f'd={d}: malformed raw arrays')
        if (int(archive['num_detectors']) != detectors or
            int(archive['num_observables']) != obs or int(archive['shots']) != shots or
            int(archive['seed']) != int(row['sampling_seed'])):
            raise Stage5EvidenceError(f'd={d}: archive metadata mismatch')
        if detectors % 8:
            padding_mask = (255 << (detectors % 8)) & 255
            if np.any(packed[:, -1] & padding_mask):
                raise Stage5EvidenceError(f'd={d}: nonzero padding bits in packed syndromes')
        failures = int(np.sum(np.any(truth != predictions, axis=1)))
        if failures != int(row['logical_failures']):
            raise Stage5EvidenceError(f'd={d}: logical failure count differs from raw data')
        _assert_close(failures / shots, float(row['logical_failure_rate']), f'd={d} error rate')
        # Optional independent decoder verification works in bounded chunks.
        # Stim/PyMatching are not imported for integrity-only auditing.
        if redo_decode or replay_sampling:
            import stim
            import pymatching
            model = stim.DetectorErrorModel.from_file(str(dem))
            decoder = pymatching.Matching.from_detector_error_model(model)
            if replay_sampling:
                archived_circuit = stim.Circuit.from_file(str(circuit))
                sampler = archived_circuit.compile_detector_sampler(seed=int(row['sampling_seed']))
                # Stim frame-sampling RNG batching can depend on the requested
                # shot count; replay the *original single sample call* exactly.
                replay_syndromes, replay_truth = sampler.sample(
                    shots=shots, separate_observables=True,
                )
            for start in range(0, shots, chunk):
                stop = min(shots, start + chunk)
                unpacked = np.unpackbits(packed[start:stop], axis=1, bitorder='little')[:, :detectors]
                if redo_decode:
                    independently_decoded = decoder.decode_batch(unpacked)
                    if not np.array_equal(independently_decoded, predictions[start:stop]):
                        raise Stage5EvidenceError(f'd={d}: stored predictions differ from PyMatching')
                if replay_sampling:
                    if not np.array_equal(unpacked, replay_syndromes[start:stop]) or not np.array_equal(truth[start:stop], replay_truth[start:stop]):
                        raise Stage5EvidenceError(f'd={d}: Stim seed replay mismatch')
    return {'distance': d, 'shots_verified': shots, 'logical_failures_verified': failures,
            'sha256_verified': True, 'predictions_redecoded': redo_decode,
            'seeded_sampling_replayed': replay_sampling}


def audit_stage5(directory: Path, *, require_all_nine: bool = False,
                 redo_decode: bool = False, replay_sampling: bool = False) -> dict:
    """Audit all recorded distances. Missing/inconsistent evidence always fails."""
    directory = Path(directory)
    evidence_path = _require_file(directory, 'evidence.json')
    evidence = json.loads(evidence_path.read_text())
    rows = evidence.get('local')
    if not isinstance(rows, list) or not rows:
        raise Stage5EvidenceError('No new local measurements in evidence.json')
    expected = [int(row['distance']) for row in rows]
    if len(expected) != len(set(expected)):
        raise Stage5EvidenceError('Duplicate distance measurements')
    if any(d not in PAPER_DISTANCES for d in expected):
        raise Stage5EvidenceError('Non-paper distance in evidence')
    if require_all_nine and set(expected) != set(PAPER_DISTANCES):
        raise Stage5EvidenceError(f'Incomplete published distance series: {expected}')
    with _require_file(directory, 'timing_summary.csv').open(newline='') as handle:
        csv_list = list(csv.DictReader(handle))
        csv_rows = {int(row['distance']): row for row in csv_list}
    if len(csv_rows) != len(csv_list):
        raise Stage5EvidenceError('Duplicate distances in timing CSV')
    if set(csv_rows) != set(expected):
        raise Stage5EvidenceError('CSV/evidence distance mismatch')
    if 'artifact_sha256' not in evidence:
        raise Stage5EvidenceError('No artifact manifest; legacy timing archive cannot pass strict audit')
    for filename, digest in evidence['artifact_sha256'].items():
        _require_file(directory, filename, digest)
    if evidence.get('timing_scope') != 'decode_batch only, excluding circuit generation, syndrome sampling and graph construction':
        raise Stage5EvidenceError('Unexpected timed work scope')
    results = []
    for row in rows:
        d, n = int(row['distance']), int(row['shots'])
        if n < 1 or int(row['timing_repeats']) < 2 or row['p'] != .001 or int(row['rounds']) != d:
            raise Stage5EvidenceError(f'd={d}: invalid protocol settings')
        elapsed = row['timing_trials_wall_seconds']
        if len(elapsed) != int(row['timing_repeats']) or not all(np.isfinite(elapsed)) or min(elapsed) <= 0:
            raise Stage5EvidenceError(f'd={d}: invalid trial durations')
        us = median(elapsed) * 1e6 / n
        _assert_close(us, float(row['microseconds_per_shot']), f'd={d} median shot time')
        _assert_close(us / d, float(row['microseconds_per_round']), f'd={d} median round time')
        _assert_close(us / d, float(csv_rows[d]['local_us_per_round_median']), f'd={d} CSV round time')
        _assert_close(min(elapsed)*1e6/n, float(csv_rows[d]['local_us_per_shot_min']), f'd={d} min time')
        _assert_close(max(elapsed)*1e6/n, float(csv_rows[d]['local_us_per_shot_max']), f'd={d} max time')
        _assert_close(float(row['logical_failure_rate']), float(csv_rows[d]['local_logical_failure_rate']), f'd={d} CSV accuracy')
        published = [x for x in evidence.get('reference', []) if int(x['distance']) == d]
        if len(published) != 1 or float(published[0]['p']) != .001:
            raise Stage5EvidenceError(f'd={d}: missing published reference entry')
        _assert_close(float(published[0]['microseconds_per_round']),
                      float(csv_rows[d]['published_us_per_round_m1_max']),
                      f'd={d}: published reference value')
        results.append(_verify_raw(directory, row, redo_decode=redo_decode,
                                   replay_sampling=replay_sampling))
    full = set(expected) == set(PAPER_DISTANCES)
    protocol = full and all(int(x['shots']) >= 10000 and int(x['timing_repeats']) >= 7 for x in rows)
    if require_all_nine and not protocol:
        raise Stage5EvidenceError('Strict full protocol requires all nine distances, >=10000 shots, >=7 repeats')
    return {
        'status': 'PASS', 'distance_count': len(results), 'all_nine_distances': full,
        'all_nine_10000_shots_7_repeats': protocol,
        'verified': results, 'computed_from': 'archived packed syndromes, outcomes, predictions and trial durations',
        'scope': ('Artifact integrity, internal consistency and optional decoder/seed replay; '
                  'not independent hardware timing or published three-decoder reproduction.'),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Strict audit of Stage 5 measured evidence')
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--all-nine', action='store_true', help='Require nine reference distances')
    parser.add_argument('--redo-decode', action='store_true', help='Independently decode stored syndromes')
    parser.add_argument('--replay', action='store_true', help='Bitwise Stim replay from stored seed')
    args = parser.parse_args(argv)
    try:
        result = audit_stage5(args.output_dir, require_all_nine=args.all_nine,
                              redo_decode=args.redo_decode, replay_sampling=args.replay)
        result['audit_runner'] = 'qec_decoder_lab.stage5_audit'
        print(json.dumps(result, indent=2))
        (args.output_dir / 'audit_stage5.json').write_text(json.dumps(result, indent=2) + '\n')
    except (Stage5EvidenceError, ValueError, RuntimeError, OSError) as exc:
        raise SystemExit(f'Stage 5 audit FAILED: {exc}') from exc


if __name__ == '__main__':
    main()
