"""Isolated-process same-machine decoding latency and peak-process-memory study.

Both decoders use identical archived held-out detector shots and run in fresh
Python subprocesses. Decoder construction is excluded from timed repetitions,
but INCLUDED in peak process RSS. The benchmark compares implementations
(PyMatching C++ batch versus Python recovery) as delivered, not algorithms
with equal optimization effort. Reused batches may benefit from CPU caching.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import resource
import statistics
import subprocess
import sys
from time import perf_counter
import tracemalloc

import numpy as np

from .frozen_validation import (_case_id, _case_paths, _canonical_bytes, _digest,
                                audit_case, load_locked_protocol)


def _peak_rss_bytes() -> int:
    peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak if sys.platform == 'darwin' else peak * 1024


def measure_worker(root: Path, case_id: str, decoder_name: str, *,
                   shots: int, repetitions: int) -> dict:
    if repetitions < 2 or shots < 1:
        raise ValueError('repetitions >= 2 and shots >= 1 required')
    protocol = load_locked_protocol(root)
    specs = { _case_id(item): item for item in protocol['case_specs'] }
    if case_id not in specs:
        raise ValueError('case not in frozen protocol')
    audit_case(root, protocol, specs[case_id])
    with np.load(_case_paths(root, case_id)['npz'], allow_pickle=False) as f:
        syndromes = f['detection_events'][:shots].copy()
        expected_predictions = f['mwpm_predictions' if decoder_name == 'mwpm'
                                 else 'recovery_predictions'][:shots].copy()
    if len(syndromes) != shots or shots > protocol['paired_shots_per_case']:
        raise ValueError('timed subset must fit both decoders and be identical')
    import stim
    dem = stim.DetectorErrorModel(_case_paths(root, case_id)['dem'].read_text())
    if decoder_name == 'hybrid' and protocol.get('strategy') != 'hybrid':
        raise ValueError('hybrid performance worker requires a frozen hybrid protocol')
    if decoder_name == 'recovery' and protocol.get('strategy') == 'hybrid':
        raise ValueError('hybrid protocol must benchmark the final hybrid decoder')
    if decoder_name == 'mwpm':
        import pymatching
        factory = lambda: pymatching.Matching.from_detector_error_model(dem)
    elif decoder_name in ('recovery', 'hybrid'):
        from .recovery import SyndromeRecoveryDecoder
        def factory():
            recovery = SyndromeRecoveryDecoder.from_dem(
                dem, iterations=protocol['iterations'], damping=protocol['damping'],
                refinement_budget=protocol['refinement_budget'],
                check_update=protocol['check_update'],
                basis_update=protocol.get('basis_update','python'))
            if decoder_name == 'recovery':
                return recovery
            import pymatching
            from .hybrid import SelectiveHybridDecoder
            return SelectiveHybridDecoder(
                pymatching.Matching.from_detector_error_model(dem),
                recovery, threshold=protocol['gate_fraction'])
    else:
        raise ValueError('decoder must be mwpm or recovery')
    tracemalloc.start()
    decoder = factory()
    # Warm-up and ensure output is correctly shaped before timing.
    if decoder_name == 'hybrid' and protocol['check_update'] in ('compiled', 'minsum'):
        # Compile optional kernel before timing, even if first shot is gated off.
        decoder.recovery.decode(syndromes[0])
    warm = decoder.decode_batch(syndromes[:1])
    if warm.shape != (1, dem.num_observables) or not np.array_equal(warm, expected_predictions[:1]):
        raise ValueError('warm-up decoder predictions differ from frozen evidence')
    _, python_peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()   # Avoid instrumentation perturbing timed calls.
    timings = []
    for _ in range(repetitions):
        t = perf_counter()
        predicted = decoder.decode_batch(syndromes)
        timings.append(float(perf_counter() - t))
        if predicted.shape != (shots, dem.num_observables) or not np.array_equal(predicted, expected_predictions):
            raise RuntimeError('timed decoder predictions changed relative to frozen evidence')
    result = {
        'decoder': decoder_name, 'case_id': case_id,
        'strategy': protocol.get('strategy', 'recovery'),
        'protocol_sha256': protocol['protocol_sha256'],
        'shots': shots, 'repetitions': repetitions,
        'trial_wall_seconds': timings,
        'median_microseconds_per_shot': statistics.median(timings) * 1e6 / shots,
        'peak_process_rss_bytes': _peak_rss_bytes(),
        'tracemalloc_construct_and_warmup_peak_bytes': int(python_peak_bytes),
        'machine': {'platform': platform.platform(), 'python': sys.version.split()[0],
                    'processor': platform.processor(), 'architecture': platform.machine()},
        'timing_scope': 'warmed decode_batch; construction excluded; identical stored input each repeat',
        'predictions_replayed_against_saved_evidence': True,
        'memory_scope': 'fresh worker process peak RSS including Python/libs and decoder construction; entire process high-water',
    }
    return result


def benchmark_all(root: Path, *, timing_shots: int = 64, repetitions: int = 5,
                  limit: int = 0, resume: bool = True, output: Path | None = None) -> dict:
    if timing_shots < 1 or repetitions < 2 or limit < 0:
        raise ValueError('timing shots > 0, repetitions >=2, limit >=0 required')
    protocol = load_locked_protocol(root)
    cases = protocol['case_specs'][:limit or None]
    if timing_shots > protocol['paired_shots_per_case']:
        raise ValueError('timing shots exceed paired recovery shots')
    # Fail closed if measured source/samples no longer match the frozen protocol.
    for spec in cases:
        audit_case(root, protocol, spec)
    output = output or root / 'performance'
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, spec in enumerate(cases, 1):
        cid = _case_id(spec)
        decoders = ('mwpm', 'hybrid') if protocol.get('strategy') == 'hybrid' else ('mwpm', 'recovery')
        for decoder in decoders:
            target = output / f'{cid}_{decoder}.json'
            expected = {'protocol_sha256': protocol['protocol_sha256'], 'case_id': cid,
                        'decoder': decoder, 'shots': timing_shots,
                        'repetitions': repetitions,
                        'raw_shots_sha256': _digest(_case_paths(root, cid)['npz'])}
            data = None
            if resume and target.is_file():
                data = json.loads(target.read_text())
                if any(data.get(k) != v for k, v in expected.items()):
                    raise ValueError(f'{target}: cached performance not compatible; do not silently mix measurements')
            if data is None:
                try:
                    proc = subprocess.run([
                    sys.executable, '-m', 'qec_decoder_lab.performance', 'worker',
                    '--root', str(root), '--case-id', cid, '--decoder', decoder,
                    '--shots', str(timing_shots), '--repetitions', str(repetitions),
                    ], text=True, capture_output=True, check=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f'performance worker failed for {cid}/{decoder}:\n{exc.stderr[-4000:]}') from exc
                data = json.loads(proc.stdout)
                data['raw_shots_sha256'] = expected['raw_shots_sha256']
                target.write_bytes(_canonical_bytes(data))
            records.append(data)
            print(f'[performance {i}/{len(cases)}] {cid} {decoder}: '
                  f'{data["median_microseconds_per_shot"]:.3f} us/shot, '
                  f'{data["peak_process_rss_bytes"]/2**20:.1f} MiB process peak', flush=True)
    rows = [{k: v for k, v in r.items() if k in (
        'case_id', 'decoder', 'shots', 'repetitions', 'median_microseconds_per_shot',
        'peak_process_rss_bytes', 'tracemalloc_construct_and_warmup_peak_bytes')}
            for r in records]
    with (output/'performance_summary.csv').open('w', newline='') as file:
        w = csv.DictWriter(file, fieldnames=list(rows[0]) if rows else []);
        if rows: w.writeheader(); w.writerows(rows)
    result = {'status': 'PASS' if len(cases) == len(protocol['case_specs']) else 'PARTIAL',
              'protocol_sha256': protocol['protocol_sha256'],
              'cases_measured': len(cases), 'cases_expected': len(protocol['case_specs']),
              'timing_shots_per_decoder': timing_shots,
              'repetitions_per_decoder': repetitions,
              'records': records,
              'limitations': [
                  'System load, CPU frequency and thermal throttling are not controlled.',
                  'PyMatching is optimized native C++; recovery is Python and algorithmically different.',
                  'Peak RSS includes Python and library overhead in each fresh process, not isolated algorithm bytes.',
                  'Warm CPU cache and duplicated syndromes can affect observed latency.',
                  'New timing data must be interpreted on this machine, not relative to the published PyMatching timings.',
              ]}
    (output/'performance_results.json').write_bytes(_canonical_bytes(result))
    return result


def audit_performance(root: Path, output: Path | None = None) -> dict:
    protocol = load_locked_protocol(root)
    output = output or root/'performance'
    manifest = json.loads((output/'performance_results.json').read_text())
    if manifest['protocol_sha256'] != protocol['protocol_sha256']:
        raise ValueError('performance study protocol hash mismatch')
    if manifest.get('status') != 'PASS':
        raise ValueError('cannot audit an incomplete performance study as complete')
    records = manifest['records']
    decoders = ('mwpm', 'hybrid') if protocol.get('strategy') == 'hybrid' else ('mwpm', 'recovery')
    expected_ids = [(_case_id(spec), dec) for spec in protocol['case_specs'] for dec in decoders]
    if [ (r['case_id'], r['decoder']) for r in records] != expected_ids:
        raise ValueError('performance dataset is incomplete or not in preregistered order')
    machines = {json.dumps(record.get('machine', {}), sort_keys=True) for record in records}
    if len(machines) != 1:
        raise ValueError('workers report different machines; not a same-machine comparison')
    for record in records:
        source_path = output / f'{record["case_id"]}_{record["decoder"]}.json'
        if json.loads(source_path.read_text()) != record:
            raise ValueError('timing record does not match its source JSON')
        if record['raw_shots_sha256'] != _digest(_case_paths(root, record['case_id'])['npz']):
            raise ValueError('benchmarked shot archive changed')
        trials = record['trial_wall_seconds']
        if len(trials) != record['repetitions'] or any(t < 0 or not np.isfinite(t) for t in trials):
            raise ValueError('invalid repeated timing data')
        recomputed = statistics.median(trials) * 1e6 / record['shots']
        if not np.isclose(recomputed, record['median_microseconds_per_shot'], rtol=1e-12):
            raise ValueError('timing median arithmetic mismatch')
        if not record.get('predictions_replayed_against_saved_evidence', False):
            raise ValueError('performance record lacks frozen prediction replay verification')
        if record['peak_process_rss_bytes'] <= 0:
            raise ValueError('invalid process memory evidence')
    return {'status': 'PASS', 'cases': len(expected_ids)//2, 'decoder_workers': len(records)}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    bench = sub.add_parser('run')
    bench.add_argument('--root', type=Path, default=Path('outputs/heldout'))
    bench.add_argument('--output', type=Path)
    bench.add_argument('--shots', type=int, default=64)
    bench.add_argument('--repetitions', type=int, default=5)
    bench.add_argument('--limit', type=int, default=0)
    bench.add_argument('--no-resume', action='store_true')
    audit = sub.add_parser('audit')
    audit.add_argument('--root', type=Path, default=Path('outputs/heldout'))
    audit.add_argument('--output', type=Path)
    worker = sub.add_parser('worker')
    worker.add_argument('--root', type=Path, required=True)
    worker.add_argument('--case-id', required=True)
    worker.add_argument('--decoder', choices=['mwpm','recovery','hybrid'], required=True)
    worker.add_argument('--shots', type=int, required=True)
    worker.add_argument('--repetitions', type=int, required=True)
    args = p.parse_args(argv)
    if args.action == 'worker':
        print(json.dumps(measure_worker(args.root, args.case_id, args.decoder,
                                        shots=args.shots, repetitions=args.repetitions)))
    elif args.action == 'run':
        result = benchmark_all(args.root, timing_shots=args.shots,
                               repetitions=args.repetitions, limit=args.limit,
                               resume=not args.no_resume, output=args.output)
        print(result['status'])
    else:
        print(json.dumps(audit_performance(args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
