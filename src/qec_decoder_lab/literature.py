"""Replot publisher-hosted PyMatching v2 timing CSV and rerun matched protocol.

Reference: Higgott & Gidney, 'Sparse Blossom: correcting a million errors
per core second with minimum-weight matching', Quantum 9, 1600 (2025).
Data are from the authors' PyMatching benchmarks/surface_codes directory.
The returned plot is a *partial reconstruction* of their v2 series, not a
reproduction of the original three-decoder graph or a cross-machine speedup.
"""
import csv
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
from statistics import median

from .stage5_audit import PAPER_DISTANCES, sha256
from .stats import summarize_errors
from time import perf_counter

import numpy as np

SOURCE_URL = (
    'https://raw.githubusercontent.com/oscarhiggott/PyMatching/refs/heads/master/'
    'benchmarks/surface_codes/'
    'surface_code_rotated_memory_x_p_0.001_d_5_7_9_13_17_23_29_39_50_both_bases/'
    'pymatching_v2.csv'
)
PAPER_DOI = 'https://doi.org/10.22331/q-2025-01-20-1600'


def read_reference_csv(path: Path) -> tuple[list[dict], str]:
    raw = Path(path).read_bytes()
    with Path(path).open(newline='', encoding='utf8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ['d', 'p', 'microseconds']:
            raise ValueError('unexpected upstream benchmark CSV schema')
        rows = []
        for row in reader:
            d, p, us = int(row['d']), float(row['p']), float(row['microseconds'])
            if d < 3 or not (0 < p < 1) or us <= 0 or not np.isfinite(us):
                raise ValueError('invalid reference benchmark row')
            rows.append({'distance': d, 'p': p, 'microseconds_per_shot': us,
                         'microseconds_per_round': us / d, 'detectors_per_round': d * d - 1})
    if not rows:
        raise ValueError('reference benchmark empty')
    return rows, hashlib.sha256(raw).hexdigest()


def recreate_reference_plot(rows: list[dict], output: Path, measured: list[dict] | None = None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 4.5))
    ax.plot([r['detectors_per_round'] for r in rows],
            [r['microseconds_per_round'] for r in rows], 'o-',
            label='Original PyMatching v2 dataset (M1 Max)')
    if measured:
        ax.plot([r['detectors_per_round'] for r in measured],
                [r['microseconds_per_round'] for r in measured], 's--',
                label='Local rerun (different host/version)')
    ax.set(xscale='log', yscale='log', xlabel='Stabilizer detectors per syndrome round (d²−1)',
           ylabel='MWPM decode time (µs per round)',
           title='Higgott–Gidney PyMatching v2 reference series (partial reconstruction)')
    ax.grid(True, which='both', alpha=0.22)
    ax.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=190)
    plt.close(fig)



def compare_scaling_shapes(reference: list[dict], measured: list[dict]) -> dict:
    """Dimensionless trend comparison; not cross-machine speed equivalence."""
    by_d = {r['distance']: r for r in reference}
    if len(measured) < 2 or len(set(r['distance'] for r in measured)) != len(measured):
        raise ValueError('Need >=2 unique distances for a scaling comparison')
    paired = sorted(measured, key=lambda r: r['distance'])
    if any(r['distance'] not in by_d or r['p'] != by_d[r['distance']]['p'] for r in paired):
        raise ValueError('Cannot compare unmatched circuit/noise parameters')
    x = np.log(np.array([r['detectors_per_round'] for r in paired], dtype=float))
    ref = np.log(np.array([by_d[r['distance']]['microseconds_per_round'] for r in paired]))
    local = np.log(np.array([r['microseconds_per_round'] for r in paired]))
    ref_shape = ref - ref[0]
    local_shape = local - local[0]
    return {
        'distances': [r['distance'] for r in paired],
        'reference_loglog_slope': float(np.polyfit(x, ref, 1)[0]),
        'local_loglog_slope': float(np.polyfit(x, local, 1)[0]),
        'normalized_shape_log_rmse': float(np.sqrt(np.mean((ref_shape - local_shape)**2))),
        'reference_relative_round_times': np.exp(ref_shape).tolist(),
        'local_relative_round_times': np.exp(local_shape).tolist(),
        'meaning': ('Exploratory timing growth versus detector-count proxy; '
                    'different CPUs/package versions prevent absolute timing replication.'),
    }


def run_matched_benchmark(
    reference: Path, output_dir: Path, *, distances: tuple[int, ...] = PAPER_DISTANCES,
    shots: int = 10000, seed: int = 20261103, repeats: int = 7,
    resume: bool = False,
) -> dict:
    import pymatching
    import stim

    if shots < 1 or repeats < 2 or not distances or any(d < 3 for d in distances):
        raise ValueError('invalid distance or shot count')
    if len(set(distances)) != len(distances):
        raise ValueError('duplicate benchmark distances')
    rows, digest = read_reference_csv(reference)
    p = 0.001
    available = {row['distance'] for row in rows if row['p'] == p}
    if any(d not in available for d in distances):
        raise ValueError('distance not present in upstream reference data')
    output_dir.mkdir(parents=True, exist_ok=True)
    host_fingerprint = {
        'platform': platform.platform(), 'machine': platform.machine(),
        'node': platform.node(), 'processor': platform.processor(),
        'logical_cpus': os.cpu_count(),
    }
    measured = []
    artifact_hashes = {}
    checkpoint = output_dir / 'timing_checkpoint.json'
    saved = {}
    if resume and checkpoint.exists():
        state = json.loads(checkpoint.read_text())
        expected_run = {'shots': shots, 'seed': seed, 'repeats': repeats,
                        'reference_sha256': digest, 'host_fingerprint': host_fingerprint,
                        'versions': {'stim': stim.__version__, 'pymatching': version('pymatching')}}
        if any(state.get(k) != value for k, value in expected_run.items()):
            raise ValueError('Cannot resume: benchmark configuration or software versions changed')
        saved = {int(row['distance']): row for row in state['completed']}
        if len(saved) != len(state['completed']):
            raise ValueError('Cannot resume: duplicate checkpoint distances')
        for d, row in saved.items():
            if d not in distances:
                continue
            for filename, recorded_sha in (
                (row['raw_shots_file'], row['raw_shots_sha256']),
                (f'd{d}.stim', row['circuit_sha256']),
                (f'd{d}.dem', row['dem_sha256']),
            ):
                path = output_dir / filename
                if not path.is_file() or sha256(path) != recorded_sha:
                    raise ValueError(f'Cannot resume: invalid checkpoint artifact {path}')
    resumed_count = 0
    if not resume:
        # A new run must not leave a previous successful evidence report behind
        # if execution is interrupted before any new final manifest is written.
        for obsolete in ('evidence.json', 'audit_stage5.json',
                         'timing_summary.csv', 'reference_vs_local.png',
                         'timing_checkpoint.json'):
            (output_dir / obsolete).unlink(missing_ok=True)
    for d in distances:
        if d in saved:
            row = saved[d]
            measured.append(row)
            resumed_count += 1
            artifact_hashes[row['raw_shots_file']] = sha256(output_dir / row['raw_shots_file'])
            artifact_hashes[f'd{d}.stim'] = sha256(output_dir / f'd{d}.stim')
            artifact_hashes[f'd{d}.dem'] = sha256(output_dir / f'd{d}.dem')
            print(f'd={d} resumed from verified checkpoint', flush=True)
            continue
        expected_detector_mb = shots * (d*d-1) * d / 1024**2
        print(f'd={d}: sampling {shots} shots, {repeats} timing repeats; '
              f'approximate dense detector input {expected_detector_mb:.0f} MiB', flush=True)
        # Exact circuit construction method published in the PyMatching repo.
        circuit = stim.Circuit.generated(
            'surface_code:rotated_memory_x', distance=d, rounds=d,
            after_clifford_depolarization=p,
            before_round_data_depolarization=p,
            before_measure_flip_probability=p,
            after_reset_flip_probability=p,
        )
        dem = circuit.detector_error_model(decompose_errors=True)
        matching = pymatching.Matching.from_detector_error_model(dem)
        syndromes, truth = circuit.compile_detector_sampler(seed=seed+d).sample(
            shots=shots, separate_observables=True,
        )
        matching.decode_batch(syndromes[:1])  # Warm up as in the authors' procedure.
        elapsed = []
        prediction = None
        for _ in range(repeats):
            t0 = perf_counter()
            result = matching.decode_batch(syndromes)
            elapsed.append(perf_counter() - t0)
            if prediction is None:
                prediction = result
            elif not np.array_equal(prediction, result):
                raise RuntimeError('Repeated matching returned inconsistent predictions')
        us = median(elapsed) * 1e6 / shots
        if truth.ndim != 2 or prediction.shape != truth.shape or len(syndromes) != shots:
            raise RuntimeError('Invalid archived syndrome or logical observable dimensions')
        num_detector_bits = int(syndromes.shape[1])
        num_observable_bits = int(truth.shape[1])
        raw_name = f'd{d}_raw_shots.npz'
        # Store the complete syndromes in a deterministic bit-packed encoding.
        # This retains all shots for independent replay without the size of a
        # dense uint8 array (d=50, 10000 shots approaches 1.25 GB dense).
        np.savez_compressed(output_dir / raw_name,
            detectors_packed=np.packbits(syndromes, axis=1, bitorder='little'),
            actual_observables=np.asarray(truth, dtype=np.uint8),
            mwpm_predictions=np.asarray(prediction, dtype=np.uint8),
            num_detectors=np.array(syndromes.shape[1], dtype=np.int64),
            num_observables=np.array(truth.shape[1], dtype=np.int64),
            shots=np.array(shots, dtype=np.int64),
            seed=np.array(seed+d, dtype=np.int64),
        )
        del syndromes
        artifact_hashes[raw_name] = sha256(output_dir / raw_name)
        failures = int(np.sum(np.any(prediction != truth, axis=1)))
        wilson = summarize_errors(np.any(prediction != truth, axis=1))
        measured.append({
            'distance': d, 'detectors_per_round': d * d - 1, 'rounds': d,
            'p': p, 'microseconds_per_shot': us, 'microseconds_per_round': us / d,
            'logical_failure_rate': failures / shots,
            'logical_failures': failures, 'shots': shots, 'sampling_seed': seed+d,
            'num_detectors': num_detector_bits,
            'num_observables': num_observable_bits,
            'raw_shots_file': raw_name, 'raw_shots_sha256': artifact_hashes[raw_name],
            'logical_failure_wilson_95_low': wilson['wilson_95_low'],
            'logical_failure_wilson_95_high': wilson['wilson_95_high'],
            'timing_trials_wall_seconds': elapsed,
            'median_batched_microseconds_per_shot': us,
            'timing_repeats': repeats,
            'min_batched_microseconds_per_shot': min(elapsed) * 1e6 / shots,
            'max_batched_microseconds_per_shot': max(elapsed) * 1e6 / shots,
            'decoder_throughput_shots_per_second': shots / median(elapsed),
            'circuit_sha256': hashlib.sha256(str(circuit).encode()).hexdigest(),
            'dem_sha256': hashlib.sha256(str(dem).encode()).hexdigest(),
        })
        (output_dir / f'd{d}.stim').write_text(str(circuit))
        (output_dir / f'd{d}.dem').write_text(str(dem))
        artifact_hashes[f'd{d}.stim'] = sha256(output_dir / f'd{d}.stim')
        artifact_hashes[f'd{d}.dem'] = sha256(output_dir / f'd{d}.dem')
        # Remove the large in-memory arrays before advancing to next distance.
        del truth, prediction
        checkpoint_data = {
            'shots': shots, 'seed': seed, 'repeats': repeats,
            'reference_sha256': digest, 'host_fingerprint': host_fingerprint,
            'versions': {'stim': stim.__version__, 'pymatching': version('pymatching')},
            'completed': measured,
        }
        # Atomic update so an interrupted d=50 run cannot destroy d<=39 work.
        temporary = checkpoint.with_suffix('.tmp')
        temporary.write_text(json.dumps(checkpoint_data, indent=2, allow_nan=False) + '\n')
        temporary.replace(checkpoint)
        print(f'd={d}: {us/d:.6f} us/round, {failures} logical failures, raw shots archived', flush=True)
    recreate_reference_plot(rows, output_dir / 'reference_vs_local.png', measured)
    reference_by_distance = {r['distance']: r for r in rows}
    with (output_dir / 'timing_summary.csv').open('w', newline='', encoding='utf8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'distance', 'physical_error_probability', 'shots', 'repeats',
            'local_us_per_shot_median', 'local_us_per_shot_min', 'local_us_per_shot_max',
            'local_us_per_round_median', 'published_us_per_round_m1_max',
            'local_decoder_shots_per_second', 'local_logical_failure_rate',
        ])
        writer.writeheader()
        for r in measured:
            writer.writerow({
                'distance': r['distance'], 'physical_error_probability': r['p'],
                'shots': shots, 'repeats': repeats,
                'local_us_per_shot_median': r['microseconds_per_shot'],
                'local_us_per_shot_min': r['min_batched_microseconds_per_shot'],
                'local_us_per_shot_max': r['max_batched_microseconds_per_shot'],
                'local_us_per_round_median': r['microseconds_per_round'],
                'published_us_per_round_m1_max': reference_by_distance[r['distance']]['microseconds_per_round'],
                'local_decoder_shots_per_second': r['decoder_throughput_shots_per_second'],
                'local_logical_failure_rate': r['logical_failure_rate'],
            })
    artifact_hashes['timing_summary.csv'] = sha256(output_dir / 'timing_summary.csv')
    artifact_hashes['reference_vs_local.png'] = sha256(output_dir / 'reference_vs_local.png')
    if checkpoint.is_file():
        artifact_hashes['timing_checkpoint.json'] = sha256(checkpoint)
    cpu_model = platform.processor()
    if not cpu_model and Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text(errors='replace').splitlines():
            if line.lower().startswith('model name'):
                cpu_model = line.split(':', 1)[1].strip()
                break
    result = {
        'status': ('protocol_matched_cross_machine_comparison_not_exact_timing_reproduction'
                   if shots >= 10000 else 'exploratory_fewer_shots_than_published_protocol'),
        'full_nine_distances': set(distances) == set(PAPER_DISTANCES),
        'resumed_distances_from_checkpoint': resumed_count,
        'full_nine_10000_shots_7_repeats': (set(distances) == set(PAPER_DISTANCES)
                                            and shots >= 10000 and repeats >= 7),
        'artifact_sha256': artifact_hashes,
        'raw_evidence_format': 'npz: detectors_packed uses little-endian numpy.packbits per shot, '
                               'with num_detectors determining padding; original truth/prediction arrays preserved',
        'paper': PAPER_DOI, 'upstream_csv': SOURCE_URL,
        'vendored_reference_sha256': digest, 'reference': rows, 'local': measured,
        'sampling_shots_each': shots, 'seed_rule': 'seed + distance',
        'meets_published_minimum_shots': shots >= 10000,
        'timing_repeats_each': repeats, 'timing_aggregation': 'median of warmed batch calls',
        'timing_scope': 'decode_batch only, excluding circuit generation, syndrome sampling and graph construction',
        'files': sorted(artifact_hashes),
        'shape_comparison': compare_scaling_shapes(rows, measured) if len(measured) > 1 else None,
        'host': {'platform': platform.platform(), 'processor': cpu_model,
                 'logical_cpus': os.cpu_count(), 'python': platform.python_version(),
                 'github_runner_name': os.environ.get('RUNNER_NAME'),
                 'github_run_id': os.environ.get('GITHUB_RUN_ID')},
        'versions': {'stim': stim.__version__, 'pymatching': version('pymatching')},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'limitations': [
            'Original graph also compares PyMatching v0.7 and NetworkX; those are not reconstructed here',
            'Vendored numeric values checked against upstream display, but file bytes not independently verified',
            'Original timings were on an M1 Max, so raw latencies across computers are not comparable',
            'Current PyMatching version may differ from the v2.0 source benchmark',
            'Published CSV timing curve is replotted, not experimentally regenerated on original hardware',
            'Measuring d=50 uses substantial memory (~1.25 GB for 10000 dense uint8 detector shots); machine may swap',
            'No threshold or decoder advantage follows from paper-matched MWPM timing measurements',
        ],
    }
    (output_dir / 'evidence.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result
