"""CLI for nine-distance, paper-matched, measured PyMatching benchmark."""
import argparse
import json
from pathlib import Path

from qec_decoder_lab.literature import read_reference_csv, recreate_reference_plot, run_matched_benchmark
from qec_decoder_lab.stage5_audit import PAPER_DISTANCES


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument('--reference', type=Path, default=root/'data/literature/pymatching_v2_p001.csv')
    parser.add_argument('--output-dir', type=Path, default=Path('outputs/literature'))
    parser.add_argument('--distances', nargs='+', type=int, default=list(PAPER_DISTANCES))
    parser.add_argument('--shots', type=int, default=10000)
    parser.add_argument('--repeats', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20261103)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--reference-only', action='store_true',
                        help='Recreate published v2 timing series without external quantum packages')
    args = parser.parse_args()
    if args.reference_only:
        rows, digest = read_reference_csv(args.reference)
        recreate_reference_plot(rows, args.output_dir/'reference_only.png')
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / 'reference_only_evidence.json').write_text(json.dumps({
            'status': 'reference_replot_only_no_new_simulation',
            'source_digest_sha256': digest, 'reference_rows': rows,
            'data_origin': 'transcribed values visually cross-checked against upstream on 2026-11-03; bytes unverified',
            'published_reference': 'https://doi.org/10.22331/q-2025-01-20-1600',
        }, indent=2, allow_nan=False) + '\n')
        print(f'Reference series replotted from transcribed CSV sha256={digest}')
    else:
        result = run_matched_benchmark(
            args.reference, args.output_dir, distances=tuple(args.distances), shots=args.shots, repeats=args.repeats, seed=args.seed, resume=args.resume,
        )
        print(f"Wrote {args.output_dir / 'evidence.json'}: {result['status']}")


if __name__ == '__main__':
    main()
