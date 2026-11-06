"""Evidence gate: verify all required experiment artifacts before packaging."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .stage5_audit import Stage5EvidenceError, audit_stage5


class EvidenceGateError(ValueError):
    pass


def require_complete_evidence(root: Path, *, stage5_only: bool = False) -> dict:
    root = Path(root)
    report = {'scope': 'stage5_only' if stage5_only else 'stages_3_4_5', 'gates': {}}
    if not stage5_only:
        verification = json.loads((root / 'stage3/verification.json').read_text())
        penny = json.loads((root / 'stage3/pennylane_check.json').read_text())
        if verification.get('status') != 'PASS' or penny.get('status') != 'passed':
            raise EvidenceGateError('Stage 3: failed evidence replay or PennyLane check')
        report['gates']['stage3'] = 'PASS'
        study = json.loads((root / 'stage4/study_manifest.json').read_text())
        cases = study.get('cases', [])
        from itertools import product
        expected_ids = {
            f'{profile}_{basis}_d{d}_r{d}_p{p:.9g}'
            for profile, basis, d, p in product(
                ('circuit_level', 'data_only', 'measurement_only'),
                ('x', 'z'), (3, 5, 7), (.001, .003, .005, .01)
            )
        }
        case_ids = [c.get('case_id', '') for c in cases]
        if (study.get('status') != 'PASS' or len(cases) != 72 or len(set(case_ids)) != 72 or set(case_ids) != expected_ids or study.get('errors') or
            not (root / 'stage4/paired_comparisons.csv').is_file() or
            not (root / 'stage4/summary.csv').is_file() or
            any(c['audit'].get('status') != 'PASS' or
                'raw_shots_sha256' not in c['audit'].get('gates', []) or
                'independent_mwpm_redecode' not in c['audit'].get('gates', [])
                for c in cases)):
            raise EvidenceGateError('Stage 4: incomplete 72-case audited matrix')
        report['gates']['stage4'] = 'PASS'
    try:
        stage5 = audit_stage5(root / 'stage5_paper', require_all_nine=True,
                              redo_decode=True)
    except Stage5EvidenceError as exc:
        raise EvidenceGateError(f'Stage 5: {exc}') from exc
    logged_audit = json.loads((root / 'stage5_paper/audit_stage5.json').read_text())
    logged_rows = logged_audit.get('verified', [])
    if (logged_audit.get('status') != 'PASS' or len(logged_rows) != 9 or
        not all(row.get('predictions_redecoded', False) for row in logged_rows)):
        raise EvidenceGateError('Stage 5: independent MWPM redecoding not recorded')
    if stage5['distance_count'] != 9 or not stage5['all_nine_10000_shots_7_repeats']:
        raise EvidenceGateError('Stage 5: incomplete nine-distance protocol')
    report['gates']['stage5'] = 'PASS'
    report['status'] = 'PASS'
    report['limits'] = ('PASS means required files and internal evidence were audited. '
                        'Statistical power, quantum hardware validity and multi-decoder '
                        'figure reproduction require separate scientific review.')
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Require completed stage evidence before packaging')
    parser.add_argument('--outputs', type=Path, default=Path('outputs'))
    parser.add_argument('--stage5-only', action='store_true')
    args = parser.parse_args(argv)
    try:
        report = require_complete_evidence(args.outputs, stage5_only=args.stage5_only)
    except (EvidenceGateError, OSError, ValueError, KeyError) as exc:
        raise SystemExit(f'EVIDENCE VERIFICATION FAILED: {exc}') from exc
    output = args.outputs / 'final_evidence/evidence_audit.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(f'PASS: {output}')


if __name__ == '__main__':
    main()
