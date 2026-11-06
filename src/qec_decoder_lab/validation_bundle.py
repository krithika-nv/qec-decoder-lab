"""Audit-gated, portable bundle of held-out evidence (never auto-committed)."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

from .frozen_validation import audit_study, _digest
from .performance import audit_performance


def pack(root: Path, zip_path: Path, *, require_performance: bool = True) -> dict:
    if not root.is_dir() or zip_path.resolve().is_relative_to(root.resolve()):
        raise ValueError('bundle destination must be outside the evidence root')
    audit = audit_study(root)
    if audit['status'] != 'PASS':
        raise RuntimeError('held-out evidence is incomplete; refuse to package as validated')
    if require_performance:
        audit_performance(root)
    # Reject extraneous symlinks and stale partial files; no arbitrary directory traversal.
    files = sorted(p for p in root.rglob('*') if p.is_file())
    if any(p.is_symlink() for p in files):
        raise ValueError('symlink in evidence directory')
    if not files:
        raise ValueError('no evidence files')
    manifest = {
        'scope': 'artifact and accounting validation, not scientific superiority',
        'heldout_cases_verified': audit['cases_verified'],
        'performance_required': require_performance,
        'files': {str(p.relative_to(root)): {'sha256': _digest(p), 'bytes': p.stat().st_size}
                  for p in files},
    }
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(zip_path.suffix+'.tmp')
    with ZipFile(tmp, 'w', compression=ZIP_DEFLATED, compresslevel=6) as bundle:
        for path in files:
            bundle.write(path, 'heldout/' + str(path.relative_to(root)))
        bundle.writestr('BUNDLE_MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    tmp.replace(zip_path)
    return manifest


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('outputs/heldout'))
    p.add_argument('--zip', type=Path, default=Path('qec-heldout-validation-evidence.zip'))
    p.add_argument('--no-performance', action='store_true')
    args = p.parse_args(argv)
    result = pack(args.root, args.zip, require_performance=not args.no_performance)
    print(f'PACKED: {args.zip} | {len(result["files"])} evidence files | '
          f'{result["heldout_cases_verified"]} held-out cases')


if __name__ == '__main__':
    main()
