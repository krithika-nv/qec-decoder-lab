"""Cross-platform smoke regression tests for the Bash runner."""
import os
import subprocess
from pathlib import Path


def test_stage5_script_handles_empty_optional_argument_groups(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env.update(PYTHON='/bin/true', DISTANCES='5 7', SHOTS='10',
               REPEATS='1', OUTPUT_DIR=str(tmp_path / 'timing'))
    result = subprocess.run(['bash', 'scripts/shell/RUN_STAGE5_PAPER.sh'],
                            cwd=root, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'unbound variable' not in result.stderr


def test_stage4_script_rejects_invalid_mode():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(['bash', 'scripts/shell/RUN_STAGE4_COMPARISON.sh'],
                            cwd=root, env={**os.environ, 'MODE': 'invalid'},
                            capture_output=True, text=True)
    assert result.returncode == 2
    assert 'MODE must be quick or full' in result.stderr
