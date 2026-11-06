"""Ensure the source-release documentation does not contain broken local links."""

from pathlib import Path
import importlib.util


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "python" / "check_project_docs.py"


def test_documentation_links_exist():
    spec = importlib.util.spec_from_file_location("check_project_docs", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.check_links() == []
