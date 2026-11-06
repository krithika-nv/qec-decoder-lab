import json
import pytest

from qec_decoder_lab.evidence_gate import EvidenceGateError, require_complete_evidence


def test_evidence_packaging_refuses_unexecuted_studies(tmp_path):
    with pytest.raises(FileNotFoundError):
        require_complete_evidence(tmp_path)
    with pytest.raises(EvidenceGateError, match="Stage 5"):
        require_complete_evidence(tmp_path, stage5_only=True)


def test_evidence_requires_actual_stage3_pass(tmp_path):
    stage3 = tmp_path/'stage3'
    stage3.mkdir()
    (stage3/'verification.json').write_text(json.dumps({'status': 'INCOMPLETE'}))
    (stage3/'pennylane_check.json').write_text(json.dumps({'status': 'passed'}))
    with pytest.raises(EvidenceGateError, match='Stage 3'):
        require_complete_evidence(tmp_path)
