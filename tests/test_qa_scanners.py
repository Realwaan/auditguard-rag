import json
from types import SimpleNamespace

import pytest
from app.ast_builder import ASTNode, ASTNodeType
from app.audit_scanner import verify_control_substance, DEFAULT_COMPLIANCE_CHECKS
import app.discrepancy_scanner as scanner


@pytest.mark.parametrize('raw', ['{}', '{"status":"ALIGN"}', '{"status":"UNKNOWN"}', 'not JSON'])
def test_malformed_discrepancy_output_does_not_assert_alignment(monkeypatch, raw):
    monkeypatch.setattr(scanner, 'USE_GOOGLE_GENAI', False)
    monkeypatch.setattr(scanner, 'load_image_bytes', lambda url: (b'image', 'image/png'))
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=raw))])
    monkeypatch.setattr(scanner, 'get_openai_client', lambda: SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: response))))
    p = ASTNode(id='p', type=ASTNodeType.PARAGRAPH, text='A service diagram.', title='Service', page=1, materialized_path='root/p')
    v = ASTNode(id='v', type=ASTNodeType.VISUAL, text='A service diagram.', title='Service', page=1, materialized_path='root/v', image_url='/static/crops/x.png')
    assert scanner.audit_cross_modal_pair(p, v)['status'] == 'UNVERIFIED'


def test_model_cannot_override_discrepancy_provenance():
    raw = {'status': 'ALIGN', 'title': 'Aligned', 'text_claim': 'A', 'visual_fact': 'B',
           'discrepancy_delta': 'None', 'confidence': .8, 'multimodal_verified': True, 'page': 999}
    parsed = scanner._validated_model_finding(json.dumps(raw))
    assert parsed and 'page' not in parsed and 'multimodal_verified' not in parsed
    for value in [-1, 2, float('nan')]:
        raw['confidence'] = value
        assert scanner._validated_model_finding(json.dumps(raw)) is None


@pytest.mark.parametrize('text,check_index', [('Uptime is guaranteed at 0%.', 0), ('MFA is not required for authentication.', 2)])
def test_explicit_missing_controls_do_not_pass(text, check_index):
    assert not verify_control_substance(text, DEFAULT_COMPLIANCE_CHECKS[check_index])['substantive_pass']


def test_uptime_zero_check_does_not_reject_ninety_point_zero():
    assert verify_control_substance('Guaranteed uptime is 90.0%.', DEFAULT_COMPLIANCE_CHECKS[0])['substantive_pass']
