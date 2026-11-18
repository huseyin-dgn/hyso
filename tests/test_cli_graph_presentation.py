"""Retained Graph comparison CLI delegates to existing readers and reporting."""
import json
from pathlib import Path

import pytest

from hyso.analysis.graph_evidence import GraphComparisonResult
from hyso.reporting import create_report
from hyso.cli import main
from test_cli import no_workloads


FIXTURE = Path(__file__).parent / 'fixtures/graph-cli-comparison-v1.json'


@pytest.mark.parametrize('format', ['json', 'markdown'])
@pytest.mark.parametrize('language', ['en', 'tr'])
def test_retained_graph_render_matches_public_api(tmp_path, capsys, format, language):
    original = FIXTURE.read_bytes()
    source = GraphComparisonResult.from_json(FIXTURE)
    expected = create_report(source, language=language, level='technical')
    output = tmp_path / 'rendered.txt'
    assert main(['reports', 'render', str(FIXTURE), '--format', format,
                 '--language', language, '--output', str(output), '--json']) == 0
    actual = json.loads(capsys.readouterr().out)['result']
    assert actual == expected.to_dict()
    assert output.read_text(encoding='utf8') == getattr(expected, 'to_' + format)()
    raw = actual['sources'][0]['raw']
    assert raw == source.to_dict()
    assert raw['status'] == source.to_dict()['status']
    assert all(v['graph_v2'] == 'unverified' for v in raw['verification'].values())
    assert actual['presentation_only'] is True
    assert actual['authoritative_evidence'] is False
    assert FIXTURE.read_bytes() == original


@pytest.mark.parametrize('mutation', ['version', 'digest', 'format'])
def test_invalid_retained_graph_rejected(tmp_path, capsys, mutation):
    data = json.loads(FIXTURE.read_text(encoding='utf8'))
    if mutation == 'version':
        data['schema_version'] = 999
    elif mutation == 'digest':
        data['status'] = 'COMPARABLE'
        data['content_sha256'] = '0' * 64
    else:
        data['format'] = 'unsupported.graph'
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(data), encoding='utf8')
    output = tmp_path / 'must-not-exist.json'
    assert main(['reports', 'render', str(path), '--output', str(output), '--json']) == 2
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'error' and 'result' not in result
    assert not output.exists()
