"""Saved Graph evaluation CLI uses the public reader, never a new capture."""
import json
from pathlib import Path

import pytest

from hyso.analysis.graph_evidence import GraphEvaluationResult
from hyso.cli import main
from hyso.cli.common import read_result
from hyso.reporting import create_report
from test_cli import no_workloads


@pytest.fixture(params=['baseline', 'candidate'])
def saved(tmp_path, request):
    # Exact retained association embedded in the previously accepted QA fixture.
    data = json.loads((Path(__file__).parent / 'fixtures/graph-cli-comparison-v1.json').read_text())
    path = tmp_path / 'evaluation.json'
    path.write_text(json.dumps(data['sources'][request.param]), encoding='utf8')
    return path


def test_dispatch_uses_public_reader(saved, monkeypatch):
    reader = GraphEvaluationResult.from_json
    calls = []

    def observe(path):
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(GraphEvaluationResult, 'from_json', observe)
    result = read_result(saved)
    assert calls == [saved]
    assert result.to_dict() == reader(saved).to_dict()


@pytest.mark.parametrize('format', ['json', 'markdown', 'html'])
def test_exports_match_public_reporting(saved, tmp_path, capsys, format):
    original = saved.read_bytes()
    source = GraphEvaluationResult.from_json(saved)
    expected = create_report(source, language='en', level='technical')
    output = tmp_path / ('output.' + format)
    assert main(['reports', 'render', str(saved), '--format', format, '--language', 'en',
                 '--output', str(output), '--json']) == 0
    actual = json.loads(capsys.readouterr().out)['result']
    assert actual == expected.to_dict()
    assert output.read_text(encoding='utf8') == getattr(expected, 'to_' + format)()
    raw = actual['sources'][0]['raw']
    assert raw == source.to_dict()
    statement = json.loads(original)
    for field in ('schema_version', 'evaluation', 'source_receipt', 'record', 'context'):
        assert raw[field] == statement[field]
    assert raw['verification']['graph_v2'] == 'unverified'
    assert raw['verification']['evaluation_capture'] == 'retained_statement'
    assert raw['relation']['status'] == 'unavailable'
    assert actual['presentation_only'] is True and actual['authoritative_evidence'] is False
    assert saved.read_bytes() == original


@pytest.mark.parametrize('mutation', ['schema', 'digest', 'json', 'format'])
def test_invalid_file_rejected(saved, tmp_path, capsys, mutation):
    data = json.loads(saved.read_text())
    if mutation == 'schema':
        data['schema_version'] = 999
    elif mutation == 'digest':
        data['content_sha256'] = '0' * 64
    elif mutation == 'format':
        data['format'] = 'unsupported.graph_evaluation'
    saved.write_text('{' if mutation == 'json' else json.dumps(data), encoding='utf8')
    output = tmp_path / 'must-not-exist'
    assert main(['reports', 'render', str(saved), '--output', str(output), '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'error'
    assert not output.exists()


def test_unsupported_export_rejected(saved, capsys):
    assert main(['reports', 'render', str(saved), '--format', 'pdf', '--json']) == 2
    assert 'invalid choice' in json.loads(capsys.readouterr().out)['error']
