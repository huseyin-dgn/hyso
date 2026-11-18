"""Saved presentation CLI exports preserve the public reader's exact snapshot."""
import json
from pathlib import Path

import pytest

from hyso.cli import main
from hyso.reporting import PresentationReport, create_report
from hyso.analysis.graph_evidence import GraphComparisonResult
from test_cli import no_workloads


@pytest.fixture
def saved(tmp_path):
    source = Path(__file__).parent / 'fixtures/graph-cli-comparison-v1.json'
    report = create_report(GraphComparisonResult.from_json(source), language='en', level='expert')
    path = tmp_path / 'saved report.json'
    report.to_json(path)
    return path


@pytest.mark.parametrize('format', ['json', 'markdown', 'html'])
def test_exports_equal_public_reader(saved, tmp_path, capsys, monkeypatch, format):
    raw = saved.read_bytes()
    expected = PresentationReport.from_json(saved)
    reader = PresentationReport.from_json
    calls = []

    def observe(path):
        calls.append(path)
        return reader(path)

    monkeypatch.setattr(PresentationReport, 'from_json', observe)
    # Re-wrapping a presentation would discard its source/eligibility boundaries.
    monkeypatch.setattr('hyso.reporting.create_report', lambda *a, **kw: pytest.fail('saved report rebuilt'))
    output = tmp_path / ('output.' + format)
    assert main(['reports', 'render', str(saved), '--format', format,
                 '--language', 'tr', '--level', 'beginner', '--output', str(output), '--json']) == 0
    result = json.loads(capsys.readouterr().out)['result']
    assert calls == [saved]
    assert result == expected.to_dict()
    assert output.read_text(encoding='utf8') == getattr(expected, 'to_' + format)()
    assert result['language'] == 'en' and result['level'] == 'expert'
    assert result['presentation_only'] is True and result['authoritative_evidence'] is False
    assert saved.read_bytes() == raw


@pytest.mark.parametrize('field,value', [('schema_version', 999), ('presentation_only', False),
                                        ('authoritative_evidence', True), ('content_sha256', '0'*64),
                                        ('format', 'unsupported.presentation')])
def test_invalid_saved_report_rejected(saved, tmp_path, capsys, field, value):
    data = json.loads(saved.read_text(encoding='utf8'))
    data[field] = value
    saved.write_text(json.dumps(data), encoding='utf8')
    output = tmp_path / 'must-not-exist'
    assert main(['reports', 'render', str(saved), '--output', str(output), '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'error'
    assert not output.exists()


@pytest.mark.parametrize('option', ['--context', '--advisor-source'])
def test_saved_report_rejects_source_overrides(saved, capsys, option):
    assert main(['reports', 'render', str(saved), option, str(saved), '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'error'


def test_saved_output_cannot_replace_input(saved, capsys):
    original = saved.read_bytes()
    assert main(['reports', 'render', str(saved), '--output', str(saved), '--overwrite', '--json']) == 2
    assert json.loads(capsys.readouterr().out)['status'] == 'error'
    assert saved.read_bytes() == original
