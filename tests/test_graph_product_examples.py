"""Execute only the tiny CPU Graph examples and their saved-result reader path."""
import json
from pathlib import Path

import pytest
import torch

from hyso import ExperimentManager, Trainer, performance
from hyso.models.graph import GraphModel
from hyso.reporting import PresentationReport
from hyso.training import resume
from examples.models.graph_evidence_workflow import train_named, run, compare_saved
from test_graph_training_resume import cpu_only
from test_trainer_checkpoint_restore import assert_state


def test_named_example_continues_an_interrupted_checkpoint(tmp_path, monkeypatch):
    native_save = resume.save_last
    class Interrupted(Exception):
        pass
    def interrupt(*args, **kwargs):
        native_save(*args, **kwargs)
        raise Interrupted()
    with monkeypatch.context() as scoped:
        scoped.setattr(resume, 'save_last', interrupt)
        with pytest.raises(Interrupted):
            train_named(tmp_path / 'interrupted')
    result = train_named(tmp_path / 'continued', resume_from=tmp_path / 'interrupted/last.pth')
    baseline = train_named(tmp_path / 'complete')
    assert result['completed_epochs'] == 2 and result['applied_updates'] == 4
    assert result['history'] == baseline['history']
    actual = torch.load(result['checkpoint'], weights_only=True)
    expected = torch.load(baseline['checkpoint'], weights_only=True)
    assert actual['format_version'] == 2
    assert_state(actual['model'], expected['model'])
    # The same example also handles an already completed budget without a write.
    completed = train_named(tmp_path / 'noop', resume_from=result['checkpoint'])
    assert completed['history'] == result['history']
    assert completed['checkpoint'] == result['checkpoint']
    assert not (tmp_path / 'noop/last.pth').exists()
    with pytest.raises(FileExistsError):
        train_named(tmp_path / 'complete')


def test_saved_graph_example_exports_both_languages_without_new_measurement(tmp_path, monkeypatch):
    output = tmp_path / 'workflow'
    summary = run(output)
    assert summary['applied_updates'] == [2, 2]
    assert summary['wrong_identity_rejected']
    assert summary['saved_comparison']['structure'] == 'COMPARABLE'
    assert summary['saved_comparison']['imported_graph_trust'] == ['unverified', 'unverified']
    for language in ('tr', 'en'):
        path = output / 'saved-results' / language
        report = PresentationReport.from_json(path.with_suffix('.json')).to_dict()
        assert report['presentation_only'] and not report['authoritative_evidence']
        section = next(s for s in report['sources'] if s['kind'] == 'comparison')
        comparison = json.loads((output / 'saved-results/comparison.json').read_text(encoding='utf8'))
        assert section['raw'] == comparison
        assert all(row['difference'] is not None and row['status'] == 'COMPARABLE'
                   for row in comparison['metrics'] if row['feature_id'] == 'P26')
        assert all(row['difference'] is None and row['status'] == 'INSUFFICIENT_EVIDENCE'
                   for row in comparison['metrics'] if row['feature_id'] == 'P12')
        for suffix in ('.md', '.html'):
            text = path.with_suffix(suffix).read_text(encoding='utf8')
            assert 'INSUFFICIENT_EVIDENCE' in text and 'COMPARABLE' in text
            assert 'unverified' in text
            for metric in section['metrics']:
                for value in (metric.get('baseline'), metric.get('candidate'), metric['value']):
                    if value is not None:
                        assert str(value) in text
    manager = ExperimentManager(output / 'experiments')
    try:
        details = manager.get_experiment_history(summary['experiment_id'])
        identities = list(dict.fromkeys(row['artifact']['id'] for run_info in details['runs']
            for row in manager.get_run_details(run_info['id'])['artifacts']
            if row['artifact']['kind'] == 'graph_evaluation_association'))
        assert len(identities) == 2
        def forbidden(*args, **kwargs):
            pytest.fail('Saved example started training, forward or profiling')
        for target, name in ((Trainer, 'fit'), (GraphModel, 'forward'), (performance, 'profile_operators')):
            monkeypatch.setattr(target, name, forbidden)
        reused = compare_saved(manager, summary['experiment_id'], identities, tmp_path / 'read-only')
        assert reused['metrics'] == summary['saved_comparison']['metrics']
    finally:
        manager.close()


def test_documented_two_loss_recipe_runs_on_cpu():
    guide = Path(__file__).resolve().parents[1] / 'docs/guides/training-config.md'
    section = guide.read_text(encoding='utf8').split('### Two named classification losses', 1)[1]
    code = section.split('```python\n', 1)[1].split('```', 1)[0]
    namespace = {}
    exec(compile(code, str(guide), 'exec'), namespace)
    assert namespace['trainer'].update_count == 1
    assert namespace['history']['val_acc'] == [None]
