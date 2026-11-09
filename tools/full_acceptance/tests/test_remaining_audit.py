"""Offline reconciliation refuses drift and incomplete claims."""
import copy

import pytest

from tools.full_acceptance import core, remaining_audit as audit


def sources():
    features = {f'P{i:02}': dict(name=str(i), decision='PARTIAL' if i in (13, 23, 30) else 'PASS',
        current_execution='INCOMPLETE', experiments={}, decision_basis='scoped fixture',
        retained_scope='synthetic test', prior_decision_source={}) for i in range(1, 36)}
    base = dict(path='base', manifest=dict(sha256='a'*64), state=dict(preservation_status='FAIL'), features=features)
    long = copy.deepcopy(base)
    long.update(path='long', stages={'10.stability':dict(status='PASS', criteria=dict(P31_uninterrupted_7200=True,
        no_sampling_gaps=True), metrics=[dict(name='actual_work_seconds', value=7200.)]),
        '11.memory':dict(status='PASS', criteria=dict(actual_long_run_series=True))})
    roofline = copy.deepcopy(base)
    roofline.update(path='roofline', stages={'06.roofline':dict(status='PASS', criteria=dict(P18_calibration_screen=True))})
    return base, long, roofline


def test_union_retains_scope_and_failed_preservation():
    rows = audit.feature_rows(*sources())
    assert len(rows) == 35 and sum(r['decision'] == 'PASS' for r in rows) == 32
    assert {r['id'] for r in rows if r['decision'] == 'PARTIAL'} == {'P13', 'P23', 'P30'}
    assert rows[0]['source_preservation'] == 'FAIL'


@pytest.mark.parametrize('damage', ['short', 'gaps', 'memory', 'roofline', 'false_p13'])
def test_missing_real_criterion_is_not_a_pass(damage):
    base, long, roofline = sources()
    if damage == 'short': long['stages']['10.stability']['metrics'][0]['value'] = 7199.
    elif damage == 'gaps': long['stages']['10.stability']['criteria']['no_sampling_gaps'] = False
    elif damage == 'memory': long['stages']['11.memory']['status'] = 'PARTIAL'
    elif damage == 'roofline': roofline['stages']['06.roofline']['criteria']['P18_calibration_screen'] = False
    else: base['features']['P13']['decision'] = 'PASS'
    with pytest.raises(ValueError): audit.feature_rows(base, long, roofline)


def test_hash_drift_does_not_become_reconciled_evidence(tmp_path):
    path = tmp_path / 'raw.json'
    path.write_text('{}')
    ref = core.record(path)
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='artifact changed'): audit.pinned(ref)
