"""Only selection, sealed-source admission, retry and ownership boundaries."""
import json
from pathlib import Path
import pytest
from tools.product_acceptance import runner as r


def test_exact_four_cpu_cases_and_no_gpu_request(tmp_path):
    nodes=r.cpu_selection()
    assert len(nodes)==4
    assert {n.rsplit('::test_',1)[-1] for n in nodes}=={'U15','U16','U17','U18'}
    for group,selection in [('01-admission',nodes),('CPU_REGRESSIONS',{})]:
        with pytest.raises(RuntimeError,match='CPU_OFFLINE_ONLY'):
            r.request_for(tmp_path,tmp_path/'work',group,{},selection)


def test_own_request_scope_and_cpu_worker(monkeypatch,tmp_path):
    monkeypatch.setattr(r,'BASE_REQUEST',lambda *args:dict(campaign_id='cuda-functional',profile='cpu'))
    request=r.request_for(tmp_path,tmp_path/'work','CPU_REGRESSIONS',{},r.cpu_selection())
    assert request['campaign_id']=='product-experience' and request['profile']=='cpu'
    assert request['argv'][0].endswith('user_acceptance/cpu_worker.py') or request['argv'][0].endswith('user_acceptance\\cpu_worker.py')


def test_normal_resume_keeps_pass_and_failed_results_until_explicit_retry():
    records={'PASS':{'status':'PASS'},'FAIL':{'status':'FAIL'},'BLOCKED':{'status':'BLOCKED'}}
    assert r.user.pending_cases(['PASS','FAIL','BLOCKED','NEW'],records,set())==['NEW']
    assert r.user.pending_cases(['PASS','FAIL','BLOCKED','NEW'],records,{'FAIL'})==['FAIL','NEW']


def test_cleanup_keeps_wrong_scope_and_missing_marker(tmp_path):
    from campaign_operations import cleanup_owned
    work=tmp_path/'payload';work.mkdir();(work/'proof.txt').write_text('retain')
    for marker in (None,{'campaign_id':'cuda-functional'}):
        if marker:(work/'.hyso-owned-work.json').write_text(json.dumps(marker))
        with pytest.raises(ValueError,match='owned marker'):cleanup_owned(work,tmp_path,'product-experience')
        assert (work/'proof.txt').read_text()=='retain'
    # A newly created test-owned marker, never retroactive production repair.
    (work/'.hyso-owned-work.json').write_text(json.dumps({'campaign_id':'product-experience'}))
    cleanup_owned(work,tmp_path,'product-experience')
    assert not work.exists()


def test_baseline_requires_plan_seal_before_trusting_pass(monkeypatch):
    monkeypatch.setattr(r.engine,'read_json',lambda p,*args: {} if p.name=='plan.json' else {'plan_sha256':'forged'})
    with pytest.raises(RuntimeError,match='PLAN_SEAL_MISMATCH'):r.verify_baseline()


def test_baseline_rejects_incomplete_or_held_records(monkeypatch):
    plan={'groups':{'one':['ADMISSION']}}
    cp={'plan_sha256':r.engine.digest(plan),'cpu_results':{},'results':{}}
    monkeypatch.setattr(r.engine,'read_json',lambda p,*args:plan if p.name=='plan.json' else cp)
    with pytest.raises(RuntimeError,match='26_CASE'):r.verify_baseline()


def test_prepare_has_no_attempts_no_gpu_and_seals_prior_evidence(monkeypatch,tmp_path):
    prior=tmp_path/'prior-proof.json';prior.write_text('{"status":"PASS"}')
    sha=r.engine.file_hash(prior)
    monkeypatch.setattr(r,'ROOT',tmp_path/'product')
    monkeypatch.setattr(r,'INVENTORY',tmp_path/'inventory.json')
    monkeypatch.setattr(r,'verify_baseline',lambda:({'budgets':{'job_bytes':123},'deferred':['B4']},{}))
    monkeypatch.setattr(r,'build_inventory',lambda:{'schema_version':1})
    monkeypatch.setattr(r,'snapshot',lambda:{'reviewed.py':'bound'})
    monkeypatch.setattr(r,'preserved',lambda:{str(prior):sha})
    monkeypatch.setattr(r.engine,'user_snapshot',lambda:{})
    monkeypatch.setattr(r.engine,'git',lambda *args:'recorded')
    monkeypatch.setattr(r,'progress',lambda *args,**kwargs:None)
    monkeypatch.setattr(r.engine,'gpu_info',lambda:pytest.fail('offline preparation queried GPU'))
    p,cp=r.prepare()
    assert p['scenario_counts']=={'CPU':4,'GPU':0} and p['groups']=={} and p['runtime']=={}
    assert cp['attempts']==[] and cp['results']=={}
    assert p['preserved_evidence']=={str(prior):sha}
    assert cp['plan_sha256']==r.engine.digest(p) and r.engine.file_hash(prior)==sha
