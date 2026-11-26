"""Fixed release-scope regression selection using the existing owned controller.

Does not initialize, resume, invalidate or rewrite the completed general campaign.
"""
from __future__ import annotations
import argparse
import ast
import json
import msvcrt
import tempfile
import sys
import uuid
from collections import Counter
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from common import ROOT,REPO,read_json,atomic_json,digest,file_hash,source_snapshot,user_snapshot,utc,git,tree_bytes
from campaign_controller import execute
from campaign_evidence import reconcile_attempt
from campaign_preflight import runtime_plan,check_config,resources,ExecutionState
from campaign_operations import cleanup_owned

OUT=ROOT/'support-scope-finalization-20261126'
TARGETS={
 'tests/test_packaging.py':{'test_wheel_and_sdist_content_and_metadata'},
 'tests/test_user_guide.py':{
 'test_generated_guide_is_current_and_version_matches','test_complete_physical_language_and_parallel_meaning_fields',
 'test_utf8_roundtrip_and_language_characters','test_unique_stable_topics_complete_features_and_no_broken_anchors',
 'test_analysis_topics_have_all_required_fields_and_critical_constraints','test_html_has_no_external_dependencies_or_executable_examples',
 'test_archives_contain_the_same_single_four_language_html','test_clean_wheel_guide_without_torch_or_checkout'},
 'tests/test_contextual_help.py':{'test_api_links_match_the_actual_model_or_analysis','test_index_version_all_topics_and_source_docstrings','test_clean_wheel_topic_reader_and_cli'},
 'tests/test_analysis_execution.py':{'test_profiles_catalog_and_selection_are_pure'},
 'tests/test_cli.py':{'test_analysis_catalog_and_plan','test_native_outcome_exit_codes','test_unsupported_native_device_is_distinct_from_failure'},
 'tests/test_runtime_analysis.py':{'test_real_cpu_duration_chunks_reopen','test_cpu_model_state_modes_rng_preserved'},
 'tests/test_b4_containment.py':{'test_public_capture_denied_before_worker_queries_or_tempfiles','test_analyzer_preflight_and_authorized_execution_keep_B4_reason','test_failed_capture_is_not_zero_or_success_and_old_envelopes_remain_readable'},
}


def snapshot():
    value=source_snapshot()
    for name in ('tests/test_release_support_scope.py','docs/guides/release-support.md'):
        value[name]=file_hash(REPO/name)
    return value


def selection():
    completed=read_json(ROOT/'checkpoint.json')
    selected={n:dict(v,test_source_sha256=file_hash(REPO/v['path'])) for n,v in completed['inventory'].items()
        if v['path'] in TARGETS and n.split('::')[-1].split('[')[0] in TARGETS[v['path']]}
    name='tests/test_release_support_scope.py'
    reviewed={'test_matrix_api_and_four_language_help_share_release_scope',
        'test_deferred_cuda_long_run_is_rejected_before_public_owned_and_worker_side_effects',
        'test_cpu_and_short_gpu_scope_remain_independent',
        'test_native_preflight_returns_controlled_unsupported_and_cli_exit_code'}
    declarations={n.name for n in ast.parse((REPO/name).read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}
    if not reviewed<=declarations:raise RuntimeError('reviewed scope regression missing')
    for function in sorted(reviewed):
        node=name+'::'+function
        selected[node]=dict(nodeid=node,path=name,profile='cpu',test_source_sha256=file_hash(REPO/name))
    for path,names in TARGETS.items():
        if names-{n.split('::')[-1].split('[')[0] for n in selected if n.split('::')[0]==path}:
            raise RuntimeError('targeted regression identity missing: '+path)
    if any(v.get('risk_class') for v in selected.values()):raise RuntimeError('risk node entered fixed selection')
    return selected


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    config=read_json(Path(__file__).with_name('campaign_config.json'));check_config(config)
    inventory=selection();sealed=snapshot();users=user_snapshot()
    plan=dict(scope='FIXED_CPU_SUPPORT_REGRESSIONS_ONLY',selected_nodes=list(inventory),test_count=len(inventory),
        source_snapshot=sealed,package_scope='Only targeted current wheel/sdist guide/payload checks; no 36-case E2E rerun',
        completed_general_campaign_unchanged=True,no_cuda=True,no_stress=True,no_long_run=True,no_retries=True)
    atomic_json(OUT/'targeted-plan.json',plan)
    if args.prepare_only:
        print(json.dumps(dict(status='PREPARED',test_count=len(inventory),tests_executed=0)));return 0
    live=resources()
    if live['available_commit_bytes']<config['start_available_commit_bytes'] or live['disk_free_bytes']<config['minimum_start_disk_bytes']:
        print('BLOCKED_RESOURCE: targeted verification not admitted');return 3
    runtime=runtime_plan()
    if runtime['missing']:raise RuntimeError('UNSUPPORTED_RUNTIME: '+', '.join(runtime['missing']))
    runtime=dict(runtime,python=runtime['cpu_python'])
    lock=(ROOT/'manager.lock').open('a+b');lock.seek(0)
    if lock.read(1)==b'':lock.write(b'0');lock.flush()
    lock.seek(0)
    try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:lock.close();print('Existing manager owns the lock');return 3
    try:
        attempt_id=uuid.uuid4().hex;folder=OUT/('attempt-'+attempt_id);folder.mkdir()
        scratch=Path(tempfile.mkdtemp(prefix='hyso-support-scope-'));work=scratch/'worker'
        request=dict(schema_version=1,evidence_protocol='hyso.event-journal.v2',mode='tests',profile='cpu',
            campaign_id='support-scope-'+attempt_id,attempt_id=attempt_id,group='support-scope-fixed',
            source_commit=git('rev-parse','HEAD'),environment_id=digest(runtime['runtime_paths']),token=uuid.uuid4().hex,
            started_utc=utc(),work=str(work),scratch_root=str(scratch),inventory=inventory,selected_nodes=list(inventory),
            blocked_nodes=[n for n,v in read_json(ROOT/'checkpoint.json')['inventory'].items() if v.get('risk_class')],
            known_nodes=list(inventory),files=sorted({v['path'] for v in inventory.values()}),
            timeout_seconds=1800,no_progress_timeout_seconds=300,initial_evidence_bytes=tree_bytes(ROOT),
            collection_dependencies_sha256=__import__('campaign_collection').collection_dependencies(sealed),
            conftest_sha256={n:sha for n,sha in sealed.items() if Path(n).name=='conftest.py'},
            argv=[str(Path(__file__).with_name('campaign_runner.py')),str(folder/'request.json')])
        with ExecutionState():
            receipt=execute(folder,request,config,runtime,update=lambda *a:None,stop_requested=lambda:False,
                integrity=lambda:snapshot()==sealed and user_snapshot()==users)
        audit=reconcile_attempt(folder,baseline=dict(commit=request['source_commit'],source_sha256=sealed))
        counts=dict(Counter(v['status'] for v in audit['results'].values()))
        good=(not audit['global_errors'] and receipt['exit_code']==0 and receipt['cleanup_verified']
            and len(audit['results'])==len(inventory) and counts=={'PASS':len(inventory)})
        atomic_json(folder/'targeted-result.json',dict(status='PASS' if good else 'FAIL',counts=counts,
            issues=audit['global_errors'],results=audit['results'],receipt=receipt,
            note='Targeted current-source verification; original general campaign results remain unchanged.'))
        if receipt['cleanup_verified']:
            cleanup_owned(work,scratch,request['campaign_id']);scratch.rmdir()
        print(json.dumps(dict(status='PASS' if good else 'FAIL',counts=counts,evidence=str(folder))))
        return 0 if good else 1
    finally:
        lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_UNLCK,1);lock.close()


if __name__=='__main__':raise SystemExit(main())
