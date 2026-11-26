"""Independent command-line campaign. --prepare-only performs no test execution."""
from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import tempfile
import time
import traceback
import uuid
import zipfile
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (LEGACY, REPO, ROOT, atomic_json, digest, file_hash, git, new_json,
                    read_json, read_journal, source_snapshot, user_snapshot, utc)
from campaign_checkpoint import Checkpoint, aggregate, journal_results
from campaign_inventory import prepared_inventory, import_legacy, v3_coverage
from campaign_preflight import check_config, runtime_plan, resources, process_identity, same_process, ExecutionState
from campaign_reporting import render
from campaign_scheduler import Scheduler
from campaign_collection import (migrate_nodeids,cached_collection,reconcile_collection,
    collection_dependencies,discover_unreviewed_files)
from campaign_reconciliation import protected_inputs, preserve_legacy, validate_checkpoint_evidence
from campaign_evidence import recover_completed_attempts,provisional_progress
from campaign_operations import (prepare_package_workspace, package_commands, install_commands,
    audit_archives, copy_package_reports, cleanup_owned, expected_package_cases, mirror_package_journals)


def log(event, **fields):
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'campaign.log').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(dict(time_utc=utc(), event=event, **fields), ensure_ascii=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def initialize(refresh=False):
    config = read_json(Path(__file__).with_name('campaign_config.json'))
    check_config(config)
    inventory = prepared_inventory()
    adopted, review = import_legacy(inventory, config)
    runtime = runtime_plan()
    fingerprint = source_snapshot()
    existing = read_json(ROOT / 'checkpoint.json')
    baseline = dict(campaign_id=(existing or {}).get('baseline', {}).get('campaign_id', uuid.uuid4().hex),
        prepared_utc=utc(), commit=git('rev-parse', 'HEAD'), branch=git('branch', '--show-current'),
        source_sha256=fingerprint, source_fingerprint=digest(fingerprint),
        user_files=user_snapshot(), config=config, runtime=runtime)
    cp = Checkpoint(ROOT, inventory, baseline)
    validate_checkpoint_evidence(cp)
    if existing:
        cp.data.setdefault('reviewed_collection_baselines',{})[existing['baseline']['source_fingerprint']]=collection_dependencies(existing['baseline']['source_sha256'])
    if existing and existing['baseline']['source_sha256'] != fingerprint and not refresh:
        raise RuntimeError('EVIDENCE_INVALID: prepared source changed after campaign attempts; preserve reports and prepare a reviewed source transition')
    aliases=migrate_nodeids(cp,inventory,reviewed_changes=read_json(Path(__file__).with_name('retired-tests.json'),{}))
    if existing and existing['baseline']['source_sha256'] != fingerprint and existing.get('attempts'):
        before = existing['baseline']['source_sha256']
        changed = sorted(n for n in set(before)|set(fingerprint) if before.get(n)!=fingerprint.get(n))
        # Explicit -PrepareOnly transition records every old result and conservatively
        # invalidates affected scopes. The normal unattended run never changes source.
        transition = dict(time_utc=utc(), old_baseline=existing['baseline'], changed_files=changed,
            results=existing['results'],package_operations=existing.get('package_operations', {}),
            infrastructure_results=existing.get('infrastructure_results',{}),nodeid_aliases=aliases)
        new_json(ROOT/'source-transitions'/('transition-'+uuid.uuid4().hex+'.json'), transition)
        production_changed = any(n.startswith(('hyso/', 'hyso_guide')) for n in changed)
        defect_plan=read_json(Path(__file__).with_name('defect-revalidation.json'),{})
        narrow_isolated_fix=({n for n in changed if n.startswith(('hyso/','hyso_guide'))}=={'hyso/analysis/isolated_bridge.py'}
            and file_hash(REPO/'hyso/analysis/isolated_bridge.py')==defect_plan.get('reviewed_product_sha256'))
        reviewed_tests=defect_plan.get('reviewed_test_sha256',{})
        selective_test_fix=not production_changed and all(
            before.get(n)==defect_plan.get('previous_test_sha256',{}).get(n)
            and fingerprint.get(n)==reviewed_tests.get(n)
            for n in changed if n.startswith('tests/') and n.endswith('.py'))
        infrastructure_changed = any(n.startswith('tools/final_acceptance/') or n=='run_final_acceptance.ps1' for n in changed)
        for node in list(cp.data['results']):
            path=node.split('::')[0]
            changed_test=path in changed and not (selective_test_fix and path in reviewed_tests
                and node not in defect_plan['nodeids'] and path not in defect_plan['test_files'])
            if production_changed and not narrow_isolated_fix or changed_test:
                cp.data['results'][node] = dict(nodeid=node,status='NOT_RUN',reason_code='SOURCE_CHANGED_REQUIRES_VALIDATION')
        cp.data['baseline'] = baseline
        cp.data['inventory'] = {r['nodeid']:r for r in inventory}
        cp.data['completed_operations'] = {}
        if production_changed or baseline['commit'] != existing['baseline']['commit']:
            cp.data['package_operations'] = {}
        if infrastructure_changed:
            cp.data.setdefault('infrastructure_validation_history',[]).append(dict(
                source_fingerprint=existing['baseline']['source_fingerprint'],results=existing.get('infrastructure_results',{})))
            for value in cp.data.get('infrastructure_results',{}).values():
                value['requires_runtime_revalidation']=True
        cp.data.setdefault('source_transitions', []).append(str(ROOT/'source-transitions'))
    if not existing or not existing.get('attempts'):
        cp.data['baseline'] = baseline
        cp.data['inventory'] = {r['nodeid']: r for r in inventory}
        cp.data['results'] = adopted
        cp.data['legacy_review'] = review
    relinked=[]
    for node,value in adopted.items():
        if cp.data['results'].get(node,{}).get('status','NOT_RUN')=='NOT_RUN':
            cp.data['results'][node]=value
            relinked.append(dict(nodeid=node,status=value['status'],evidence=value['evidence_file'],sha256=value['evidence_sha256']))
    if relinked:cp.data.setdefault('legacy_relink_history',[]).append(dict(time_utc=utc(),records=relinked))
    cp.data['legacy_review']=review
    if aliases:
        new_json(ROOT/'nodeid-migrations'/('migration-'+uuid.uuid4().hex+'.json'),dict(aliases=aliases,relinked=relinked))
    # A previously sealed CPU collection is useful after an infrastructure-only fix.
    # Revalidate original commit and source hashes; never execute pytest in preparation.
    for profile in ('cpu','stdlib'):
        cached=cached_collection(cp,profile)
        if cached:
            result=reconcile_collection(cp,cached['rows'],profile,evidence=cached['evidence'],errors=cached['errors'])
            cp.data['completed_operations']['collection-'+profile]=dict(status=result['status'],
                attempt=cached['folder'],evidence=cached['evidence'],reconciliation_schema=2)
    discover_unreviewed_files(cp,config['source_roots'])
    recover_completed_attempts(cp)
    defect_plan=read_json(Path(__file__).with_name('defect-revalidation.json'),{})
    if defect_plan and cp.data.get('applied_defect_plan')!=digest(defect_plan):
        selected=[n for n,e in cp.data['inventory'].items() if e['path'] in defect_plan['test_files'] or n in defect_plan['nodeids']]
        cp.data.setdefault('defect_revalidation_history',[]).append(dict(plan=defect_plan,
            previous_results={n:cp.data['results'].get(n) for n in selected}))
        for node in selected:cp.data['results'][node]=dict(nodeid=node,status='NOT_RUN',reason_code='DEFECT_REVALIDATION_REQUIRED')
        package_nodes=defect_plan.get('package_nodeids',[])
        if package_nodes:
            previous_package=read_json(ROOT/'package_acceptance.json',{})
            new_json(ROOT/'source-transitions'/('package-revalidation-'+uuid.uuid4().hex+'.json'),previous_package)
            for node in package_nodes:
                cp.data.setdefault('package_case_revalidation',{})[node]=dict(
                    excluded_attempt_ids=list(cp.data['attempts']),plan_sha256=digest(defect_plan))
            for row in previous_package.get('cases',[]):
                if row['nodeid'] in package_nodes:
                    row.update(status='NOT_RUN',reason_code='DEFECT_REVALIDATION_REQUIRED')
            previous_package.update(status='PARTIAL',test_counts=dict(Counter(r['status'] for r in previous_package.get('cases',[]))),
                unique_executed_cases=sum(r['status']!='NOT_RUN' for r in previous_package.get('cases',[])))
            atomic_json(ROOT/'package_acceptance.json',previous_package)
        cp.data['applied_defect_plan']=digest(defect_plan)
        atomic_json(ROOT/'defect-revalidation-selection.json',selected)
    coverage = v3_coverage(inventory, cp.data['results'])
    coverage['repair'] = 'The scheduler assigns every pending safe node directly from inventory, including unassigned V3 nodes; selection masks are not test results.'
    planner=Scheduler(cp,runtime,ROOT/'PREPARATION_ONLY_NO_WORKER',lambda *args:None,lambda:False,lambda:True)
    groups=planner.source_groups()
    assigned=[n for group in groups for n in group]
    pending_cpu={n for n in cp.pending() if cp.data['inventory'][n]['profile']=='cpu'}
    coverage['unassigned_after_repair'] = sorted(pending_cpu-set(assigned))
    if set(assigned)!=pending_cpu or len(assigned)!=len(set(assigned)):
        raise RuntimeError('inventory assignment is incomplete or overlapping')
    atomic_json(ROOT/'preparation/execution-plan.json',dict(
        source_groups=[dict(name='source-'+str(i),nodes=group,timeout_seconds=config['source_group_timeout_seconds']) for i,group in enumerate(groups,1)],
        pending_cpu=len(assigned),unknown_or_overlapping=0,
        forensic_additional_case=config['forensic_cpu_case'],forensic_additional_attempt_limit=1,
        forensic_already_reserved=cp.data['forensic_verification_reserved'],
        package_case_count=len(expected_package_cases()),native_risky_nodes='BLOCKED_NO_EXECUTION',new_P31_run=False))
    atomic_json(ROOT / 'preparation/inventory-review.json', coverage)
    atomic_json(ROOT / 'preparation/legacy-evidence-index.json', review['evidence_bindings'])
    preservation=preserve_legacy(review)
    cp.data['historical_bindings_valid']=preservation['historical_bindings_valid']
    original_users=read_json(LEGACY/'initial-state.json',{}).get('user_files',{})
    original_now=user_snapshot()
    original_check={n:dict(expected_sha256=b['sha256'],actual_sha256=original_now.get(n),
        unchanged=b['sha256']==original_now.get(n)) for n,b in original_users.items()}
    cp.data['original_user_files_check']=original_check
    cp.data['protected_inputs']=protected_inputs()
    atomic_json(ROOT / 'environment.json', dict(prepared_runtime=runtime,
        source_commit=baseline['commit'], git_status=git('status', '--short'),
        CPU_profile='CPU torch with real CUDA initialization fenced',
        stdlib_profile=runtime['forensic_profile'], source_imports='checkout only; installed package E2E uses external clean environments',
        source_changed_since_legacy=review['source_delta']))
    infra_path = 'tools/final_acceptance/tests/test_campaign_contracts.py'
    import ast
    tree = ast.parse((REPO/infra_path).read_text(encoding='utf-8'))
    cp.data['infrastructure_inventory'] = {
        infra_path+'::'+n.name: dict(nodeid=infra_path+'::'+n.name,path=infra_path,
            line=n.lineno, test_source_sha256=file_hash(REPO/infra_path),profile='infrastructure')
        for n in tree.body if isinstance(n, ast.FunctionDef) and n.name.startswith('test_')}
    cp.data.setdefault('infrastructure_results', {})
    atomic_json(ROOT/'user_file_preservation.json', dict(start=cp.data['baseline']['user_files'],
        current=user_snapshot(), user_files_unchanged=user_snapshot()==cp.data['baseline']['user_files'],
        original_user_files=original_check, protected_model_and_data_inputs=cp.data['protected_inputs'],
        source_unchanged=fingerprint==cp.data['baseline']['source_sha256'], scope='STATIC_PREPARATION'))
    atomic_json(ROOT/'resource_usage.json', dict(last_sample=resources(), limits=config, execution_started=False))
    log('PREPARATION_RECONCILED', inventory=len(inventory), counts=cp.counts())
    cp.save()
    return cp, runtime


def recover(cp):
    """Owner-death recovery never terminates a PID based on its number alone."""
    for attempt_id, attempt in cp.data['attempts'].items():
        if attempt.get('status') != 'RUNNING':
            continue
        folder = Path(attempt['directory'])
        receipt = read_json(folder / 'controller.json')
        request=read_json(folder/'request.json',{})
        if receipt is None:
            admission = read_json(folder / 'admission-receipt.json', {})
            launch = read_json(folder/'launch-receipt.json',{})
            owner = admission.get('owner') or launch.get('owner')
            identities = admission.get('identities', {})
            if owner and same_process(owner):
                raise RuntimeError('existing owned controller is still running; refusing overlapping campaigns')
            binding=launch.get('launch_binding',{})
            members = [identities.get('launcher'), identities.get('interpreter'),binding.get('identity')]
            if any(member and same_process(member) for member in members):
                raise RuntimeError('OWNERSHIP_UNVERIFIED: old admitted identity is still alive; no unowned process will be terminated')
            verified = bool(owner and admission.get('kill_on_close') and admission.get('exact_membership')
                            and admission.get('limits_readback') and admission.get('no_breakaway'))
            launch_verified=bool(owner and launch.get('schema_version')==2 and launch.get('kill_on_close')
                and binding.get('created_suspended') and binding.get('exact_job_membership_before_resume')
                and binding.get('limits_readback') and binding.get('no_breakaway')
                and binding.get('identity') and binding['identity']['pid']==binding.get('launcher_pid'))
            verified=verified or launch_verified
            admit=read_json(folder/'admit.json',{})
            admitted=bool(admission and admit.get('pid')==identities.get('interpreter',{}).get('pid')
                and digest(admit.get('token'))==identities.get('nonce_sha256'))
            receipt = dict(schema_version=2 if launch.get('schema_version')==2 else 1,
                payload_admitted=admitted,attempt_id=attempt_id, monitor_reason='CONTROLLER_INTERRUPTED',
                cleanup_verified=verified, source_integrity=True,
                cleanup=None,cleanup_proof=dict(kind='OWNER_DEATH_KILL_ON_CLOSE',
                    kill_on_close=bool(launch.get('kill_on_close') or admission.get('kill_on_close')),
                    no_breakaway=bool(binding.get('no_breakaway') or admission.get('no_breakaway')),
                    identities=[owner,*[m for m in members if m]]),
                identities=identities,launch_binding=binding,
                request_sha256=file_hash(folder/'request.json'),
                limits=dict(process_bytes=cp.data['baseline']['config']['process_memory_bytes'],
                    job_bytes=cp.data['baseline']['config']['job_memory_bytes'],max_processes=cp.data['baseline']['config']['max_processes']),
                limits_source='Recorded fixed policy plus original launch limit readback; no new native measurement',
                recovery='Exact owner/interpreter creation identities are absent; non-inherited kill-on-close Job contract preserved.')
            new_json(folder / ('recovery-' + uuid.uuid4().hex + '.json'), receipt)
        receipt['source_integrity'] = bool(receipt.get('source_integrity') and
            (attempt.get('source_fingerprint')==cp.data['baseline']['source_fingerprint']
             or request.get('collection_dependencies_sha256')==collection_dependencies(cp.data['baseline']['source_sha256'])))
        results, issues = journal_results(folder / 'events.jsonl', receipt)
        if any(i['reason'] == 'EVIDENCE_INVALID' for i in issues):
            raise RuntimeError('EVIDENCE_INVALID: broken checkpoint journal')
        target = 'infrastructure_results' if attempt.get('profile')=='infrastructure' else 'results'
        inventory = cp.data.get('infrastructure_inventory', {}) if target!='results' else cp.data['inventory']
        for node, value in results.items():
            if attempt.get('profile') not in ('cpu','stdlib') and node in inventory and value['status'] != 'NOT_RUN':
                value['evidence_sha256']=file_hash(folder/'events.jsonl')
                cp.data.setdefault(target,{})[node] = value
        attempt.update(status='RECOVERED', recovery=receipt)
        if (folder/'events.jsonl').is_file():attempt['journal_sha256']=file_hash(folder/'events.jsonl')
        if (folder/'controller.json').is_file():attempt['controller_sha256']=file_hash(folder/'controller.json')
        cp.data['interruptions'].append(dict(attempt_id=attempt_id, reason='RECOVERED_OWNER_INTERRUPTION', time_utc=utc()))
    cp.save()


def package_state(cp, work, steps):
    rows, bindings = {}, {}
    for path in sorted(Path(work).glob('*-external/E2E/package-events-*.jsonl'), key=lambda p: p.stat().st_mtime_ns):
        attempt_id = path.stem.replace('package-events-', '')
        record = cp.data['attempts'].get(attempt_id, {})
        receipt = record.get('controller') or record.get('recovery')
        values, issues = journal_results(path, receipt)
        if issues and any(i['reason']=='EVIDENCE_INVALID' for i in issues):
            raise RuntimeError('EVIDENCE_INVALID: package phase journal')
        for node, value in values.items():
            excluded=cp.data.get('package_case_revalidation',{}).get(node,{}).get('excluded_attempt_ids',[])
            if attempt_id in excluded:
                continue  # Keep original FAIL evidence; only a new bound attempt can replace it.
            if value['status'] != 'NOT_RUN' and receipt and receipt.get('cleanup_verified'):
                rows[node] = value
    for kind in ('wheel', 'sdist'):
        terminal, files = {}, {}
        for node, value in rows.items():
            if node.startswith('package:' + kind + '::'):
                terminal[node.split('::')[-1]] = value['status']
                if value['status'] == 'PASS':
                    for phase in value['phases']:
                        files.update(phase.get('artifact_bindings', {}))
        atomic_json(Path(work) / (kind + '-resume.json'), dict(terminal_cases=terminal, artifact_bindings=files))
    for node in expected_package_cases():
        rows.setdefault(node, dict(nodeid=node,status='NOT_RUN',reason_code='PACKAGE_CASE_NOT_COMPLETED'))
    counts = Counter(v['status'] for v in rows.values())
    archives = audit_archives(work) if (Path(work) / 'source/commercial-profile.json').exists() else []
    all_ops_pass = all(value.get('status') == 'PASS' for value in steps.values())
    status = 'PASS' if len(rows) == 36 and counts['PASS'] == 36 and all_ops_pass and len(archives)==2 and all(a['status']=='PASS' for a in archives) else 'PARTIAL'
    if any(v['status'] in ('FAIL', 'ERROR') for v in rows.values()) or any(v.get('status')=='FAIL' for v in steps.values()):
        status = 'FAIL'
    result = dict(status=status, source_commit=cp.data['baseline']['commit'], planned_cases=36,
        unique_executed_cases=sum(v['status']!='NOT_RUN' for v in rows.values()), test_counts=dict(counts), cases=list(rows.values()),
        operations=steps, archive_audit=archives, clean_system_site_packages=False,
        prior_c389_evidence_preserved=read_json(LEGACY / 'installed-packages.json', []),
        notes=['Package and source test denominators are distinct.',
               'No production/model code is changed automatically.',
               'Clean E2E source paths and CPU-only Torch are asserted by the retained installed harness.'])
    atomic_json(ROOT / 'package_acceptance.json', result)
    return result


def run_packages(cp, scheduler):
    current = read_json(ROOT / 'package_acceptance.json', {})
    if current.get('source_commit') == cp.data['baseline']['commit'] and current.get('status') == 'PASS':
        return current
    if cp.data.get('package_fixture_state_lost'):
        result=dict(current,status='PARTIAL',reason_code='PACKAGE_FIXTURE_STATE_LOST',
            note='Completed case evidence is retained. Missing TEMP state prevents safe continuation; no completed training case is silently rerun.')
        atomic_json(ROOT/'package_acceptance.json',result)
        return result
    if scheduler.stop():
        return current
    work = prepare_package_workspace(scheduler.scratch, cp.data['baseline']['campaign_id'], cp.data['baseline']['commit'])
    steps = cp.data.setdefault('package_operations', {})
    def operation(step, package_sha=None):
        name = step['name']
        if name in steps and steps[name]['status'] in ('PASS', 'FAIL', 'ERROR'):
            if name.endswith('-E2E') and steps[name]['status']!='PASS':
                existing_cases=package_state(cp,work,steps).get('cases',[])
                kind=name.split('-')[0]
                incomplete=any(r['nodeid'].startswith('package:'+kind+'::') and r['status']=='NOT_RUN' for r in existing_cases)
                if not incomplete:return False
                cp.data.setdefault('package_operation_history',[]).append(dict(name=name,previous=steps[name]))
            else:
                return steps[name]['status']=='PASS'
        if step.get('internal') == 'extract':
            try:
                with zipfile.ZipFile(work / 'source.zip') as archive:
                    # Git archive paths are validated before materializing our own scratch copy.
                    if any(Path(n).is_absolute() or '..' in Path(n).parts for n in archive.namelist()):
                        raise ValueError('unsafe Git archive path')
                    archive.extractall(work / 'source')
                steps[name] = dict(status='PASS', source_commit=cp.data['baseline']['commit'])
                cp.save()
                return True
            except BaseException:
                steps[name] = dict(status='FAIL', traceback=traceback.format_exc())
                cp.save()
                return False
        metadata = dict(package_sha256=package_sha) if package_sha else {}
        outcome = None
        for retry in range(scheduler.config['max_resource_interruptions_per_group']+1):
            outcome = scheduler.attempt(name, mode='command', command=step['command'], cwd=step['cwd'],
                timeout=step.get('timeout', 900), metadata=metadata)
            if not outcome or outcome['receipt']['monitor_reason'] not in ('BLOCKED_RESOURCE','DISK_RESERVE'):
                break
            cp.data['retries'].append(dict(group=name, reason=outcome['receipt']['monitor_reason'],
                retry=retry+1,time_utc=utc(),scope='Incomplete operation only; package tests retain terminal cases'))
            cp.save()
            if name.endswith('-E2E'):
                package_state(cp, work, steps)
        if outcome is None:
            steps[name] = dict(status='BLOCKED', reason_code='BLOCKED_RESOURCE')
            cp.save()
            return False
        code = outcome['receipt']['exit_code']
        reason = outcome['receipt']['monitor_reason']
        good = code == 0 and reason is None and outcome['receipt']['cleanup_verified']
        transient = reason in ('BLOCKED_RESOURCE','DISK_RESERVE','USER_STOP')
        steps[name] = dict(status='PASS' if good else 'NOT_RUN' if transient else 'FAIL',reason_code=reason,
                          evidence=str(Path(outcome['folder'])/'controller.json'),
                          source_commit=cp.data['baseline']['commit'])
        cp.save()
        return good
    builds = package_commands(work, scheduler.runtime, cp.data['baseline']['commit'])
    archive_ok = operation(builds[0])
    extract_ok = operation(builds[1]) if archive_ok else False
    if extract_ok:
        for step in builds[2:]:
            if not scheduler.stop():
                operation(step)
    for kind in ('wheel', 'sdist'):
        commands, artifact = install_commands(work, scheduler.runtime, kind)
        prerequisite = bool(commands)
        for step in commands:
            if scheduler.stop():
                break
            if step['name'].endswith('-E2E'):
                package_state(cp, work, steps)
                step['command'] += ['--resume-state', str(work / (kind+'-resume.json'))]
            if prerequisite:
                prerequisite = operation(step, artifact['sha256'])
            else:
                steps.setdefault(step['name'], dict(status='BLOCKED', reason_code='DEPENDENCY_OPERATION_FAILED'))
        copy_package_reports(work, ROOT / 'package-reports')
    result = package_state(cp, work, steps)
    # Persist phase journals and distribution hashes; large environments stay outside the bundle.
    for path in Path(work).glob('*-external/E2E/package-events-*.jsonl'):
        target = ROOT / 'package-journals' / path.name
        target.parent.mkdir(exist_ok=True)
        if not target.exists():
            target.write_bytes(path.read_bytes())
    for kind in ('wheel','sdist'):
        evidence = work/(kind+'-external/installed-payload.json')
        if evidence.is_file():
            atomic_json(ROOT/('package-'+kind+'-installed-payload.json'),read_json(evidence))
    atomic_json(ROOT/'package-artifacts.json', audit_archives(work) if (work/'source/commercial-profile.json').exists() else [])
    return result


def _run_locked(prepare_only=False):
    invocation_utc=utc()
    cp, runtime = initialize(refresh=prepare_only)
    started = time.monotonic()
    elapsed_before=cp.data.get('elapsed_seconds_previous_runs',0)
    initial_pass = cp.counts()['PASS']
    cancelled = False
    root = ROOT
    progress = dict(status='PREPARED', started_utc=None, elapsed_seconds=0,
        previous_valid_PASS=initial_pass, new_PASS=0, active_group=None, active_nodeid=None,
        total_unique_tests=len(cp.data['inventory']), last_checkpoint=cp.data.get('updated_utc'),
        remaining_tests=len(cp.pending()), completion_percent=100*(len(cp.data['inventory'])-len(cp.pending()))/len(cp.data['inventory']))
    package = read_json(ROOT / 'package_acceptance.json')
    if prepare_only:
        progress['static_preparation_only'] = True
        atomic_json(ROOT / 'preparation/static-plan.json', dict(inventory=len(cp.data['inventory']),
            counts=cp.counts(), pending_safe_cpu=sum(cp.data['inventory'][n]['profile']=='cpu' for n in cp.pending()),
            forensic_inventory=sum(r['profile']=='stdlib' for r in cp.data['inventory'].values()),
            runtime=runtime, code_executed='stdlib offline parsing/hashing only; no pytest, Job launch, training or CUDA'))
        render(cp, progress, package, bundle=True)
        return 0
    recover(cp)
    recover_completed_attempts(cp)
    initial_pass=cp.counts()['PASS']
    progress['previous_valid_PASS']=initial_pass
    for node, value in list(cp.data['results'].items()):
        if value.get('reason_code') == 'BLOCKED_RESOURCE' or value.get('resume_incomplete'):
            cp.data['results'][node] = dict(nodeid=node, status='NOT_RUN', reason_code='RESOURCE_RESUME_PENDING')
    owner = process_identity(os.getpid())
    atomic_json(ROOT / 'owner.json', dict(owner=owner, started_utc=utc(), commit=cp.data['baseline']['commit']))
    stop_dir = ROOT / 'control'
    stop_dir.mkdir(exist_ok=True)
    seen_before = {p.name for p in stop_dir.glob('stop-*.json')
                   if read_json(p,{}).get('time_utc','') < invocation_utc}
    cp.data['processed_stop_requests'] = sorted(set(cp.data['processed_stop_requests']) | seen_before)
    cp.save()
    def stopped():
        return cancelled or any(p.name not in cp.data['processed_stop_requests'] for p in stop_dir.glob('stop-*.json'))
    def interrupt(*unused):
        nonlocal cancelled
        cancelled = True
    old_signal = signal.signal(signal.SIGINT, interrupt)
    baseline = cp.data['baseline']
    def integrity():
        return (git('rev-parse', 'HEAD') == baseline['commit']
            and source_snapshot()==baseline['source_sha256']
            and user_snapshot()==baseline['user_files']
            and all(r['status']=='PASS' for r in protected_inputs().values()))
    temp_root = cp.data.get('scratch')
    if not temp_root:
        temp_root = tempfile.mkdtemp(prefix='hyso-final-'+baseline['campaign_id'][:8]+'-')
        cp.data['scratch'] = temp_root
        cp.save()
    elif not Path(temp_root).is_dir():
        # Reboot/OS cleanup can remove TEMP. Old evidence remains; new scratch has a new identity.
        cp.data.setdefault('missing_scratch_history',[]).append(temp_root)
        if package and package.get('source_commit')==baseline['commit'] and package.get('unique_executed_cases',0):
            cp.data['package_fixture_state_lost']=True
        temp_root=tempfile.mkdtemp(prefix='hyso-final-'+baseline['campaign_id'][:8]+'-')
        cp.data['scratch']=temp_root
        cp.data['package_operations']={}
        cp.save()
    def update(status, scheduler):
        counts = cp.counts()
        progress.update(provisional_progress(cp,(scheduler.active or {}).get('attempt_id')))
        progress.update(status=status, elapsed_seconds=time.monotonic()-started,
            new_PASS=max(0, counts['PASS']-initial_pass), active_group=(scheduler.active or {}).get('group'),
            active_attempt_id=(scheduler.active or {}).get('attempt_id'),
            active_nodeid=(scheduler.active or {}).get('nodeid'), remaining_tests=len(cp.pending()),
            completion_percent=100*(len(cp.data['inventory'])-len(cp.pending()))/len(cp.data['inventory']),
            available_host_commit_bytes=scheduler.live.get('available_commit_bytes'),
            peak_process_commit_bytes=cp.data.get('resource_peaks',{}).get('peak_process_commit_bytes'),
            peak_job_commit_bytes=cp.data.get('resource_peaks',{}).get('peak_job_commit_bytes'),
            interruptions=len(cp.data['interruptions']), retries=len(cp.data['retries']),
            last_checkpoint=cp.data.get('updated_utc'),
            elapsed_total_seconds=elapsed_before+time.monotonic()-started)
        cp.data['elapsed_seconds_previous_runs']=elapsed_before+time.monotonic()-started
        atomic_json(ROOT / 'resource_usage.json', dict(last_sample=scheduler.live,peaks=cp.data.get('resource_peaks',{}),
            limits={k: baseline['config'][k] for k in ('start_available_commit_bytes','stop_available_commit_bytes','process_memory_bytes','job_memory_bytes','max_processes')}))
        atomic_json(ROOT/'progress.json',dict(progress, counts=counts))
        if time.monotonic()-getattr(scheduler,'last_full_render',0)>30 or status=='WAITING_RESOURCE':
            render(cp, progress, read_json(ROOT / 'package_acceptance.json'))
            scheduler.last_full_render = time.monotonic()
    scheduler = Scheduler(cp, runtime, temp_root, update, stopped, integrity)
    cp.data.setdefault('first_execution_utc',utc())
    progress['started_utc'] = cp.data['first_execution_utc']
    progress['current_invocation_started_utc']=utc()
    cp.save()
    log('CAMPAIGN_STARTED', commit=baseline['commit'], owner=owner)
    error = None
    try:
        with ExecutionState():
            critical_missing=[name for name in runtime['missing'] if name in
                ('CPU Python executable','verified CPU torch 2.5.1+cpu distribution','torch','pytest','numpy')]
            if critical_missing:
                raise RuntimeError('UNSUPPORTED_RUNTIME: '+', '.join(critical_missing))
            if any(not r['unchanged'] for r in cp.data['original_user_files_check'].values()) or any(
                    r['status']!='PASS' for r in protected_inputs().values()):
                raise RuntimeError('EVIDENCE_INVALID: protected user/model/data files differ from preserved baseline')
            if not integrity():
                raise RuntimeError('EVIDENCE_INVALID: source/user file snapshot changed')
            scheduler.run_startup_checks()
            if not scheduler.run_infrastructure():
                raise RuntimeError('Campaign infrastructure contract verification incomplete/failed; no product test launched')
            # Live collection is deferred until the user command and uses the same owned CPU/stdlib profiles.
            for profile in ('cpu','stdlib'):
                files=sorted({r['path'] for r in cp.data['inventory'].values() if r['profile']==profile})
                key = 'collection-'+profile
                cached_operation=cp.data['completed_operations'].get(key,{})
                if not cached_operation.get('reconciliation_schema')==2:
                    outcome = scheduler.attempt(key, mode='collect', profile=profile, files=files, timeout=900)
                    if not outcome:
                        for node,entry in cp.data['inventory'].items():
                            if entry['profile']==profile and cp.data['results'].get(node,{}).get('status','NOT_RUN')=='NOT_RUN':
                                cp.data.setdefault('inventory_holds',{})[node]=dict(reason_code='BLOCKED_RESOURCE',nodeid=node)
                        cp.save()
                        continue
                    path=Path(outcome['folder'])/'collected.json'
                    actual=read_json(path,[])
                    result=reconcile_collection(cp,actual,profile,
                        evidence=dict(path=str(path),sha256=file_hash(path) if path.is_file() else None),
                        errors=[e for e in outcome['events'] if e['kind'] in ('collection_error','framework_error')])
                    cp.data['completed_operations'][key]=dict(status=result['status'],attempt=outcome['folder'],reconciliation_schema=2)
                    cp.save()
            scheduler.run_evidence_canary()
            scheduler.run_source()
            if not stopped():
                scheduler.run_forensic()
            if not stopped():
                package = run_packages(cp, scheduler)
    except BaseException as exc:
        error = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        log('CAMPAIGN_STOPPED_WITH_ERROR', **error)
        progress['last_error'] = error
    finally:
        try:
            preservation = dict(start=baseline['user_files'], end=user_snapshot(), source_unchanged=source_snapshot()==baseline['source_sha256'])
            preservation['user_files_unchanged'] = preservation['start'] == preservation['end']
            preservation['protected_model_and_data_inputs']=protected_inputs()
            preservation['original_user_files']=cp.data['original_user_files_check']
            atomic_json(ROOT / 'user_file_preservation.json', preservation)
            cp.save()
            cp.data['elapsed_seconds_previous_runs'] = elapsed_before+time.monotonic()-started
            cp.save()
            progress.update(status='STOPPED' if stopped() else 'PARTIAL' if error or cp.pending() else 'FINISHED',
                elapsed_seconds=time.monotonic()-started, active_group=None, active_nodeid=None,
                remaining_tests=len(cp.pending()), last_checkpoint=cp.data.get('updated_utc'))
            log('REPORTS_WRITTEN', status=progress['status'], counts=cp.counts())
            render(cp, progress, package or read_json(ROOT/'package_acceptance.json'), final=True, bundle=True)
        finally:
            signal.signal(signal.SIGINT, old_signal)
    return 1 if error or cp.counts()['FAIL'] or cp.counts()['ERROR'] else 0


def run(prepare_only=False):
    # Lock before initializing or touching checkpoint state, including preparation.
    import msvcrt
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = (ROOT / 'manager.lock').open('a+b')
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    try:
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    except OSError:
        lock.close()
        print('A campaign manager already owns the lock; use -Status or -Stop.')
        return 3
    try:
        return _run_locked(prepare_only)
    except BaseException as exc:
        log('SAFE_STARTUP_FAILURE', exception=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        previous = read_json(ROOT/'progress.json', {})
        atomic_json(ROOT/'progress.json', dict(previous, status='PARTIAL', last_error=str(exc)))
        stored = read_json(ROOT/'checkpoint.json')
        if stored:
            try:
                cp = Checkpoint(ROOT, [], stored['baseline'])
                render(cp, dict(previous,status='PARTIAL',last_error=str(exc)),
                    read_json(ROOT/'package_acceptance.json'),final=True,bundle=True)
            except BaseException as report_exc:
                log('REPORT_GENERATION_ERROR',message_detail=str(report_exc))
        return 2
    finally:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)
        lock.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_mutually_exclusive_group()
    choices.add_argument('--prepare-only', action='store_true')
    choices.add_argument('--status', action='store_true')
    choices.add_argument('--stop', action='store_true')
    args = parser.parse_args()
    if args.status:
        print(json.dumps(read_json(ROOT/'progress.json', dict(status='NOT_STARTED')), ensure_ascii=False, indent=2))
        return 0
    if args.stop:
        new_json(ROOT/'control'/('stop-'+uuid.uuid4().hex+'.json'), dict(time_utc=utc(), reason='USER_STOP'))
        print('Safe stop requested; completed journals remain durable.')
        return 0
    return run(args.prepare_only)


if __name__ == '__main__':
    raise SystemExit(main())
