"""Serial scheduling, finite resource waiting and unique attempt directories."""
from __future__ import annotations

import json
import os
import time
import uuid
from collections import defaultdict
from pathlib import Path

from common import REPO, ROOT, atomic_json, digest, new_json, read_json, read_journal, utc, tree_bytes, file_hash
from campaign_checkpoint import journal_results
from campaign_controller import execute
from campaign_operations import harvest, cleanup_owned, mirror_package_journals
from campaign_preflight import resources
from campaign_collection import reconcile_collection, collection_dependencies
from campaign_evidence import reconcile_attempt,accept_attempt


class Scheduler:
    def __init__(self, checkpoint, runtime, scratch, report, stop, integrity):
        self.cp, self.runtime, self.scratch = checkpoint, runtime, Path(scratch)
        self.config = checkpoint.data['baseline']['config']
        self.report, self.stop, self.integrity = report, stop, integrity
        self.waited = 0.0
        self.active = None
        self.live = {}
        self.initial_evidence_bytes = tree_bytes(ROOT)

    def admitted_resources(self):
        interval = self.config['resource_backoff_seconds']
        index = 0
        while True:
            if self.stop():
                return False
            self.live = resources()
            enough = self.live['available_commit_bytes'] >= self.config['start_available_commit_bytes']
            enough = enough and self.live['disk_free_bytes'] >= self.config['minimum_start_disk_bytes']
            if enough:
                return True
            if self.waited >= self.config['resource_wait_budget_seconds']:
                return False
            pause = min(interval[min(index, len(interval)-1)], self.config['resource_wait_budget_seconds']-self.waited)
            self.report('WAITING_RESOURCE', self)
            begin = time.monotonic()
            while time.monotonic()-begin < pause:
                if self.stop():
                    return False
                time.sleep(min(1, max(.01, pause-(time.monotonic()-begin))))
            self.waited += time.monotonic()-begin
            index += 1
            self.cp.data['resource_wait_seconds'] = self.waited
            self.cp.save()

    def _update(self, attempt, live, usage):
        self.live = dict(live, job=usage)
        peaks=self.cp.data.setdefault('resource_peaks',{})
        for key in ('peak_process_commit_bytes','peak_job_commit_bytes'):
            peaks[key]=max(peaks.get(key,0),usage.get(key,0))
        usage_path=ROOT/'resource-samples.jsonl'
        if not usage_path.exists() or usage_path.stat().st_size<16*1024**2:
            import os
            with usage_path.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(self.live,time_utc=utc(),attempt=Path(attempt).name))+'\n')
                stream.flush()
                os.fsync(stream.fileno())
        path = Path(attempt) / 'events.jsonl'
        receipts = read_json(Path(attempt) / 'controller.json')
        request = read_json(Path(attempt) / 'request.json', {})
        target = 'infrastructure_results' if request.get('profile') == 'infrastructure' else 'results'
        inventory = self.cp.data.get('infrastructure_inventory', {}) if target != 'results' else self.cp.data['inventory']
        if request.get('mode')=='tests':
            audit=reconcile_attempt(attempt,baseline=self.cp.data['baseline'],receipt=receipts,provisional=True)
            values,issues=audit['results'],audit['stream']['issues']
            for value in values.values():value['provisional_evidence_warning']=audit['global_errors']
            self.active['evidence_warnings']=audit['global_errors']+audit['local_issues']
        else:
            values,issues=journal_results(path,receipts,allow_verified_prefix=receipts is None)
        self.cp.data['provisional_results']={n:v for n,v in self.cp.data.get('provisional_results',{}).items()
            if v.get('attempt_id')!=request.get('attempt_id')}
        for node, value in values.items():
            if node in inventory and value['status'] != 'NOT_RUN':
                # During an active owned Job, these are provisional until cleanup succeeds.
                value['provisional_until_cleanup'] = True
                self.cp.data.setdefault('provisional_results', {})[node] = value
        if request.get('mode') == 'command' and request.get('group', '').endswith('-E2E'):
            mirrored = mirror_package_journals(self.scratch, ROOT / 'package-journals')
            if mirrored:
                package_events, _ = read_journal(mirrored[-1])
                if package_events:
                    self.active['nodeid'] = package_events[-1].get('nodeid')
        events, _ = read_journal(path)
        if events:
            self.active['last_event_utc'] = events[-1]['time_utc']
            self.active['nodeid'] = events[-1].get('nodeid') or self.active.get('nodeid')
        self.cp.save()
        self.report('RUNNING', self)

    def attempt(self, group, *, files=None, nodes=None, profile='cpu', mode='tests',
                command=None, cwd=None, timeout=None, metadata=None):
        if not self.admitted_resources():
            return None
        if not self.integrity():
            raise RuntimeError('EVIDENCE_INVALID: prepared source or user files changed')
        attempt_id = uuid.uuid4().hex
        folder = ROOT / 'attempts' / attempt_id
        folder.mkdir(parents=True, exist_ok=False)
        work = self.scratch / ('attempt-' + attempt_id)
        selected = list(nodes or [])
        inventory = self.cp.data.get('infrastructure_inventory', {}) if profile == 'infrastructure' else self.cp.data['inventory']
        target = 'infrastructure_results' if profile == 'infrastructure' else 'results'
        entries = {node: inventory[node] for node in selected}
        runtime = dict(self.runtime, python=self.runtime['cpu_python'])
        if profile == 'stdlib':
            if not self.runtime.get('forensic_python'):
                return None
            runtime.update(python=self.runtime['forensic_python'],
                runtime_paths=self.runtime['fallback_test_dependencies'])
        request = dict(schema_version=1, mode=mode, profile=profile,
            collection_dependencies_sha256=collection_dependencies(self.cp.data['baseline']['source_sha256']),
            conftest_sha256={n:sha for n,sha in self.cp.data['baseline']['source_sha256'].items() if Path(n).name=='conftest.py'},
            attempt_id=attempt_id, group=group, campaign_id=self.cp.data['baseline']['campaign_id'],
            started_utc=utc(), source_commit=self.cp.data['baseline']['commit'],
            environment_id=digest(dict(paths=runtime['runtime_paths'], versions=runtime['versions'], profile=profile)),
            token=uuid.uuid4().hex, work=str(work), inventory=entries,
            scratch_root=str(self.scratch), initial_evidence_bytes=self.initial_evidence_bytes,
            selected_nodes=selected, blocked_nodes=[n for n,r in self.cp.data['inventory'].items() if r.get('risk_class')],
            known_nodes=list(inventory),
            files=files or [], timeout_seconds=timeout or self.config['source_group_timeout_seconds'],
            no_progress_timeout_seconds=self.config['source_no_progress_timeout_seconds'],
            argv=[str(Path(__file__).with_name('campaign_runner.py')), str(folder / 'request.json')])
        request['evidence_protocol']='hyso.event-journal.v2'
        if mode == 'collect':
            request['inventory'] = {n:r for n,r in inventory.items() if r['profile']==profile}
            request['selected_nodes'] = []
        if mode == 'command':
            request.update(command=command, cwd=str(cwd), command_timeout_seconds=timeout or 900)
        request.update(metadata or {})
        self.active = dict(group=group, attempt_id=attempt_id, nodeid=None, directory=str(folder))
        self.cp.data['attempts'][attempt_id] = dict(group=group, directory=str(folder),
            started_utc=request['started_utc'], source_commit=request['source_commit'],
            mode=mode, profile=profile, selected_nodes=selected, status='RUNNING',
            source_fingerprint=self.cp.data['baseline']['source_fingerprint'])
        self.cp.save()
        try:
            receipt = execute(folder, request, self.config, runtime, update=self._update,
                              stop_requested=self.stop, integrity=self.integrity)
        except BaseException:
            # Keep the safety failure and its real reason durable before propagating.
            receipt=read_json(folder/'controller.json')
            if receipt:
                record=self.cp.data['attempts'][attempt_id]
                record.update(status='STOPPED_SAFETY',controller=receipt,
                    controller_sha256=file_hash(folder/'controller.json'))
                self.cp.data['interruptions'].append(dict(attempt_id=attempt_id,group=group,
                    reason=receipt.get('monitor_reason'),safety_error='OWNED_CLEANUP_UNVERIFIED'
                        if not receipt.get('cleanup_verified') else 'CONTROLLER_INTERRUPTED',time_utc=utc()))
                self.cp.save()
            raise
        harvested = harvest(work, folder)
        if mode=='tests':
            record=self.cp.data['attempts'][attempt_id]
            record.update(status='FINISHED',controller=receipt,
                controller_sha256=file_hash(folder/'controller.json'),
                journal_sha256=file_hash(folder/'events.jsonl'))
            audit,accepted=accept_attempt(self.cp,folder,receipt=receipt)
            values,issues=audit['results'],audit['stream']['issues']
        else:
            values,issues=journal_results(folder/'events.jsonl',receipt)
        journal_sha = file_hash(folder/'events.jsonl') if (folder/'events.jsonl').is_file() else None
        for node in selected:
            value = values.get(node)
            if value:
                value.update(provisional_until_cleanup=False, controller_evidence=str(folder / 'controller.json'),
                    evidence_sha256=journal_sha)
                if value['status'] in ('NOT_RUN','TIMEOUT') and not value.get('phases_complete'):
                    value['resume_incomplete']=node!=self.config['forensic_cpu_case']
                # accept_attempt is the only finalization path for real tests.
                if mode!='tests':self.cp.data.setdefault(target, {})[node] = value
                self.cp.data.setdefault('provisional_results', {}).pop(node, None)
        events, _ = read_journal(folder / 'events.jsonl')
        errors = [r for r in events if r['kind'] in ('collection_error', 'framework_error', 'integrity_error')]
        for error in errors:
            if error['kind'] == 'integrity_error':
                raise RuntimeError('EVIDENCE_INVALID: ' + error.get('reason', 'runner integrity failure'))
            if error['kind']=='framework_error':continue
            affected = error.get('collection_nodeid', '').split('::')[0]
            for node in selected:
                if node not in values and (not affected or node.split('::')[0] == affected):
                    self.cp.data.setdefault(target, {})[node] = dict(nodeid=node, status='ERROR', reason_code='SETUP_ERROR',
                        failure=dict(when='setup', traceback=error.get('traceback', '')), attempt_id=attempt_id,
                        source_commit=request['source_commit'], environment_id=request['environment_id'],
                        evidence_file=str(folder / 'events.jsonl'))
        reason = receipt['monitor_reason']
        record = self.cp.data['attempts'][attempt_id]
        record.update(status='FINISHED', controller=receipt, evidence_issues=issues, harvested=harvested)
        record['controller_sha256']=file_hash(folder/'controller.json')
        record['journal_sha256']=journal_sha
        if (folder/'collected.json').is_file():
            record['collected_sha256']=file_hash(folder/'collected.json')
        if mode=='tests' and (folder/'collected.json').is_file():
            # Deselecting unknown nodes remains mandatory if dynamic collection
            # changes after preflight; hold only this group's missing/duplicate nodes.
            reconcile_collection(self.cp,read_json(folder/'collected.json'),profile,
                evidence=dict(path=str(folder/'collected.json'),sha256=record['collected_sha256']),expected_nodes=selected)
        mirror_package_journals(self.scratch, ROOT / 'package-journals')
        try:
            cleanup_owned(work, self.scratch, self.cp.data['baseline']['campaign_id'])
            record['scratch_cleanup'] = 'VERIFIED_MARKER_BOUND_DIRECTORY_REMOVED'
        except OSError as exc:
            record['scratch_cleanup'] = dict(status='ERROR', error=str(exc))
        expected_startup_exit=(mode=='startup-exit-before-latch' and reason=='BOOTSTRAP_EXIT_BEFORE_ADMISSION'
            and receipt.get('exit_code')==73 and receipt.get('cleanup_verified') is True)
        if reason and not expected_startup_exit:
            self.cp.data['interruptions'].append(dict(attempt_id=attempt_id, group=group, reason=reason, time_utc=utc()))
        if not receipt['source_integrity'] or mode!='tests' and any(i['reason']=='EVIDENCE_INVALID' for i in issues):
            raise RuntimeError('EVIDENCE_INVALID: stop entire campaign')
        self.active = None
        self.cp.save()
        self.report('RUNNING', self)
        return dict(folder=str(folder), receipt=receipt, results=values, events=events, request=request)

    def run_startup_checks(self):
        """Always runs before pytest: validate early exit and normal latched startup."""
        evidence=[]
        for mode in ('startup-exit-before-latch','startup-probe'):
            outcome=self.attempt(mode,mode=mode,profile='infrastructure',timeout=60)
            if outcome is None:
                raise RuntimeError('BLOCKED_RESOURCE: owned startup verification not admitted')
            receipt=outcome['receipt']
            good=receipt.get('schema_version') in (2,3) and receipt.get('cleanup_verified') is True
            good=good and receipt.get('source_integrity') is True and receipt.get('cleanup',{}).get('active_processes')==0
            if mode=='startup-exit-before-latch':
                good=good and receipt.get('exit_code')==73 and receipt.get('payload_admitted') is False
                good=good and receipt.get('monitor_reason')=='BOOTSTRAP_EXIT_BEFORE_ADMISSION'
            else:
                good=good and receipt.get('exit_code')==0 and receipt.get('payload_admitted') is True
                good=good and receipt.get('monitor_reason') is None and any(
                    e['kind']=='startup_probe_pass' and e.get('random_bytes_available') is True
                    and e.get('hash_randomization') is True for e in outcome['events'])
            evidence.append(dict(mode=mode,status='PASS' if good else 'FAIL',attempt=outcome['folder']))
            new_json(ROOT/'startup-checks'/(outcome['request']['attempt_id']+'.json'),evidence[-1])
            if not good:
                raise RuntimeError('OWNED_STARTUP_VERIFICATION_FAILED: '+mode+'; inspect the recorded controller/stdout evidence')
        self.cp.data['completed_operations']['startup-checks']=dict(status='PASS',checks=evidence)
        self.cp.save()

    def run_infrastructure(self):
        inventory = self.cp.data.get('infrastructure_inventory', {})
        previous=self.cp.data.get('infrastructure_results',{})
        if any(v.get('status') in ('FAIL','ERROR','TIMEOUT') and not v.get('resume_incomplete') for v in previous.values()):
            return False  # Preserve a failed safety check; no assertion retry loop.
        pending = [n for n in inventory if previous.get(n, {}).get('status','NOT_RUN') == 'NOT_RUN'
                   or previous.get(n,{}).get('resume_incomplete') or previous.get(n,{}).get('requires_runtime_revalidation')]
        if not pending:
            return True
        outcome = self.attempt('campaign-infrastructure', nodes=pending, profile='infrastructure',
            files=sorted({inventory[n]['path'] for n in pending}), timeout=300)
        return bool(outcome and all(self.cp.data.get('infrastructure_results', {}).get(n, {}).get('status')=='PASS'
            and not self.cp.data.get('infrastructure_results',{}).get(n,{}).get('requires_runtime_revalidation') for n in inventory))

    def run_evidence_canary(self):
        node='tests/test_experiment_artifacts.py::test_ingest_failures_block_completion_preserve_source[fsync]'
        if node not in self.cp.pending() or self.stop():return
        outcome=self.attempt('source-evidence-canary',nodes=[node],files=[self.cp.data['inventory'][node]['path']],timeout=180)
        if outcome is None:return
        value=self.cp.states()[node]
        if value['status']=='NOT_RUN':
            self.cp.data.setdefault('inventory_holds',{})[node]=dict(nodeid=node,
                reason_code='QUARANTINED_INCOMPLETE_EVIDENCE_CANARY',evidence=outcome['folder'])
            self.cp.save()

    def source_groups(self):
        grouped, active, cases, files = [], [], 0, set()
        for node in self.cp.pending():
            entry = self.cp.data['inventory'][node]
            if entry['profile'] == 'stdlib':
                continue
            path = entry['path']
            if active and (cases >= self.config['max_source_group_cases'] or
                    path not in files and len(files) >= self.config['max_source_group_files']):
                grouped.append(active)
                active, cases, files = [], 0, set()
            active.append(node)
            cases += 1
            files.add(path)
        if active:
            grouped.append(active)
        return grouped

    def run_source(self):
        for number, planned in enumerate(self.source_groups(), 1):
            interruptions = 0
            remaining = planned
            while remaining and not self.stop():
                outcome = self.attempt('source-' + str(number), nodes=remaining,
                    files=sorted({self.cp.data['inventory'][n]['path'] for n in remaining}))
                if outcome is None:
                    for node in remaining:
                        self.cp.data['results'][node] = dict(nodeid=node, status='BLOCKED', reason_code='BLOCKED_RESOURCE')
                    self.cp.save()
                    break
                reason = outcome['receipt']['monitor_reason']
                effective=self.cp.states()
                remaining = [n for n in remaining if effective[n]['status']=='NOT_RUN']
                if reason == 'USER_STOP':
                    return
                if reason in ('BLOCKED_RESOURCE', 'DISK_RESERVE') and remaining:
                    interruptions += 1
                    if interruptions > self.config['max_resource_interruptions_per_group']:
                        for node in remaining:
                            self.cp.data['results'][node] = dict(nodeid=node, status='BLOCKED', reason_code='BLOCKED_RESOURCE')
                        self.cp.save()
                        break
                    self.cp.data['retries'].append(dict(group='source-' + str(number),
                        reason=reason, only_incomplete_nodes=remaining, time_utc=utc()))
                    self.cp.save()
                    continue
                # Record only a started partial test as ERROR/TIMEOUT. Untouched
                # nodes remain NOT_RUN; independent files can still run this invocation.
                started={r['nodeid'] for r in outcome['events'] if r['kind']=='test_start'}
                interrupted_paths={n.split('::')[0] for n in started
                    if not outcome['results'].get(n,{}).get('phases_complete')}
                for node in remaining:
                    status = ('TIMEOUT' if reason in ('GROUP_TIMEOUT', 'NO_PROGRESS_TIMEOUT') else 'ERROR') if node in started else 'NOT_RUN'
                    self.cp.data['results'][node] = dict(nodeid=node, status=status,
                        reason_code=(reason or 'WORKER_CRASH') if node in started else 'GROUP_INTERRUPTION_NOT_STARTED', resume_incomplete=True,
                        evidence_file=str(Path(outcome['folder']) / 'controller.json'))
                self.cp.save()
                independent=[n for n in remaining if n not in started]
                if interrupted_paths and independent:
                    remaining=independent
                    continue  # Fresh owned worker; no crashed test is retried.
                break

    def run_forensic(self):
        allowed = self.config['forensic_cpu_case']
        pending = [n for n in self.cp.pending() if self.cp.data['inventory'][n]['profile'] == 'stdlib']
        current = self.cp.data['results'].get(allowed, {})
        if current.get('status') in ('TIMEOUT', 'FAIL', 'ERROR'):
            pending.append(allowed)
        pending = list(dict.fromkeys(pending))
        if not pending or self.stop():
            return
        # Only the previously timed-out CPU recipe receives one additional verification.
        if allowed in pending:
            if self.cp.data['forensic_verification_reserved']:
                pending.remove(allowed)
            elif self.admitted_resources():
                self.cp.data['forensic_verification_reserved'] = True
                self.cp.save()  # Reserve before process launch; never retry after a crash.
            else:
                pending.remove(allowed)
        if pending:
            outcome = self.attempt('forensic-stdlib', nodes=pending, profile='stdlib',
                files=sorted({self.cp.data['inventory'][n]['path'] for n in pending}),
                timeout=self.config['forensic_group_timeout_seconds'])
            if outcome is None:
                for node in pending:
                    if self.cp.data['results'].get(node, {}).get('status') == 'NOT_RUN':
                        self.cp.data['results'][node] = dict(nodeid=node, status='BLOCKED', reason_code='BLOCKED_RESOURCE')
                self.cp.save()
            else:
                # Preserve the fixture's fixed internal 60-second deadline classification.
                for path in Path(outcome['folder']).glob('harvested/**/worker-receipt.json'):
                    raw = read_json(path, {})
                    if raw.get('monitor_outcome') == 'timeout':
                        value = self.cp.data['results'].get(allowed, {})
                        if value.get('status') in ('FAIL', 'ERROR'):
                            value.update(status='TIMEOUT', reason_code='FORENSIC_INTERNAL_60S_DEADLINE', worker_receipt=str(path))
                self.cp.save()
                for node in pending:
                    current=self.cp.data['results'].get(node,{})
                    if current.get('status','NOT_RUN')=='NOT_RUN':
                        self.cp.data['results'][node]=dict(nodeid=node,status='ERROR',
                            reason_code=outcome['receipt']['monitor_reason'] or 'WORKER_CRASH',
                            resume_incomplete=node!=allowed,evidence_file=str(Path(outcome['folder'])/'controller.json'))
                self.cp.save()
