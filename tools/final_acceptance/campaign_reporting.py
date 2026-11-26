"""Derived reports use separate source/package/feature denominators and no fake PASS."""
from __future__ import annotations

import csv
import html
import io
import json
import re
import shutil
import zipfile
from collections import Counter
from pathlib import Path

from common import LEGACY, ROOT, STATUSES, atomic_json, digest, file_hash, read_json, utc
from campaign_operations import expected_package_cases
from campaign_evidence import provisional_progress

REPORT_LISTS = dict(PASS='passed_tests.json', FAIL='failed_tests.json', ERROR='error_tests.json',
    SKIP='skipped_tests.json', BLOCKED='blocked_tests.json', TIMEOUT='timeout_tests.json', NOT_RUN='not_run_tests.json')


def status_of(rows):
    values = [r['status'] for r in rows]
    if not values:
        return 'NOT_RUN'
    if any(s in ('FAIL', 'ERROR') for s in values):
        return 'FAIL'
    if all(s == 'PASS' for s in values):
        return 'PASS'
    if all(s == 'BLOCKED' for s in values):
        return 'BLOCKED'
    return 'PARTIAL'


def explain_failure(state, entry):
    failure = state.get('failure', {})
    trace = failure.get('traceback', failure.get('longrepr', '')) or ''
    matches = re.findall(r'(\w*(?:Error|Exception)):\s*([^\n]*)', trace)
    exception, message = matches[-1] if matches else ('UNDETERMINED', state.get('known_cause', state.get('reason_code','UNKNOWN')))
    code = state.get('reason_code','UNKNOWN')
    environment = code in ('ENVIRONMENT_ERROR', 'CONTROLLER_INTERRUPTED', 'BLOCKED_RESOURCE')
    return dict(nodeid=state['nodeid'], test_file=entry['path'], line=entry.get('line'),
        phase=failure.get('when'), exception_type=exception, message=message, full_traceback=trace,
        expected_behavior=entry.get('expected_behavior', []), actual_behavior=message,
        environment=state.get('environment_id', state.get('environment')),
        source_commit=state.get('source_commit'), duration_seconds=state.get('duration_seconds'),
        component=entry['component'], related_source_files=state.get('related_source_files', [entry['path']]),
        resource_evidence=state.get('controller_evidence'), known_cause=state.get('known_cause'),
        probable_root_cause='Not established; inspect the recorded traceback and exact environment.',
        classification='ENVIRONMENT_OR_INFRASTRUCTURE' if environment else 'UNDETERMINED_PRODUCT_OR_TEST',
        product_defect_proven=False, priority='HIGH' if state['status'] in ('FAIL', 'ERROR') else 'MEDIUM',
        user_impact='This scope is not accepted until the recorded failure is resolved.',
        recommended_fix='Review this failure in its recorded environment; no assertions or source are changed automatically.',
        regression_after_fix=[entry['path']], evidence_file=state.get('evidence_file'))


def feature_matrix(checkpoint, package, commit):
    base = read_json(LEGACY / 'P01-P35-draft.json', [])
    prior = read_json(LEGACY.parent / 'full-library-acceptance-20261125/b4-p23-p31-20261125/P01-P35-matrix.json', [])
    # The retained path is a sibling of this campaign under outputs/.
    if not prior:
        prior = read_json(LEGACY.parents[0] / 'full-library-acceptance-20261125/b4-p23-p31-20261125/P01-P35-matrix.json', [])
    details = {r['id']: r for r in prior}
    states = checkpoint.states()
    rows = []
    for old in base:
        feature = old['id']
        original = details.get(feature, {})
        files = {Path(p).name for p in original.get('tests', [])}
        related = [v for n, v in states.items() if Path(checkpoint.data['inventory'][n]['path']).name in files]
        cpu = [v for v in related if not checkpoint.data['inventory'][v['nodeid']].get('risk_class')]
        counts = Counter(v['status'] for v in related)
        failed = [v for v in related if v['status'] in ('FAIL', 'ERROR', 'TIMEOUT')]
        decision = old['status']
        if failed and decision == 'PASS':
            decision = 'PARTIAL'  # A raw test failure is not automatically a proven feature defect.
        if not checkpoint.data.get('historical_bindings_valid',False):
            decision = 'PARTIAL'
        row = dict(analysis_number=feature, analysis_name=old['name'], purpose=old['name'],
            supported_scope=old['acceptance_scope'], historical_acceptance=old['status'],
            CPU_validation=dict(current_tests=status_of(cpu), historical_scope='PRESERVED_RECORDED_SCOPE'),
            CUDA_validation=dict(current='BLOCKED', historical='PRESERVED_ONLY_WHERE_RECORDED',
                reason='No fresh CUDA admission; mock/CPU cases do not verify real GPU execution.'),
            API_validation=status_of([v for v in related if any(k in v['nodeid'] for k in ('api', 'native', 'analysis', 'advisor'))]),
            CLI_validation=status_of([v for v in related if 'cli' in v['nodeid']]),
            package_validation=package.get('status', 'NOT_RUN'), related_test_count=len(related),
            passed_test_count=counts['PASS'], failed_test_count=counts['FAIL']+counts['ERROR']+counts['TIMEOUT'],
            source_commit=commit, historical_evidence_commit='Retained original artifacts; not relabeled as current-source measurements',
            evidence_files=[v.get('evidence_file') for v in related if v.get('evidence_file')],
            historical_evidence_bindings=read_json(LEGACY / 'historical-evidence-bindings.json', {}),
            evidence_hashes={v['evidence_file']:v.get('evidence_sha256',v.get('phase_evidence_sha256'))
                for v in related if v.get('evidence_file')},
            missing_conditions=old['missing_conditions'], known_limitations=[],
            open_errors=[dict(nodeid=v['nodeid'], status=v['status'], reason=v['reason_code']) for v in failed],
            decision=decision)
        if feature in ('P01', 'P02', 'P11'):
            row['known_limitations'] += ['Historical trained A/B comparison capture remains INSUFFICIENT_EVIDENCE.',
                'New schema-3 comparisons cover the observed resident CPU input batch; dataset origin and upstream preprocessing remain explicitly missing.']
        if feature == 'P23':
            row['subscopes'] = old.get('subscopes', [])
            row['CUDA_validation'].update(current='BLOCKED_B4_UNRESOLVED')
            row['known_limitations'].append('CPU/basic training memory is supported; CUDA saved-activation is actively disabled, with unavailable metrics rather than zero/PASS.')
        if feature == 'P31':
            row['CPU_validation']['historical_7200_seconds'] = old.get('subscope_acceptance')
            row['CPU_validation']['historical_integrity']=read_json(ROOT/'preparation/P31-historical-integrity.json',{})
            row['known_limitations'].append('Two original uninterrupted CPU records are historical evidence; no new 7200-second run is authorized in this campaign.')
        rows.append(row)
    return rows


def render(checkpoint, progress, package=None, *, final=False, bundle=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    progress.update(provisional_progress(checkpoint,progress.get('active_attempt_id')))
    package = package or dict(status='NOT_RUN', current_commit=checkpoint.data['baseline']['commit'],
        planned_cases=36,test_counts=dict(NOT_RUN=36),
        cases=[dict(nodeid=n,status='NOT_RUN',reason_code='NOT_STARTED') for n in expected_package_cases()],
        prior_clean_installs=read_json(LEGACY / 'installed-packages.json', []),
        note='The c389ad1 wheel/sdist evidence is preserved; changed shipped files require fresh current-HEAD packages.')
    if package.get('source_commit',package.get('current_commit')) != checkpoint.data['baseline']['commit']:
        package=dict(status='NOT_RUN',source_commit=checkpoint.data['baseline']['commit'],
            prior_version_evidence=package,planned_cases=36,test_counts=dict(NOT_RUN=36),
            cases=[dict(nodeid=n,status='NOT_RUN',reason_code='CURRENT_SOURCE_PACKAGE_NOT_VERIFIED') for n in expected_package_cases()])
    retired=list(checkpoint.data.get('retired_tests',{}).values())
    atomic_json(ROOT/'retired_tests.json',retired)
    progress['retired_source_tests']=len(retired)
    states = checkpoint.states()
    counts = checkpoint.counts()
    assert sum(counts.values()) == len(checkpoint.data['inventory'])
    for status, name in REPORT_LISTS.items():
        atomic_json(ROOT / name, [v for v in states.values() if v['status'] == status])
    atomic_json(ROOT / 'test_inventory.json', list(checkpoint.data['inventory'].values()))
    failures = [explain_failure(v, checkpoint.data['inventory'][n]) for n, v in states.items()
                if v['status'] in ('FAIL', 'ERROR', 'TIMEOUT')]
    atomic_json(ROOT / 'failure_analysis.json', failures)
    failure_groups={}
    for failure in failures:
        key=digest(dict(exception=failure['exception_type'], message=failure['message'],
            environment=failure['environment'],classification=failure['classification']))[:16]
        group=failure_groups.setdefault(key,dict(group_id=key,classification=failure['classification'],
            known_cause=failure['known_cause'],root_cause_proven=False,nodeids=[],
            proposed_action=failure['recommended_fix']))
        group['nodeids'].append(failure['nodeid'])
    atomic_json(ROOT/'failure_groups.json',list(failure_groups.values()))
    safe = [v for n, v in states.items() if not checkpoint.data['inventory'][n].get('risk_class')]
    source_acceptance = status_of(safe)
    cpustatus = status_of([v for n,v in states.items() if checkpoint.data['inventory'][n]['profile']=='cpu'
                          and not checkpoint.data['inventory'][n].get('risk_class')])
    api = [v for n, v in states.items() if any(k in n for k in ('test_api_', 'test_public_', 'test_contextual_help', 'test_project_wide_pythonic_api', 'test_python_first'))]
    cli = [v for n, v in states.items() if 'test_cli' in n]
    forensic = [v for n, v in states.items() if checkpoint.data['inventory'][n]['profile'] == 'stdlib']
    matrix = feature_matrix(checkpoint, package, checkpoint.data['baseline']['commit'])
    feature_counts = Counter(v['decision'] for v in matrix)
    overall = 'FAIL' if source_acceptance == 'FAIL' or package.get('status') == 'FAIL' else 'PARTIAL'
    readiness = 'CONDITIONAL' if cpustatus == 'PASS' and package.get('status') == 'PASS' else 'NOT_READY'
    decisions = dict(SOURCE_ACCEPTANCE=source_acceptance, CPU_ACCEPTANCE=cpustatus,
        CUDA_ACCEPTANCE='BLOCKED', PACKAGE_ACCEPTANCE=package.get('status', 'NOT_RUN'),
        API_ACCEPTANCE=status_of(api), CLI_ACCEPTANCE=status_of(cli), FORENSIC_ACCEPTANCE=status_of(forensic),
        P01_P35_ACCEPTANCE='PARTIAL', OVERALL_RELEASE_ACCEPTANCE=overall)
    summary = dict(schema_version=1, generated_utc=utc(), campaign_status=progress['status'],
        source_commit=checkpoint.data['baseline']['commit'], unique_source_inventory=len(states),
        source_test_counts=counts, retired_source_tests=retired, package_tests=package.get('test_counts', {}),
        infrastructure_tests=checkpoint.data.get('infrastructure_results', {}),
        P01_P35_counts=dict(feature_counts), full_pass=False, readiness=readiness,
        decisions=decisions, no_automatic_source_changes=True, no_real_CUDA=True,
        no_failfast_or_allocation_stress=True, no_new_P31_7200_run=True,
        denominator_note='Source node IDs, package cases and 35 acceptance features have separate denominators.')
    quarantined=list(checkpoint.data.get('quarantined_inventory',{}).values())
    summary['quarantine']=dict(diagnostic_records=len(quarantined),
        additional_unknown_unique=len({r['nodeid'] for r in quarantined
            if r['nodeid'] not in states and r['nodeid'] not in checkpoint.data.get('infrastructure_inventory',{})}),
        note='Quarantine diagnostics can refer to known held nodes; they are not added to the source denominator.')
    summary['recovered_outcomes']=checkpoint.data.get('recovered_outcomes',{})
    summary['evidence_quarantine_count']=len(checkpoint.data.get('evidence_quarantine',{}))
    atomic_json(ROOT/'evidence_quarantine.json',checkpoint.data.get('evidence_quarantine',{}))
    atomic_json(ROOT/'attempt_acceptance.json',checkpoint.data.get('attempt_acceptance',{}))
    atomic_json(ROOT/'infrastructure_failure_analysis.json',[
        dict(attempt_id=k,classification='EVIDENCE_OBSERVER_OR_INFRASTRUCTURE',product_defect_proven=False,
            evidence=v['evidence'],issues=v['issues']) for k,v in checkpoint.data.get('evidence_quarantine',{}).items()])
    atomic_json(ROOT/'quarantined_tests.json',quarantined)
    atomic_json(ROOT / 'summary.json', summary)
    terminal=[v for v in states.values() if v['status'] not in ('NOT_RUN','BLOCKED')]
    successes=[v for v in terminal if v['status']=='PASS']
    problems=[v for v in terminal if v['status'] in ('FAIL','ERROR','TIMEOUT')]
    successes.sort(key=lambda v:v.get('finished_utc') or '')
    problems.sort(key=lambda v:v.get('finished_utc') or '')
    progress.update(last_success=(successes[-1]['nodeid'] if successes else None),
        last_test_error=(dict(nodeid=problems[-1]['nodeid'],reason=problems[-1].get('reason_code')) if problems else None),
        infrastructure_total=len(checkpoint.data.get('infrastructure_inventory',{})),
        infrastructure_revalidation_pending=sum(
            checkpoint.data.get('infrastructure_results',{}).get(n,{}).get('status','NOT_RUN')=='NOT_RUN'
            or bool(checkpoint.data.get('infrastructure_results',{}).get(n,{}).get('requires_runtime_revalidation'))
            for n in checkpoint.data.get('infrastructure_inventory',{})),
        infrastructure_counts=dict(Counter(v['status'] for v in checkpoint.data.get('infrastructure_results',{}).values())))
    atomic_json(ROOT / 'package_acceptance.json', package)
    atomic_json(ROOT / 'public_api_acceptance.json', dict(status=status_of(api), tests=api))
    atomic_json(ROOT / 'cli_acceptance.json', dict(status=status_of(cli), tests=cli))
    atomic_json(ROOT / 'forensic_acceptance.json', dict(status=status_of(forensic), tests=forensic,
        legacy_mixed_profile_errors='Kept in reconciliation history; never counted twice or used to invalidate valid stdlib results.',
        additional_cpu_verification_limit=1, internal_native_deadline_seconds=60))
    atomic_json(ROOT / 'P01-P35-acceptance.json', matrix)
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=['analysis_number', 'analysis_name', 'historical_acceptance', 'decision',
        'related_test_count', 'passed_test_count', 'failed_test_count', 'source_commit'])
    writer.writeheader()
    writer.writerows({key: row[key] for key in writer.fieldnames} for row in matrix)
    (ROOT / 'P01-P35-acceptance.csv').write_text(out.getvalue(), encoding='utf-8-sig')
    atomic_json(ROOT / 'progress.json', dict(progress, counts=counts, total_unique_tests=len(states)))
    atomic_json(ROOT / 'interruptions.json', checkpoint.data['interruptions'])
    atomic_json(ROOT / 'retry_history.json', checkpoint.data['retries'])
    atomic_json(ROOT / 'test_reconciliation.json', dict(unique_inventory=len(states), counts=counts,
        sum_matches_inventory=sum(counts.values()) == len(states), source_applicability=checkpoint.data.get('legacy_review', {}).get('source_delta'),
        legacy_history=checkpoint.data.get('legacy_review', {}).get('attempt_history', []),
        attempts=checkpoint.data['attempts'], no_call_only_pass=True,
        selection_completed_files_are_not_pass_evidence=True,
        nodeid_schema_version=checkpoint.data.get('nodeid_schema_version'),
        nodeid_migrations=checkpoint.data.get('nodeid_migrations',[]),
        legacy_relink_history=checkpoint.data.get('legacy_relink_history',[]),
        collection_profiles=checkpoint.data.get('collection_reconciliation',{}),
        collection_holds=checkpoint.data.get('inventory_holds',{}),
        attempt_acceptance=checkpoint.data.get('attempt_acceptance',{}),
        evidence_quarantine=checkpoint.data.get('evidence_quarantine',{}),
        recovered_outcomes=checkpoint.data.get('recovered_outcomes',{}),
        legacy_adoption=checkpoint.data.get('legacy_review',{}).get('adopted_counts'),
        supplied_2442_pass_is_not_assumed=True,
        difference_reason='Only hash-bound complete phases with cleanup are adopted. Opaque parameter IDs are preserved; corrected identities can reconnect existing valid raw outcomes. Old doc-sensitive scopes require applicable evidence.'))
    tr = [f'# HYSo nihai kabul — {progress["status"]}', '', f'Kaynak commit: `{summary["source_commit"]}`',
        f'Karar: **{readiness}**. Genel kabul: **{overall}**. FULL_PASS verilmedi.', '',
        'Kaynak testleri: ' + ', '.join(f'{key}={value}' for key, value in counts.items()),
        f'P01–P35: {dict(feature_counts)}. Paket kabulü: {package.get("status", "NOT_RUN")}.', '',
        'CPU, gerçek CUDA, paket testleri ve tarihsel performans kanıtları ayrı kapsamlarla değerlendirilir.',
        'B4 kök nedeni çözümlenmedi; CUDA saved-activation BLOCKED_B4_UNRESOLVED olarak kapalıdır.',
        'P31’in iki tarihsel CPU 7200 saniye kaydı korunur; yeni uzun koşu yapılmaz.',
        'Eski A/B comparison_capture eksikliği sonradan doldurulmaz. Yapısal farklar performans üstünlüğü değildir.',
        'Analyzer/Advisor kullanıcı modelini değiştirmez; Builder/Trainer testleri yalnız açıkça seçilen owned test örneklerinde çalışır.', '',
        '## Bileşenler', '']
    for name in sorted({r['component'] for r in checkpoint.data['inventory'].values()}):
        rows = [states[n] for n, r in checkpoint.data['inventory'].items() if r['component'] == name]
        tr.append(f'- {name}: {status_of(rows)}; {dict(Counter(r["status"] for r in rows))}')
    tr += ['', 'RETIRED: '+str(len(retired))+' historical tests; original FAIL evidence retained in retired_tests.json, excluded from active PASS counts.']
    tr += ['', '## Hatalar ve eksikler', '']
    tr += [f'- `{v["nodeid"]}`: {v["exception_type"]}; {v["message"]}. Kesin kök neden kanıtlanmadıysa ürün hatası sayılmaz.' for v in failures]
    tr += ['', 'Tam traceback, ortam ve düzeltme önerileri `failure_analysis.json` dosyasındadır.',
        'Kanıtsız veya yarım kalan vaka PASS değildir. Kaynak/ortam/altyapı hataları ve güvenlik engelleri ayrı tutulur.',
        'Önerilen sonraki adım: kayıtlı gerçek hataları düzeltip yalnız etkilenen regresyonları hazırlamak; riskli aşamalar için ayrı doğrulanmış admission sağlamak.']
    tr += ['', '## Ayrı kabul kararları','']+[f'- {key}: {value}' for key,value in decisions.items()]
    tr += ['', '## P01–P35', '', '| Madde | Tarihsel kapsam | Güncel karar | CPU testleri | CUDA |',
        '|---|---|---|---|---|']
    tr += [f'| {r["analysis_number"]} | {r["historical_acceptance"]} | {r["decision"]} | {r["CPU_validation"]["current_tests"]} | {r["CUDA_validation"]["current"]} |' for r in matrix]
    protection=read_json(ROOT/'user_file_preservation.json',{})
    tr += ['',f'Kullanıcı dosyaları koruma kontrolü: {protection.get("user_files_unchanged","NOT_VERIFIED")}.',
        f'Önceki gerçek olaylardan kurtarılan sonuçlar: {checkpoint.data.get("recovered_outcomes",{})}. Kanıt karantinası: {summary["evidence_quarantine_count"]}.',
        f'Karantina tanı kaydı: {len(quarantined)}. Bilinmeyen veya tekrarlanan node’lar yürütülmez; bağımsız güvenli gruplar devam eder.',
        'Worker cleanup: her girişimin controller/recovery makbuzuyla değerlendirilir; doğrulanmamış cleanup PASS üretemez.',
        f'Kaynaklar: {len(states)}; paket vakaları: {package.get("planned_cases",36)}; kampanya altyapısı: {len(checkpoint.data.get("infrastructure_inventory",{}))}; özellikler: 35 ayrı payda.',
        'İlk hazırlık raporu yeni test sonucu değildir. PREPARED durumunda yalnız eski kanıtların uzlaştırması gösterilir.']
    (ROOT / 'FINAL_REPORT_TR.md').write_text('\n'.join(tr)+'\n', encoding='utf-8')
    english = f'''# HYSo final acceptance — {progress['status']}

Source: `{summary['source_commit']}`. Readiness: **{readiness}**. Overall: **{overall}**.
No FULL_PASS is claimed. Source counts: {counts}. Feature counts: {dict(feature_counts)}.
Package acceptance: {package.get('status', 'NOT_RUN')}.

CPU/offline tests do not certify real CUDA behavior. B4 remains unresolved and CUDA
saved-activation stays disabled. Historical two-model 7200-second CPU records are
preserved; no new P31 long run is started. Existing A/B captures are never upgraded.
Analyzer/Advisor remain read-only; Builder/Trainer run only explicit owned fixtures.

All per-node phases, subtests, tracebacks, resource interruptions, source and
environment identities, package results and safety receipts are retained in JSON.
Unproven causes are labeled undetermined. No automatic source fixes occur.
See failure_analysis.json, P01-P35-acceptance.json and test_reconciliation.json.
'''
    english += '\nRETIRED: '+str(len(retired))+' historical test(s); original FAIL evidence retained in retired_tests.json and excluded from active PASS counts.\n'
    english += '\n## Separate acceptance decisions\n\n'+'\n'.join(f'- {k}: {v}' for k,v in decisions.items())+'\n'
    english+=f'\nRecovered real outcomes: {checkpoint.data.get("recovered_outcomes",{})}. Evidence quarantine: {summary["evidence_quarantine_count"]}. Provisional results are separate from final PASS.\n'
    english += '\n## Components\n\n'
    for name in sorted({r['component'] for r in checkpoint.data['inventory'].values()}):
        rows=[states[n] for n,r in checkpoint.data['inventory'].items() if r['component']==name]
        english+=f'- {name}: {status_of(rows)}; {dict(Counter(v["status"] for v in rows))}\n'
    english+='\n## P01–P35\n\n| Feature | Historical | Decision | CPU tests | CUDA |\n|---|---|---|---|---|\n'
    english+='\n'.join(f'| {r["analysis_number"]} | {r["historical_acceptance"]} | {r["decision"]} | {r["CPU_validation"]["current_tests"]} | {r["CUDA_validation"]["current"]} |' for r in matrix)+'\n'
    english+='\n## Failures\n\n'+'\n'.join(f'- `{r["nodeid"]}`: {r["exception_type"]}: {r["message"]}. Cause unproven unless explicitly evidenced.' for r in failures)+'\n'
    (ROOT / 'FINAL_REPORT_EN.md').write_text(english, encoding='utf-8')
    (ROOT / 'EXECUTIVE_SUMMARY_TR.md').write_text('\n'.join(tr[:15])+'\n', encoding='utf-8')
    body = '<!doctype html><html lang="tr"><meta charset="utf-8"><title>HYSo kabul</title><style>body{font:16px system-ui;max-width:1100px;margin:30px auto;padding:20px}pre{white-space:pre-wrap}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}</style>'
    body += '<h1>HYSo — '+html.escape(progress['status'])+'</h1><p>'+html.escape(readiness)+'</p><pre>'+html.escape('\n'.join(tr))+'</pre>'
    body += '<p><a href="summary.json">JSON</a> · <a href="failure_analysis.json">Tracebacks</a> · <a href="P01-P35-acceptance.csv">P01–P35 CSV</a></p></html>'
    (ROOT / 'FINAL_REPORT.html').write_text(body, encoding='utf-8')
    live = body.replace('<title>HYSo kabul</title>', '<title>HYSo canlı durum</title><meta http-equiv="refresh" content="5">')
    live = live.replace('<h1>', '<pre>'+html.escape(json.dumps(progress, ensure_ascii=False, indent=2))+'</pre><h1>', 1)
    (ROOT / 'progress.html').write_text(live, encoding='utf-8')
    if bundle:
        make_bundle(checkpoint)
    return summary


def make_bundle(checkpoint):
    forbidden = {'.pt', '.pth', '.pyd', '.dll', '.whl', '.gz', '.exe', '.pyc', '.bak', '.patch', '.key', '.pfx'}
    files = []
    for path in ROOT.rglob('*'):
        if not path.is_file() or path.is_symlink() or path.suffix.lower() in forbidden:
            continue
        relative = path.relative_to(ROOT)
        if any(part in ('bundles', 'work', '__pycache__', 'venv', '.git', 'partial-artifacts') for part in relative.parts):
            continue
        if path.name in ('request.json', 'admit.json', 'interpreter-ready.json', 'manager.lock', 'COMPLETE_EVIDENCE.zip', 'evidence_manifest.json', 'COMPLETE_EVIDENCE.sha256.json'):
            continue
        if path.name.endswith('.tmp') or path.suffix.lower() == '.zip':
            continue
        # Only reports, hashes and bounded raw logs; not arbitrary source/user files.
        if path.suffix.lower() not in ('.json', '.jsonl', '.xml', '.log', '.txt', '.md', '.html', '.csv'):
            continue
        files.append(path)
    files.sort()
    manifest = {p.relative_to(ROOT).as_posix(): dict(sha256=file_hash(p), bytes=p.stat().st_size) for p in files}
    atomic_json(ROOT / 'evidence_manifest.json', dict(schema_version=1, contents=manifest,
        excluded='model weights, user files, environments, caches, binaries, raw admission tokens, passwords and credentials'))
    limit = checkpoint.data['baseline']['config']['max_bundle_bytes']
    if sum(p.stat().st_size for p in files) > limit:
        raise RuntimeError('bundle input budget exceeded; reports remain available separately')
    bundles = ROOT / 'bundles'
    bundles.mkdir(exist_ok=True)
    path = bundles / (utc().replace(':', '-').replace('+', '_') + '.zip')
    with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for file in [*files, ROOT / 'evidence_manifest.json']:
            archive.write(file, file.relative_to(ROOT).as_posix())
    shutil.copyfile(path, ROOT / 'COMPLETE_EVIDENCE.zip.tmp')
    import os
    os.replace(ROOT / 'COMPLETE_EVIDENCE.zip.tmp', ROOT / 'COMPLETE_EVIDENCE.zip')
    atomic_json(ROOT / 'COMPLETE_EVIDENCE.sha256.json', dict(sha256=file_hash(path), bytes=path.stat().st_size, immutable_bundle=str(path)))
