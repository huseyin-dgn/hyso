"""Offline inventory/migration. No pytest import, collection, model or device call."""
from __future__ import annotations

import ast
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from common import LEGACY, REPO, ROOT, atomic_json, digest, file_hash, git, read_json
from campaign_checkpoint import aggregate
from campaign_collection import canonical_nodeid,profile_for


def component(node):
    value = node.lower()
    if '/forensics/' in value:
        return 'Forensic'
    if 'cli' in value or 'contextual' in value or 'user_guide' in value or 'api_' in value:
        return 'Public API / CLI / Help'
    if any(k in value for k in ('advisor', 'recommend', 'comparison')):
        return 'Advisor / Model Comparison'
    if any(k in value for k in ('reporting', 'presentation')):
        return 'Reporting'
    if any(k in value for k in ('builder', 'graph', 'modular', 'm3_', 'm4_', 'attention', 'registry')):
        return 'Model Builder / Graph'
    if any(k in value for k in ('train', 'checkpoint', 'resume', 'scheduler', 'evaluation', 'gradient')):
        return 'Training / Evaluation'
    if 'packag' in value or 'distribution' in value:
        return 'Packaging'
    return 'Analyzer / Performance'


@lru_cache(maxsize=None)
def _functions(filename):
    path = REPO / filename
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    found = {}
    def walk(nodes, prefix=''):
        for item in nodes:
            if isinstance(item, ast.ClassDef):
                walk(item.body, prefix + item.name + '.')
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            assertions = [ast.unparse(n) for n in ast.walk(item) if isinstance(n, ast.Assert)]
            assertions += [ast.unparse(n) for n in ast.walk(item) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr.startswith('assert')]
            found[prefix + item.name] = (item.lineno, assertions[:5])
    walk(tree.body)
    return found


def function_info(node):
    parts = node.split('::')
    wanted = '.'.join(parts[1:]).split('[')[0]
    return _functions(parts[0]).get(wanted, (None, []))


def prepared_inventory():
    original = read_json(LEGACY / 'collected-tests.json')
    if not isinstance(original, list):
        raise ValueError('retained inventory is missing')
    changes=read_json(Path(__file__).with_name('retired-tests.json'),{})
    retired={r['nodeid']:r for r in changes.get('retirements',[])}
    risk = read_json(LEGACY / 'blocked-nodeids.json', {})
    result, seen, cache = [], set(), {}
    for row in original:
        node = canonical_nodeid(row['nodeid'])
        if node in seen:
            raise ValueError('duplicate node ID: ' + node)
        seen.add(node)
        if node in retired:
            if node in risk:raise ValueError('risk tests cannot be retired by the historical-test policy')
            continue
        path = node.split('::')[0]
        if path not in cache:
            cache[path] = file_hash(REPO / path)
        line, expected = function_info(node)
        entry = dict(nodeid=node, path=path, line=line, test_source_sha256=cache[path],
                     profile=profile_for(node),
                     component=component(node), expected_behavior=expected,
                     markers=row.get('markers', []))
        if node in risk:
            item = risk[node]
            entry.update(risk_class=item['risk_class'], reason_code='BLOCKED_ADMISSION',
                         risk_reason=item['reason'])
        result.append(entry)
    if set(risk) - seen:
        raise ValueError('risk exclusions are not in the retained inventory')
    if set(retired)-seen:raise ValueError('retirement must name an original frozen node')
    for row in changes.get('additions',[]):
        node=canonical_nodeid(row['nodeid']);path=node.split('::')[0]
        if node in seen or profile_for(node)!='cpu' or file_hash(REPO/path)!=row['source_sha256']:
            raise ValueError('unreviewed/duplicate replacement test')
        line,expected=function_info(node)
        if line is None:raise ValueError('reviewed replacement function is missing')
        result.append(dict(nodeid=node,path=path,line=line,test_source_sha256=row['source_sha256'],
            profile='cpu',component=component(node),expected_behavior=expected,markers=[],reviewed_addition=row['reason']))
        seen.add(node)
    return result


def executable_tree(payload):
    tree = ast.parse(payload)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def source_delta_review(config):
    base = config['legacy_baseline_commit']
    head = git('rev-parse', 'HEAD')
    changed = git('diff', '--name-only', base, head).splitlines()
    python_semantics, other = [], []
    for path in changed:
        if path.endswith('.py'):
            before = git('show', base + ':' + path)
            after = git('show', head + ':' + path)
            if executable_tree(before) != executable_tree(after):
                python_semantics.append(path)
        elif path != 'hyso/data/api-topics.json':
            other.append(path)
    return dict(baseline_commit=base, current_commit=head, changed_files=changed,
                executable_python_changes=python_semantics, other_changes=other,
                reviewed_contextual_help_only=not python_semantics and not other,
                note='Legacy reports do not embed a commit; applicability is an explicit reconciliation using recorded test hashes and this Git delta, not retrofitted capture metadata.')


def _legacy_rows(path):
    rows = []
    with path.open('rb') as stream:
        for number, raw in enumerate(stream, 1):
            try:
                row = json.loads(raw)
            except (ValueError, UnicodeError):
                if not raw.endswith(b'\n'):
                    break
                raise ValueError('invalid legacy report ' + str(path) + ':' + str(number))
            rows.append(row)
    return rows


def import_legacy(inventory, config):
    wanted = {r['nodeid']: r for r in inventory}
    delta = source_delta_review(config)
    previous = {r['path']: r['sha256'] for r in read_json(LEGACY / 'complete-static-inventory.json', [])}
    candidates, history, bindings = defaultdict(list), [], {}
    # Order only chooses between retained complete attempts. It is not capture-time attestation.
    paths = sorted(LEGACY.glob('*-reports.jsonl'), key=lambda p: (p.stat().st_mtime_ns, p.name))
    stopped = read_json(ROOT / 'preparation/stop-existing-execution.json', {})
    for path in paths:
        stem = path.name[:-len('-reports.jsonl')]
        rows = _legacy_rows(path)
        groups = defaultdict(list)
        for row in rows:
            if row.get('nodeid') and canonical_nodeid(row['nodeid']) in wanted:
                groups[canonical_nodeid(row['nodeid'])].append(row)
        controller = read_json(LEGACY / (stem + '-controller.json'), {})
        exit_info = read_json(LEGACY / (stem + '-exit.json'), {})
        cleanup = controller.get('cleanup', {}).get('active_processes') == 0
        if stem == 'cpu-serial-v3-01' and stopped.get('remaining_owned_processes') == 0:
            cleanup = True
        # In an interrupted session, already finished tests remain valid only with cleanup evidence.
        bindings[str(path)] = dict(sha256=file_hash(path), bytes=path.stat().st_size)
        for node, phases in groups.items():
            entry = wanted[node]
            mixed = entry['profile'] == 'stdlib' and not stem.startswith('forensic-stdlib')
            # Old writers did not all identify subtests; only the last call is the main call.
            calls = [p for p in phases if p['when'] == 'call']
            normalized = []
            for original in phases:
                p = dict(original)
                if 'is_subtest' not in p:
                    p['is_subtest'] = original['when'] == 'call' and original is not calls[-1] if calls else False
                p['traceback'] = p.get('longrepr') or ''
                normalized.append(p)
            value = aggregate(node, normalized)
            value.update(attempt_id='legacy:' + stem, group=stem, evidence_file=str(path),
                         evidence_sha256=bindings[str(path)]['sha256'], source_commit=None,
                         source_commit_binding='LEGACY_CAMPAIGN_BASELINE_WITH_REVIEWED_GIT_DELTA',
                         legacy=True, environment=read_json(LEGACY / (stem + '-runtime.json'), {}),
                         source_applicability_review=delta)
            invalid=[]
            if not cleanup:invalid.append('CLEANUP_UNVERIFIED')
            if mixed:invalid.append('MIXED_FORENSIC_PROFILE')
            if exit_info.get('cuda_initialized') is True:invalid.append('UNAUTHORIZED_CUDA_INITIALIZATION')
            valid = cleanup and not mixed and exit_info.get('cuda_initialized') is not True
            if entry['profile'] == 'stdlib':
                valid = valid and exit_info.get('torch_imported') is False
                if exit_info.get('torch_imported') is not False:invalid.append('STDLIB_PARENT_NOT_VERIFIED_TORCH_FREE')
            test_unchanged = previous.get(entry['path']) == entry['test_source_sha256']
            approved_negative = entry['path'] == 'tests/forensics/test_p23_forensic_runner.py' and stem.startswith('forensic-stdlib')
            if not test_unchanged and not approved_negative:
                valid = False
                invalid.append('TEST_SOURCE_CHANGED')
            if not delta['reviewed_contextual_help_only']:
                valid = False  # A semantic source change requires fresh related evidence.
                invalid.append('UNREVIEWED_EXECUTABLE_SOURCE_DELTA')
            doc_sensitive = any(k in node.lower() for k in ('contextual_help', 'docstring', 'topic', 'discovery'))
            if doc_sensitive and delta['changed_files'] and not stem.startswith('cpu-serial-help-fix'):
                valid = False
                invalid.append('CONTEXTUAL_HELP_EVIDENCE_PREDATES_SHIPPED_DOC_CHANGE')
            if mixed:
                value['invalid_profile_reason'] = 'Torch-free forensic controller mixed with root Torch conftest'
            if node == config['forensic_cpu_case'] and stem.startswith('forensic-stdlib') and value['status'] == 'FAIL':
                receipt_path=Path(config['legacy_forensic_cpu_receipt'])
                copy=ROOT/'legacy-evidence/forensic-cpu-original-worker-receipt.json'
                if not receipt_path.is_file():receipt_path=copy
                raw_receipt=read_json(receipt_path,{})
                if (receipt_path.is_file() and file_hash(receipt_path)==config['legacy_forensic_cpu_receipt_sha256']
                    and raw_receipt.get('monitor_outcome')=='timeout'
                    and raw_receipt.get('cleanup',{}).get('active_processes')==0):
                    if not copy.exists():
                        copy.parent.mkdir(parents=True,exist_ok=True)
                        import os
                        with copy.open('xb') as stream:
                            stream.write(receipt_path.read_bytes());stream.flush();os.fsync(stream.fileno())
                    if file_hash(copy)!=config['legacy_forensic_cpu_receipt_sha256']:
                        raise RuntimeError('EVIDENCE_INVALID: original forensic receipt copy hash changed')
                    value.update(status='TIMEOUT', reason_code='FORENSIC_PRE_RECIPE_DEADLINE',
                        worker_receipt=str(copy),worker_receipt_sha256=config['legacy_forensic_cpu_receipt_sha256'],
                        known_cause='Hash-verified retained worker receipt records timeout at 60s with cleanup=0; underlying delay is not established as a product defect.')
            value['applicable'] = valid
            value['inapplicable_reasons']=invalid
            history.append(value)
            if valid and value['status'] != 'NOT_RUN':
                candidates[node].append(value)
    adopted = {node: values[-1] for node, values in candidates.items()}
    # Closed profile evidence supersedes the mixed-profile errors; no overlap in terminal categories.
    counts = Counter(v['status'] for v in adopted.values())
    return adopted, dict(source_delta=delta, evidence_bindings=bindings, attempt_history=history,
                         adopted_counts=dict(counts), adopted_unique=len(adopted),
                         raw_unique=len({v['nodeid'] for v in history}),
                         conflicting_attempts=[node for node, values in candidates.items()
                             if len({v['status'] for v in values}) > 1])


def v3_coverage(inventory, adopted):
    manifest = read_json(LEGACY / 'cpu-serial-v3-batches.json', {})
    wanted = {r['nodeid']: r for r in inventory}
    assignments = defaultdict(list)
    for group in manifest.get('batches', []):
        name = group['name']
        files = set(read_json(LEGACY / (name + '-requested.json'), []))
        done = set(read_json(LEGACY / (name + '-completed.json'), []))
        for node in wanted:
            if node.split('::')[0] in files and node not in done:
                assignments[node].append(name)
    pending = [n for n, r in wanted.items() if not r.get('risk_class') and r['profile'] == 'cpu'
               and adopted.get(n, {}).get('status') not in ('PASS', 'FAIL', 'ERROR', 'SKIP')]
    return dict(v3_groups=len(manifest.get('batches', [])), pending_safe_cpu=len(pending),
                unassigned=[n for n in pending if not assignments[n]],
                overlapping=[n for n in pending if len(assignments[n]) > 1],
                assignments={n: assignments[n] for n in pending})
