"""Preserve 658 CPU/34 GPU outcomes for the final two acceptance-only fixes."""
import ast
from pathlib import Path
from tools.final_acceptance.common import read_json,atomic_json,file_hash,digest

TARGETS=('P25','P35')
EXISTING=('tools/cuda_acceptance/feature_evidence.py','tools/cuda_acceptance/native_identity.py','tools/cuda_acceptance/six_group_review.py')
ADDED=('tools/cuda_acceptance/p25_evidence.py','tools/cuda_acceptance/p25_p35_review.py','tools/cuda_acceptance/tests/test_p25_p35.py')


class _Unaffected(ast.NodeTransformer):
    def visit_If(self,node):
        if ast.dump(node.test,include_attributes=False)==ast.dump(ast.parse("feature=='P25'").body[0].value,include_attributes=False):return None
        if any(isinstance(n,ast.Constant) and n.value=='p25-p35-remediation/before-plan.json' for n in ast.walk(node.test)):return None
        return self.generic_visit(node)


def equivalent(before,after):
    def shape(text):return ast.dump(_Unaffected().visit(ast.parse(text)),include_attributes=False)
    return shape(before)==shape(after)


def review_transition(old,plan,cp,root):
    root=Path(root);repo=root.parents[1];folder=root/'p25-p35-remediation'
    before=read_json(folder/'before-plan.json');checkpoint=read_json(folder/'before-checkpoint.json')
    def need(value,reason):
        if not value:raise RuntimeError('P25_P35_REVIEW_REFUSED: '+reason)
    changed={n for n in set(before['source_sha256'])|set(plan['source_sha256']) if before['source_sha256'].get(n)!=plan['source_sha256'].get(n)}
    need(changed<=set(EXISTING)|set(ADDED),'changes exceed P25/P35 fixes')
    need(before['cpu_inventory']==plan['cpu_inventory'] and before['groups']==plan['groups'] and before['commit']==plan['commit'],'scope/commit changed')
    need(before['runtime']==plan['runtime'] and before['budgets']==plan['budgets'] and before['feature_tools']['nsys']==plan['feature_tools']['nsys'],'runtime/resource/Nsight change')
    need(cp['cpu_results']==checkpoint['cpu_results'] and len(cp['cpu_results'])==658,'CPU PASS records changed')
    passed={n:r for n,r in checkpoint['results'].items() if r['status']=='PASS'}
    need(sum(n!='CPU_REGRESSIONS' for n in passed)==34 and all(cp['results'].get(n)==r for n,r in passed.items()),'GPU PASS records changed')
    need(not any(r['status']=='RUNNING' for r in cp['attempts']),'unreconciled execution')
    for name in EXISTING:
        saved=folder/'before-source'/name
        need(file_hash(saved)==before['source_sha256'][name],'baseline binding: '+name)
        if not name.endswith('native_identity.py'):
            need(equivalent(saved.read_text(encoding='utf-8'),(repo/name).read_text(encoding='utf-8')),'non-target execution path changed: '+name)
    original=before['feature_tools']['ort'];updated=plan['feature_tools']['ort']
    need(original['site']==updated['site'] and original['version']==updated['version'],'ORT environment changed')
    need(all(updated['allowed_binaries'].get(p)==h for p,h in original['allowed_binaries'].items()),'existing DLL pin changed')
    dependency=updated['driver_dependency']['files']['backend']
    need(set(updated['allowed_binaries'])-set(original['allowed_binaries'])=={dependency['path']} and
        updated['allowed_binaries'][dependency['path']]==dependency['sha256'] and dependency['signature']=='Valid',
        'only the independently verified installed driver backend may be added')
    need(before['preserved_evidence']==plan['preserved_evidence'],'historical campaign changed')
    manifest=read_json(folder/'preserved-attempt-files.json')
    for name,sha in manifest.items():need(file_hash(root/name)==sha,'original attempt file changed: '+name)
    value=dict(before_plan_sha256=digest(before),after_source_fingerprint=plan['source_fingerprint'],changed_files=sorted(changed),
        retained_CPU_PASS=658,retained_GPU_PASS=34,affected_previous_PASS=[],revalidation_features=list(TARGETS),
        original_attempt_files=len(manifest),rationale='P25-only ledger gate; ORT identity used only by failed P35; no product or accepted feature path changes; one verified registry-derived dependency pin.')
    path=folder/'source-impact-review.json';atomic_json(path,value)
    return dict(evidence=str(path),sha256=file_hash(path),revalidation_features=list(TARGETS),retained_CPU_PASS=658,retained_GPU_PASS=34)
