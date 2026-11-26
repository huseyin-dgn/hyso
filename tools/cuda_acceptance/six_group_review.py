"""Bounded source-impact proof for the fourteen requested failed feature cases."""
import ast
import copy
from pathlib import Path
from tools.final_acceptance.common import read_json,atomic_json,file_hash,digest

TARGETS=('P03','P04','P14','P17','P19','P20','P21','P22','P23','P24','P25','P30','P35','Trainer')
EXISTING=('tools/cuda_acceptance/runner.py','tools/cuda_acceptance/cases.py','tools/cuda_acceptance/workloads.py',
    'tools/cuda_acceptance/feature_evidence.py','tools/cuda_acceptance/validation.py','hyso/performance/streaming.py')
ADDED=('tools/cuda_acceptance/p14_probe.py','tools/cuda_acceptance/native_identity.py',
    'tools/cuda_acceptance/six_group_review.py','tools/cuda_acceptance/tests/test_six_groups.py')


def _tree(text):return ast.parse(text)
def _dump(node):return ast.dump(node,include_attributes=False)


class _Unaffected(ast.NodeTransformer):
    def __init__(self,kind):self.kind=kind
    def visit_FunctionDef(self,node):
        drop={'cases':{'pipeline','operators','capacity','video','providers','training'},
            'workloads':{'training','onnx_model'},'runner':{'prepare'},'streaming':{'_physical_uuid'}}
        if node.name in drop.get(self.kind,set()):return None
        return self.generic_visit(node)
    def visit_Expr(self,node):
        call=node.value
        if self.kind=='cases' and isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='run':
            if call.args and isinstance(call.args[0],ast.Constant) and call.args[0].value in TARGETS:return None
        return self.generic_visit(node)
    def visit_If(self,node):
        if self.kind in ('feature_evidence','validation'):
            names={n.id for n in ast.walk(node.test) if isinstance(n,ast.Name)}
            strings={n.value for n in ast.walk(node.test) if isinstance(n,ast.Constant) and isinstance(n.value,str)}
            if 'feature' in names and strings and strings<=set(TARGETS):
                result=[]
                for child in node.orelse:
                    transformed=self.visit(child)
                    if isinstance(transformed,list):result.extend(transformed)
                    elif transformed is not None:result.append(transformed)
                return result
        return self.generic_visit(node)
    def visit_Call(self,node):
        node=self.generic_visit(node)
        if self.kind=='streaming' and isinstance(node.func,ast.Name) and node.func.id=='_physical_uuid':node.func.id='str'
        return node


def unaffected_equivalent(before,after,kind):
    return _dump(_Unaffected(kind).visit(_tree(before)))==_dump(_Unaffected(kind).visit(_tree(after)))


def review_transition(old,plan,cp,root):
    if (Path(root)/'p25-p35-remediation/before-plan.json').is_file():
        from .p25_p35_review import review_transition as final_review
        return final_review(old,plan,cp,root)
    root=Path(root);repo=root.parents[1];folder=root/'six-group-remediation'
    if not (folder/'before-plan.json').is_file():return None
    original=read_json(folder/'before-plan.json');previous=read_json(folder/'before-checkpoint.json')
    def require(condition,reason):
        if not condition:raise RuntimeError('SIX_GROUP_REVIEW_REFUSED: '+reason)
    before=original['source_sha256'];after=plan['source_sha256']
    changed={n for n in set(before)|set(after) if before.get(n)!=after.get(n)}
    # This review cannot authorize unrelated later source transitions.
    if not changed<=set(EXISTING)|set(ADDED):return None
    require(original['cpu_inventory']==plan['cpu_inventory'] and original['groups']==plan['groups'],'inventory/group scope')
    require(original['commit']==plan['commit'],'source commit changed')
    require(original['runtime']==plan['runtime'] and original['budgets']==plan['budgets'],'runtime/budget change')
    require(original['preserved_evidence']==plan['preserved_evidence'],'historical campaign hashes')
    require(previous['cpu_results']==cp.get('cpu_results'),'existing CPU PASS records changed')
    preserved={n:r for n,r in previous['results'].items() if r['status']=='PASS'}
    require(all(cp['results'].get(n)==row for n,row in preserved.items()),'existing PASS record changed')
    require(not any(a['status']=='RUNNING' for a in cp['attempts']),'unreconciled attempt')
    proofs={}
    for name in EXISTING:
        saved=folder/'before-source'/name
        require(file_hash(saved)==before[name],'unbound baseline '+name)
        kind=Path(name).stem
        left=saved.read_text(encoding='utf-8');right=(repo/name).read_text(encoding='utf-8')
        require(unaffected_equivalent(left,right,kind),'change outside target functions: '+name)
        proofs[name]=dict(before_sha256=before[name],after_sha256=after[name],unaffected_AST_equal=True)
    # The shared factory changes only a same-device model.to(device), before the
    # optimizer. For device=cpu this preserves the existing ordinary FP32 model.
    left=next(n for n in _tree((folder/'before-source/tools/cuda_acceptance/workloads.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='training')
    right=copy.deepcopy(next(n for n in _tree((repo/'tools/cuda_acceptance/workloads.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='training'))
    move=_dump(ast.parse('m=m.to(device)').body[0]);moves=[n for n in right.body if _dump(n)==move]
    require(len(moves)==1,'exact pre-optimizer placement change')
    right.body=[n for n in right.body if _dump(n)!=move]
    require(_dump(left)==_dump(right),'CPU training factory behavior changed')
    for name,sha in read_json(folder/'preserved-attempt-files.json').items():
        require(file_hash(root/name)==sha,'original attempt evidence changed: '+name)
    require(len(cp['cpu_results'])==658 and sum(r['status']=='PASS' for n,r in previous['results'].items() if n!='CPU_REGRESSIONS')==22,
        'unexpected preserved denominators')
    value=dict(before_plan_sha256=digest(original),after_source_fingerprint=plan['source_fingerprint'],
        changed_files=sorted(changed),function_scope_proofs=proofs,retained_CPU_PASS=658,retained_GPU_PASS=22,
        revalidation_features=list(TARGETS),affected_previous_PASS=[],
        rationale='Only failed case functions and CUDA-only UUID equality change; CPU factory same-device move is a no-op; accepted feature AST paths remain equal.')
    path=folder/'source-impact-review.json';atomic_json(path,value)
    return dict(evidence=str(path),sha256=file_hash(path),revalidation_features=list(TARGETS),retained_CPU_PASS=658,retained_GPU_PASS=22)
