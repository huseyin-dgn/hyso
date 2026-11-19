"""Build the UX-05 source/alias/selector matrix without importing HYSo or torch.

The matrix includes all UX-03 declarations, current help aliases, and additional
public source declarations. CLI text selectors are recorded as a CLI boundary.
Detailed output goes to the selected report directory, not the terminal.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[2]
UX03 = '8ed5c0f'
UX04B = '7e954d1'
DISPATCH = {
    ('hyso/models/builder.py', 'ModelBuilder.add'): '85/57 explicit component methods',
    ('hyso/models/builder.py', 'ModelBuilder.named'): '13/1 explicit registered model methods',
    ('hyso/models/builder.py', 'ModelBuilder.modular'): 'ModelBuilder.modular_cnn',
    ('hyso/models/builder.py', 'ModelBuilder.sequential'): 'classification/regression/embedding',
    ('hyso/models/graph.py', 'GraphModelBuilder.add'): '85/57 explicit component methods and four fusion methods',
    ('hyso/models/graph.py', 'GraphModelBuilder.outputs'): 'caller-defined output names and node references',
    ('hyso/models/registry.py', 'ModelRegistry.create'): 'profile-specific typed name plus explicit Builder methods',
    ('hyso/analysis/sources.py', 'NativeBinding'): 'typed source objects and existing explicit native P01-P35 functions',
}
FREE_NAMES = {
    'name','id','title','description','statement','reason','code','model_id','model_ids',
    'workload_id','dataset_id','split','preprocessing_id','source_id','source_revision',
    'request_id','event_id','run_id','experiment_id','artifact_id','record_id','report_id',
    'path','directory','root_dir','output_name','node_id','inputs','references','target',
    'class_names','labels','factory','factory_name','source_file','factory_search_path',
    'unit','source','backend','precision','phase','domain','metric_path','field','sample_ref',
    'point_ref','protocol_ref','protocol_id','clock_id','input_kind','evaluator','averaging',
    'sample_policy','window','kind','operation_id','failure_code','retry_of_run_id',
    'resume_artifact_id','resume_from','save_last_to','ownership','role','commit_state',
}


def git_index(revision: str) -> dict:
    return json.loads(subprocess.check_output(
        ['git', 'show', f'{revision}:hyso/data/api-topics.json'], cwd=ROOT))['entries']


def declarations(tree, indexed=()):
    result = {}
    def walk(nodes, prefix=''):
        for node in nodes:
            if isinstance(node, ast.If):
                walk(node.body, prefix)
                walk(node.orelse, prefix)
            elif isinstance(node, ast.ClassDef):
                qname = prefix + node.name
                if node.name.startswith('_') and not any(q==qname or q.startswith(qname+'.') for q in indexed):
                    continue
                result[qname] = node
                walk(node.body, qname + '.')
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if not node.name.startswith('_') or node.name == '__init__' or prefix+node.name in indexed:
                    result[prefix + node.name] = node
    walk(tree.body)
    return result


def choices(annotation, aliases):
    if annotation is None:
        return []
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode='eval').body
        except SyntaxError:
            return []
    found = []
    for node in ast.walk(annotation):
        if isinstance(node, ast.Name) and node.id in aliases:
            found.extend(aliases[node.id])
        if isinstance(node, ast.Subscript) and ast.unparse(node.value).split('.')[-1] in ('Literal','_Literal'):
            try:
                value = ast.literal_eval(node.slice)
            except (ValueError, TypeError):
                continue
            found.extend(value if isinstance(value, tuple) else (value,))
    return list(dict.fromkeys(found))


def main(output: Path) -> dict:
    current = json.loads((ROOT/'hyso/data/api-topics.json').read_text(encoding='utf-8'))['entries']
    original, previous = git_index(UX03), git_index(UX04B)
    indexed = defaultdict(list)
    for symbol, entry in current.items():
        if 'path' in entry:
            indexed[entry['path'],entry['qualname']].append(symbol)
    original_keys = {(v['path'],v['qualname']) for v in original.values() if 'path' in v}
    previous_keys = {(v['path'],v['qualname']) for v in previous.values() if 'path' in v}
    enums = set()
    trees = {}
    for p in sorted((ROOT/'hyso').rglob('*.py')) + [ROOT/'hyso_guide.py']:
        path = p.relative_to(ROOT).as_posix()
        tree = ast.parse(p.read_text(encoding='utf-8-sig'), feature_version=(3,10))
        trees[path] = tree
        for node in ast.walk(tree):
            if isinstance(node,ast.ClassDef) and any('Enum' in ast.unparse(b) for b in node.bases):
                enums.add(node.name)
    rows = []
    for path, tree in trees.items():
        if '/cli/' in path or Path(path).name.startswith('_') and not path.endswith('/__init__.py'):
            # Indexed exported types from private tracing modules still belong to the public API.
            wanted = {q for p,q in indexed if p==path}
        else:
            wanted = None
        aliases = {}
        for node in ast.walk(tree):
            if isinstance(node,ast.Assign):
                values = choices(node.value,{})
                if values:
                    for target in node.targets:
                        if isinstance(target,ast.Name):
                            aliases[target.id] = list(dict.fromkeys(aliases.get(target.id,[]) + values))
        requested={q for p,q in indexed if p==path}
        for qname,node in declarations(tree,requested).items():
            if wanted is not None and qname not in wanted:
                continue
            key = path,qname
            parameters = []
            raw = []
            kwargs = None
            if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
                a = node.args
                args = a.posonlyargs+a.args+a.kwonlyargs
                defaults = [None]*(len(a.posonlyargs+a.args)-len(a.defaults))+a.defaults+a.kw_defaults
                raw = [(arg.arg,arg.annotation,default) for arg,default in zip(args,defaults) if arg.arg not in ('self','cls')]
                if a.vararg:
                    raw.append((a.vararg.arg,a.vararg.annotation,None))
                if a.kwarg:
                    kwargs = a.kwarg.arg
                    raw.append((kwargs,a.kwarg.annotation,None))
            elif isinstance(node,ast.ClassDef):
                raw = [(n.target.id,n.annotation,n.value) for n in node.body if isinstance(n,ast.AnnAssign)
                       and isinstance(n.target,ast.Name) and not n.target.id.startswith('_')]
            for name,ann,default in raw:
                annotation = ast.unparse(ann) if ann else None
                values = choices(ann,aliases)
                names = {n.id for n in ast.walk(ann) if isinstance(n,ast.Name)} if ann else set()
                if annotation and annotation.startswith(('"',"'")):
                    names.update(annotation.strip('"\'').replace('|',' ').split())
                if values:
                    role = 'closed_literal_options'
                elif names & enums:
                    role = 'existing_enum_options'
                elif name=='device' or name=='devices' or name=='gpu_selector':
                    role = 'dynamic_device_selector; Device.cpu/cuda; native validation retained'
                elif key in DISPATCH:
                    role = 'explicit_alternative: '+DISPATCH[key]
                elif name in FREE_NAMES or any(x in name for x in ('path','sha256','_id','_ids','_name','_names')):
                    role = 'caller_data_or_extensible_native_record; no new closed domain'
                else:
                    role = 'natural_python_value_or_typed_native_configuration'
                parameters.append(dict(name=name,annotation=annotation,
                                       default=ast.unparse(default) if default else None,
                                       decision=role,options=values))
            rows.append(dict(path=path,qualname=qname,line=node.lineno,
                             kind='class' if isinstance(node,ast.ClassDef) else 'callable',
                             aliases=indexed.get(key,[]),ux03=key in original_keys,
                             pre_ux05=key in previous_keys,topic=next((current[s]['topic'] for s in indexed.get(key,[])),None),
                             dispatcher_alternative=DISPATCH.get(key),variable_keywords=kwargs,
                             parameters=parameters))
    source_keys = {(r['path'],r['qualname']) for r in rows}
    missing = sorted(set(indexed)-source_keys)
    summary = dict(original_ux03_declarations=len(original_keys),original_ux03_aliases=len(original),
                   pre_ux05_indexed_declarations=len(previous_keys),current_indexed_declarations=len(indexed),
                   current_indexed_aliases=len(current),audited_source_declarations=len(rows),
                   public_source_files=len({r['path'] for r in rows}),
                   typed_literal_parameters_and_fields=sum(bool(p['options']) for r in rows for p in r['parameters']),
                   existing_enum_types=sorted(enums),
                   original_ux03_missing=sorted(original_keys-source_keys),
                   original_aliases_missing=sorted(set(original)-set(current)),
                   indexed_source_missing=missing,
                   domains=dict(Counter(r['path'].split('/')[1] if '/' in r['path'] else r['path'] for r in rows)),
                   scope='static public declaration and alias review; no workloads, GPU, P31 or acceptance')
    output.mkdir(parents=True,exist_ok=True)
    (output/'public-api-audit.json').write_text(json.dumps(dict(summary=summary,apis=rows),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    selectors=[dict(api=r['path']+':'+r['qualname'],**p) for r in rows for p in r['parameters'] if p['options']]
    (output/'closed-options.json').write_text(json.dumps(selectors,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'audit-summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    assert len(original_keys)==775 and not summary['original_ux03_missing']
    assert not summary['original_aliases_missing'] and not missing
    print('PUBLIC_API_AUDIT PASS: '+str(len(rows))+' declarations; original 775 and aliases retained')
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/ux05-project-wide-pythonic-api')
    main(parser.parse_args().output)
