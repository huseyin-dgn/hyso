"""Inventory owned public Python definitions without importing optional runtimes.

Used for release documentation review; underscore modules and CLI internals are
excluded unless explicitly exported from a public namespace.
"""
from __future__ import annotations

import ast
import json
from importlib.util import resolve_name
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# These are implementation adapters, not end-user entry points. Explicit exports
# from them (for example trace_data_stage) are still included.
IMPLEMENTATION_MODULES = {
    'hyso.export.artifacts', 'hyso.export.dependencies', 'hyso.models.graph_ops',
    'hyso.training.resume', 'hyso.analysis.native_adapters',
}
PUBLIC_EXPORT_MODULES = {'hyso.export.contracts','hyso.export.converters',
    'hyso.export.native','hyso.export.runtime','hyso.export.precision',
    'hyso.export.validation','hyso.export.deployment','hyso.export.presentation'}


def public_definitions(root=ROOT):
    """Yield shipped public definitions and their defining source locations."""
    profile = json.loads((root / 'commercial-profile.json').read_text())
    exported = set()
    for filename in profile['python_files']:
        if not filename.endswith('/__init__.py'):
            continue
        tree = ast.parse((root / filename).read_text(encoding='utf-8-sig'))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if not (alias.asname or alias.name).startswith('_'):
                        package = filename.removesuffix('/__init__.py').replace('/', '.')
                        module = resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
                        exported.add((module, alias.name))
    for filename in profile['python_files']:
        if not filename.startswith('hyso/') or '/cli/' in filename:
            continue
        tree = ast.parse((root / filename).read_text(encoding='utf-8-sig'))
        internal = any(p.startswith('_') for p in Path(filename).parts[1:-1]) or Path(filename).stem.startswith('_')
        for node in tree.body:
            if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            module = filename.removesuffix('.py').replace('/', '.').removesuffix('.__init__')
            if node.name.startswith('_') or (internal or module in IMPLEMENTATION_MODULES | PUBLIC_EXPORT_MODULES) and (module, node.name) not in exported:
                continue
            if filename == 'hyso/models/regularization.py' and not isinstance(node, ast.ClassDef):
                continue
            # Registry implementation utilities are not model construction APIs.
            if filename in ('hyso/models/components.py', 'hyso/models/advanced.py', 'hyso/models/sampling.py') and not isinstance(node, ast.ClassDef):
                continue
            yield filename, node.name, node
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (not child.name.startswith('_') or child.name in ('__init__', '__call__')):
                        if any(isinstance(d, ast.Attribute) and d.attr == 'overload' for d in child.decorator_list):
                            continue
                        yield filename, node.name + '.' + child.name, child


def parameters(node):
    """Return explicit parameters or dataclass fields, excluding the receiver."""
    if isinstance(node, ast.ClassDef):
        if any(isinstance(d, ast.Call) and any(k.arg == 'init' and isinstance(k.value, ast.Constant) and k.value.value is False for k in d.keywords) for d in node.decorator_list):
            return []
        if not any(isinstance(d, ast.Call) and isinstance(d.func, ast.Name) and d.func.id == 'dataclass' or isinstance(d, ast.Name) and d.id == 'dataclass' for d in node.decorator_list):
            return []
        return [n.target.id for n in node.body if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and not n.target.id.startswith('_')]
    return [a.arg for a in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs, *([node.args.vararg] if node.args.vararg else []), *([node.args.kwarg] if node.args.kwarg else [])] if a.arg not in ('self','cls')]


def inventory(root=ROOT):
    """Return reproducible doc coverage; presence is not semantic certification."""
    result=[]
    for filename, name, node in public_definitions(root):
        doc=ast.get_docstring(node) or ''
        result.append(dict(file=filename,name=name,kind='class' if isinstance(node,ast.ClassDef) else 'callable',
                           line=node.lineno,parameters=parameters(node),missing=[p for p in parameters(node) if p not in doc],docstring=bool(doc)))
    return result


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();rows=inventory()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(dict(definitions=len(rows),classes=sum(r['kind']=='class' for r in rows),missing_docstrings=sum(not r['docstring'] for r in rows),missing_parameters=sum(len(r['missing']) for r in rows))))
