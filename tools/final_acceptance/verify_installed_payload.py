"""Deferred clean-environment source hash check, including commercial projection."""
import ast
import hashlib
import importlib.util
import json
import sys
from pathlib import Path


def main():
    source, checkout, output = map(Path, sys.argv[1:])
    import hyso
    import hyso_guide
    installed = Path(hyso.__file__).resolve().parent.parent
    if installed.is_relative_to(checkout.resolve()) or not installed.is_relative_to(Path(sys.prefix).resolve()):
        raise RuntimeError('installed payload came from checkout or another environment')
    tree = ast.parse((source / 'setup.py').read_text(encoding='utf-8'))
    projection = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='commercial_payload')
    namespace = {'json': json}
    exec(compile(ast.Module(body=[projection], type_ignores=[]), 'commercial_projection', 'exec'), namespace)
    profile = json.loads((source / 'commercial-profile.json').read_text())
    names = profile['python_files'] + [n for n in profile['support_files'] if n.startswith('hyso/')]
    bindings = {}
    for name in names:
        expected = namespace['commercial_payload'](name, (source / name).read_bytes())
        actual = (installed / name).read_bytes()
        if actual != expected:
            raise ValueError('installed source hash mismatch: '+name)
        bindings[name] = hashlib.sha256(actual).hexdigest()
    actual_python = {p.relative_to(installed).as_posix() for p in (installed/'hyso').rglob('*.py')}
    actual_python.add('hyso_guide.py')
    if actual_python != set(profile['python_files']):
        raise ValueError('installed commercial module allowlist mismatch')
    output.write_text(json.dumps(dict(status='PASS', prefix=sys.prefix,
        outside_checkout=True, bindings=bindings), indent=2), encoding='utf-8')


if __name__=='__main__':
    main()
