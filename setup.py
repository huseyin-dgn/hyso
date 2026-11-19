"""Build the explicitly enumerated commercial source distribution."""
import json
from pathlib import Path
import tempfile

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
PROFILE = json.loads((ROOT / 'commercial-profile.json').read_text(encoding='utf-8'))
MODULES = frozenset(PROFILE['python_files'])
SOURCE_FILES = MODULES | frozenset(PROFILE['support_files'])


def commercial_payload(name, payload):
    """Omit only labeled development convenience methods and their new index rows.

    Removing their definitions also keeps installed IDE completion profile-correct.
    Original algorithms, generic add/named APIs and pre-existing guide rows remain.
    The transform is idempotent for rebuilding a wheel from the commercial sdist.
    """
    if name in ('hyso/models/builder.py', 'hyso/models/graph.py'):
        lines=payload.splitlines(keepends=True)
        starts=[i for i,line in enumerate(lines) if line.strip()==b'# BEGIN DEVELOPMENT REGISTRY WRAPPERS']
        ends=[i for i,line in enumerate(lines) if line.strip()==b'# END DEVELOPMENT REGISTRY WRAPPERS']
        if not starts and not ends:return payload
        if len(starts)!=1 or len(ends)!=1 or starts[0]>=ends[0]:
            raise ValueError('Invalid development wrapper boundaries: '+name)
        return b''.join(lines[:starts[0]]+lines[ends[0]+1:])
    if name=='hyso/data/api-topics.json':
        data=json.loads(payload.decode('utf-8'))
        data['entries']={k:v for k,v in data['entries'].items()
                         if 'commercial' in v.get('profiles',['commercial'])}
        if 'registry_completion' in data:
            data['registry_completion']['components']={k:v for k,v in data['registry_completion']['components'].items()
                                                       if 'commercial' in v['profiles']}
        return (json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    return payload


def write_commercial(path, name):
    """Replace a build-tree file without modifying a possible source hardlink."""
    before=path.read_bytes();after=commercial_payload(name,before)
    if before==after:return
    with tempfile.NamedTemporaryFile(dir=path.parent,prefix=path.name+'.',delete=False) as stream:
        temporary=Path(stream.name);stream.write(after)
    try:temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


class CommercialBuild(build_py):
    """Copy only reviewed modules, including when building a wheel directly."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [row for row in modules
                if (row[0].replace('.', '/') + '/' + row[1] + '.py') in MODULES]

    def find_modules(self):
        return [row for row in super().find_modules() if row[1]+'.py' in MODULES]

    def build_module(self, module, module_file, package):
        result=super().build_module(module,module_file,package)
        parts=package.split('.') if isinstance(package,str) else package
        name='/'.join([*parts,module])+'.py'
        if name in ('hyso/models/builder.py','hyso/models/graph.py'):
            write_commercial(Path(result[0]),name)
        return result

    def build_package_data(self):
        super().build_package_data()
        name='hyso/data/api-topics.json'
        write_commercial(Path(self.build_lib)/name,name)


class CommercialSource(sdist):
    """Apply the same allowlist to sdist payloads and their source inventory."""

    def make_release_tree(self, base_dir, files):
        allowed = [name for name in files
                   if Path(name).as_posix() in SOURCE_FILES
                   or Path(name).as_posix().startswith('hyso.egg-info/')]
        super().make_release_tree(base_dir, allowed)
        for name in ('hyso/models/builder.py','hyso/models/graph.py','hyso/data/api-topics.json'):
            write_commercial(Path(base_dir)/name,name)


setup(cmdclass={'build_py': CommercialBuild, 'sdist': CommercialSource})
