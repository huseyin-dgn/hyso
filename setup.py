"""Build the explicitly enumerated commercial source distribution."""
import json
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py
from setuptools.command.sdist import sdist


ROOT = Path(__file__).resolve().parent
PROFILE = json.loads((ROOT / 'commercial-profile.json').read_text(encoding='utf-8'))
MODULES = frozenset(PROFILE['python_files'])
SOURCE_FILES = MODULES | frozenset(PROFILE['support_files'])


class CommercialBuild(build_py):
    """Copy only reviewed modules, including when building a wheel directly."""

    def find_package_modules(self, package, package_dir):
        modules = super().find_package_modules(package, package_dir)
        return [row for row in modules
                if (row[0].replace('.', '/') + '/' + row[1] + '.py') in MODULES]


class CommercialSource(sdist):
    """Apply the same allowlist to sdist payloads and their source inventory."""

    def make_release_tree(self, base_dir, files):
        allowed = [name for name in files
                   if Path(name).as_posix() in SOURCE_FILES
                   or Path(name).as_posix().startswith('hyso.egg-info/')]
        super().make_release_tree(base_dir, allowed)


setup(cmdclass={'build_py': CommercialBuild, 'sdist': CommercialSource})
