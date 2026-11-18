"""Read executable and golden Python fixtures without depending on documentation."""
import copy
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent / 'fixtures'

def load_fixture(name):
    # Historic names retain hyphens on disk; they are loaded by path, not import.
    path = ROOT / (name.removesuffix('.json') + '.py')
    return copy.deepcopy(runpy.run_path(str(path))['DATA'])

class ExampleFixture:
    def __init__(self, name):
        self.path = ROOT / 'examples' / (name + '.py')
    def read_text(self, encoding='utf-8'):
        blocks = runpy.run_path(str(self.path))['BLOCKS']
        return ''.join('```python\n' + block + '```\n' for block in blocks)
    def __str__(self):
        return str(self.path)

def example_fixture(document):
    name = 'readme' if document == 'README.md' else Path(document).stem.replace('-', '_')
    return ExampleFixture(name)
