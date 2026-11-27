"""Small selected CPU engineering checks using existing dependency sites only."""
from pathlib import Path
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[2]

def main():
    os.environ['CUDA_VISIBLE_DEVICES']=''
    baseline=json.loads((ROOT/'outputs/export-acceptance-20261127/plan.json').read_text(encoding='utf-8'))
    cpu=json.loads((ROOT/'outputs/release-acceptance-20261125/source-runtime.json').read_text(encoding='utf-8'))
    # A CPU interpreter is mandatory. Sites contribute test/ONNX dependencies;
    # its existing CPU Torch stays first, avoiding a GPU runtime import.
    if Path(sys.executable).resolve()!=Path(cpu['python']).resolve():
        raise RuntimeError('launch this helper with the retained CPU interpreter')
    for value in [baseline['runtime']['site'],baseline['feature_tools']['ort']['site'],*cpu.get('fallback_test_dependencies',[])]:
        if Path(value).is_dir() and value not in sys.path:sys.path.append(value)
    sys.path.insert(0,str(ROOT))
    import torch
    if torch.version.cuda is not None:raise RuntimeError('CPU-only Torch required')
    import pytest
    args=sys.argv[1:] or ['tests/test_export_development.py']
    return pytest.main(['-q','-p','no:cacheprovider',*args])

if __name__=='__main__':raise SystemExit(main())
