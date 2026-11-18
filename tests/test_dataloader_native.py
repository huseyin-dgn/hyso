"""Real Windows spawn contracts for ordinary, caller-owned DataLoaders."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.skipif(os.name!='nt',reason='Windows spawn acceptance')
@pytest.mark.parametrize('case',['tensor_spawn','image_deterministic','image_random_cpu',
    'image_spawn_random','early_spawn','sampler_spawn','prefetch_persistent',
    'exception','exception_persistent','timeout','crash'])
def test_native_loader_integrity(case,tmp_path):
    # The existing supervisor owns only this synthetic test process tree.
    # Production measurement never launches it or imposes its test limits.
    root=Path(__file__).resolve().parent
    output=tmp_path/(case+'.json')
    outcome=subprocess.run([sys.executable,str(root/'p21_native_supervisor.py'),
        '--script',str(root/'loader_native_validation.py'),'--case',case,
        '--output',str(output),'--timeout','180'],capture_output=True,text=True,timeout=195)
    logs='\n'.join(p.read_text(encoding='utf-8',errors='replace') for p in tmp_path.glob('*.log'))
    assert outcome.returncode==0,outcome.stdout+outcome.stderr+logs
    receipt=json.loads(output.with_suffix('.receipt.json').read_text())
    assert receipt['exit_code']==0 and not receipt['timed_out']
    data=json.loads(output.read_text())
    assert data['all_workers_exited']
    if 'p21' in data:
        assert data['exact_state_rng_history_batch_content']
        assert data['p21']['worker_service']['value'] is None
