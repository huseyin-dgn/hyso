"""Changed B4 boundary in shipped payloads and four-language retained reports."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import pytest

from test_packaging import distributions as distributions,commercial_bytes
from test_b4_containment import binding,forbid,BLOCKED
from hyso import analysis as api,performance as perf
from hyso.reporting import create_report,PresentationReport

ROOT=Path(__file__).resolve().parents[1]
CHANGED_MODULES=('hyso/performance/isolated_training_memory.py','hyso/performance/_saved_activation.py',
    'hyso/analysis/native_adapters.py','hyso/reporting/render.py')


@pytest.mark.parametrize('language',['tr','en','de','es'])
def test_JSON_Markdown_HTML_preserve_identical_blocked_execution(language,tmp_path,monkeypatch):
    monkeypatch.setattr(perf,'measure_training_memory_isolated',forbid)
    plan=api.plan_analysis(api.ModelContext('A'),features='P23',device='cuda:0',execution_mode='native')
    result=api.execute_analysis(api.preflight_analysis(plan,(binding(),)),authorized_features=('P23',),explicit_authorization=True)
    context=api.AnalysisContext(workload_id='b4-report-consistency',task='classification',model={'id':'A'},dataset={})
    for level in ('beginner','technical','expert'):
        report=create_report(result,language=language,level=level,context=context)
        payload=json.loads(report.to_json())
        assert payload['sources'][0]['execution']==result.to_dict()
        p23=next(r for r in payload['sources'][0]['execution']['features'] if r['feature_id']=='P23')
        assert p23['status']=='UNSUPPORTED' and BLOCKED in p23['reasons'] and p23['evidence_record'] is None
        assert BLOCKED in report.to_markdown().split('```')[0] and BLOCKED in report.to_html()
        path=tmp_path/(language+'-'+level+'.json');report.to_json(path)
        assert PresentationReport.from_json(path).to_dict()==report.to_dict()
        assert not payload['sources'][0]['metrics']


def test_changed_modules_keep_existing_callable_signatures_and_shipping_contract():
    def signatures(text):
        tree=ast.parse(text);result={}
        def walk(body,prefix=''):
            for n in body:
                if isinstance(n,ast.ClassDef):walk(n.body,prefix+n.name+'.')
                elif isinstance(n,ast.FunctionDef):result[prefix+n.name]=(ast.dump(n.args),ast.dump(n.returns) if n.returns else None)
        walk(tree.body);return result
    profile=json.loads((ROOT/'commercial-profile.json').read_text(encoding='utf-8'))
    assert len(profile['python_files'])==172
    for name in CHANGED_MODULES:
        before=subprocess.check_output(['git','show','d2d4af291aac4c56a4978b74c824aca014461dc3:'+name],cwd=ROOT).decode('utf-8')
        assert signatures(before)==signatures((ROOT/name).read_text(encoding='utf-8'))
        assert name in profile['python_files']


INSTALLED_SMOKE=r'''
import json,pathlib,sys
target=pathlib.Path(sys.argv[1]).resolve();sys.path.insert(0,str(target))
import torch
assert not torch.cuda.is_initialized()
def denied(*a,**kw):raise AssertionError('Installed B4 capture reached forbidden CUDA/worker work')
torch.cuda._lazy_init=denied;torch._C._cuda_init=denied
import hyso
from hyso import performance as perf
from hyso.performance import isolated_training_memory as module
assert pathlib.Path(hyso.__file__).resolve().is_relative_to(target)
module._launch=denied
for key in ('is_initialized','is_available','device_count','set_device'):
    def guard(*a,_key=key,**kw):return denied(*a,**kw)
    setattr(torch.cuda,key,guard)
recipe=perf.TrainingMemoryRecipe('absent:factory',device='cuda:0')
protocol=perf.IsolatedTrainingMemoryProtocol(saved_activation_lifetime=True,memory_protocol=perf.TrainingMemoryProtocol(device='cuda:0'))
value=perf.measure_training_memory_isolated(recipe,protocol=protocol);data=value.to_dict()
assert data['outcome']=='unsupported' and 'BLOCKED_B4_UNRESOLVED' in data['reason']
assert data['measurement'] is data['receipt'] is None
assert all(m['value'] is None for m in value.basic_metrics()['metrics'].values())
print(json.dumps(dict(status='PASS',package=str(hyso.__file__),worker_started=False,measurement=None)))
'''


@pytest.mark.parametrize('kind',['wheel','sdist'])
def test_shipped_current_boundary_blocks_before_worker_without_checkout_import(distributions,kind,tmp_path):
    _,wheel,sdist=distributions;files={}
    if kind=='wheel':
        with zipfile.ZipFile(wheel) as z:
            files={n:z.read(n) for n in z.namelist() if n.startswith('hyso/') and not n.endswith('/')}
    else:
        with tarfile.open(sdist) as z:
            for m in z.getmembers():
                parts=Path(m.name).parts
                if m.isfile() and len(parts)>1 and parts[1]=='hyso':files[Path(*parts[1:]).as_posix()]=z.extractfile(m).read()
    for name in CHANGED_MODULES:assert files[name]==commercial_bytes(name)
    target=(tmp_path/'payload').resolve();target.mkdir()
    for name,payload in files.items():
        p=(target/name).resolve();assert p.is_relative_to(target)
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(payload)
    result=subprocess.run([sys.executable,'-I','-B','-c',INSTALLED_SMOKE,str(target)],cwd=tmp_path,
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1'),
        text=True,capture_output=True,timeout=180)
    assert result.returncode==0,result.stdout+result.stderr
    value=json.loads(result.stdout)
    assert value['status']=='PASS' and value['worker_started'] is False and value['measurement'] is None
