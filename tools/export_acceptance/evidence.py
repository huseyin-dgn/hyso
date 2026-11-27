"""Stdlib post-cleanup validation of real export/runtime/equivalence witnesses."""
from pathlib import Path
from tools.cuda_acceptance.feature_evidence import verify_feature as original,SafetyViolation
from tools.final_acceptance.common import file_hash
import base64
import hashlib
import json
import math
import struct

def verify_comparisons(data):
    # Load the shared stdlib-only proof checker without importing hyso/Torch in
    # the campaign manager. The file is part of the sealed source inventory.
    import importlib.util
    path=Path(__file__).resolve().parents[2]/'hyso/export/_evidence.py'
    spec=importlib.util.spec_from_file_location('export_numerical_evidence',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.verify_comparisons(data)


def verify_feature(feature,data,runtime):
    if feature=='ADMISSION':return original(feature,data,runtime)
    native_features=('E03_TRT','E04_TRT','E07_TRT_FP16','E08_TRT_QDQ','E09_TRT_COHORT','E10_TRT_BENCHMARK','E11_TRT_COMPARISON')
    if feature not in ('E02_CUDA','E05_CUDA',*native_features):raise ValueError('unreviewed export feature')
    def need(v,reason):
        if not v:raise ValueError(feature+': '+reason)
    need(data.get('deployment_verified') is True,'conversion alone is not deployment PASS')
    body={k:v for k,v in data.items() if k!='content_sha256'}
    need(hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()==data['content_sha256'],'metadata digest mismatch')
    need(set(data['stages'])=={'dependencies','admission','conversion','artifact','runtime_loading','runtime_execution','equivalence','preservation','cleanup'},'complete stage ledger')
    need(all(r['status']=='PASS' for r in data['stages'].values()),'all nine actual stages required')
    need(data['artifact_files'] and any(Path(data['artifact_path']).name==r['path'] for r in data['artifact_files']),'complete main artifact manifest')
    for r in data['artifact_files']:
        p=Path(data['artifact_root'])/r['path']
        need(not p.is_symlink() and p.resolve().is_relative_to(Path(data['artifact_root']).resolve()) and p.stat().st_size==r['bytes'] and file_hash(p)==r['sha256'],'artifact hash/path mismatch')
    comparisons=data['stages']['equivalence']['comparisons']
    need(bool(comparisons) and all(r['matched'] is True and r['finite'] is True and r['max_absolute_error']>=0 and r['max_relative_error']>=0 for r in comparisons),'actual finite tolerance-bound output comparisons')
    verify_comparisons(data)
    native=data['runtime']
    if 'report' in native:
        c=native['controller'];report=native['report']
    else:c=data['controller'];report=native
    need(c['exit_code']==0 and c['reason'] is None and c['kill_on_close'] and c['cleanup']['active_processes']==0,'nested native cleanup')
    need(report['environment']['gpu_uuid']==runtime['uuid'] and report['loaded_binaries'],'actual UUID and loaded binaries')
    if feature=='E02_CUDA':
        capture=report['runtime'];analysis=capture['analysis']
        need(analysis['status']=='expected_execution_observed' and analysis['fallback_detected'] is False and not analysis['uncovered_source_nodes'] and bool(analysis['events']) and all(e['provider']=='CUDAExecutionProvider' for e in analysis['events']),'real CUDA node events without CPU fallback')
    if feature=='E05_CUDA':need(report['runtime']['completed'] is True and report['runtime']['device']=='cuda:0','completed real TorchScript CUDA forward')
    if feature in native_features:need(report['completed'] is True and report['outputs'] and report['allocator']['peak_requested_bytes']>0,'real bounded engine load/execution, not mock or serialization alone')
    extension=data.get('development_extension',{})
    if feature=='E09_TRT_COHORT':need(extension.get('validation',{}).get('numerical',{}).get('status')=='PASS','real cohort output validation required')
    if feature in ('E10_TRT_BENCHMARK','E11_TRT_COMPARISON'):
        measured=extension.get('benchmark',{});receipt=measured.get('native_receipt',{})
        need(measured.get('samples_ms') and measured.get('conditions',{}).get('device')=='cuda','actual native timing samples required')
        need(receipt.get('controller',{}).get('cleanup',{}).get('active_processes')==0,'benchmark owned cleanup required')
    if feature=='E11_TRT_COMPARISON':need(extension.get('comparison',{}).get('speedup') is None and extension['comparison']['comparable'] is False,'cross-boundary CPU/GPU speedup must be rejected')
