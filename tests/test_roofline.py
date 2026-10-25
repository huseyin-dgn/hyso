"""Independent P18 math, real CPU integration, provenance and unsupported domains."""
from dataclasses import FrozenInstanceError
import copy
import json
from pathlib import Path

import pytest
import torch

from hyso.performance import (RooflineProtocol,RooflineReference,RooflineResult,analyze_roofline,measure_roofline,
    ColdStartRecipe,ProfilingProtocol,profile_operators,analyze_flops)
from hyso.performance.roofline import _math,_payload


def test_math_si_units_and_roof_minimum():
    # 2 billion logical ops / .5 s = 4 GFLOP/s. 1 billion bytes -> intensity 2.
    m=_math(2e9,.5,1e9,20e9,3e9)
    assert m==dict(achieved_flops_per_second=4e9,achieved_gflops_per_second=4.,
        arithmetic_intensity_flops_per_byte=2.,scenario_roof_flops_per_second=6e9,scenario_ridge_flops_per_byte=20/3)
    assert _math(2e9,.5,1e9,5e9,3e9)['scenario_roof_flops_per_second']==5e9
    assert _math(0,.5)['achieved_gflops_per_second']==0
    assert _math(2e9,.5)['arithmetic_intensity_flops_per_byte'] is None


@pytest.mark.parametrize('arguments',[(1,0),(1,-1),(-1,1),(True,1),(1,float('nan')),
    (1,1,0),(1,1,True),(1,1,2,4,None),(1e308,1e-308)])
def test_math_invalid_or_overflow_is_not_a_measurement(arguments):
    with pytest.raises((ValueError,OverflowError)):_math(*arguments)


def test_payload_bytes_are_explicit_occurrences_not_storage_or_dram():
    tensor=dict(kind='tensor',shape=[2,3],dtype='torch.float32',device='cpu')
    assert _payload(dict(input=tensor,alias=tensor))==48  # deliberate duplicate logical occurrence
    assert _payload([tensor,dict(kind='tensor',shape=[],dtype='torch.int64',device='cpu')])==32
    assert _payload(dict(kind='tensor',shape=[2],dtype='unknown',device='cpu')) is None
    assert _payload(dict(kind='unrecorded',reason='metadata_depth_limit')) is None


def recipe(factory='spatial',**kwargs):
    return ColdStartRecipe('profiling_factories:'+factory,factory_search_path=Path(__file__).parent,
        input_shape=(2,2,5,7),**kwargs)


def capture_protocol():
    return ProfilingProtocol(warmup_iterations=0,measure_iterations=2,timeout_seconds=45,
        cpu_memory_limit_bytes=3*1024**3,input_limit_bytes=32*1024**2)


@pytest.fixture(scope='module')
def captured():return profile_operators(recipe(),protocol=capture_protocol())


def test_real_cpu_same_capture_p12_and_measured_latency(captured,monkeypatch):
    import hyso.performance.roofline as module
    before=captured.to_dict()
    def forbidden(*a,**kw):raise AssertionError('analysis must not execute or initialize CUDA')
    monkeypatch.setattr(module,'profile_operators',forbidden)
    monkeypatch.setattr(torch.cuda,'_lazy_init',forbidden)
    result=analyze_roofline(captured);d=result.to_dict();m=d['measurements']
    expected=2*4*5*7*2*9*2+2*4*5*7
    assert expected==10360
    assert m['logical_flops']['value']==analyze_flops(captured).to_dict()['measurements']['total_flops_per_batch']==expected
    assert m['logical_flops']['status']=='estimated' and m['timing']['status']=='measured'
    seconds=sum(before['measurements']['overhead']['instrumented_host_forward_windows_ms'])/1000
    assert m['achieved_compute']['value']==pytest.approx(2*expected/seconds/1e9)
    assert m['traffic']['status']=='estimated' and m['arithmetic_intensity']['status']=='estimated'
    assert m['arithmetic_intensity']['value']==pytest.approx(expected/m['traffic']['value'])
    assert m['hardware_roof_calibration']['value'] is m['bottleneck']['value'] is m['hardware_headroom']['value'] is None
    assert captured.to_dict()==before
    assert d['identity']['source_result_id']==before['result_id'] and d['raw_evidence']['source']['result_id']==before['result_id']


def test_unmodeled_ops_do_not_become_total_flops_or_exact_intensity():
    r=measure_roofline(recipe('Norm'),profiling_protocol=capture_protocol()).to_dict()['measurements']
    assert r['logical_flops']['coverage']['unknown_calls']>0
    assert r['logical_flops']['value'] is r['achieved_compute']['value'] is r['arithmetic_intensity']['value'] is None
    assert r['achieved_compute']['status']=='partial'
    assert r['achieved_compute']['supported_subtotal_gflops_per_second']>0


def test_theoretical_reference_is_not_calibration_or_bottleneck(captured):
    ref=RooflineReference.for_profile(captured,compute_flops_per_second=1e12,memory_bytes_per_second=1e11,
        source='synthetic theoretical scenario, not native hardware calibration')
    m=analyze_roofline(captured,reference=ref).to_dict()['measurements']
    assert m['theoretical_reference']['verification']=='unverified'
    assert m['theoretical_reference']['scenario_roof_flops_per_second']==pytest.approx(
        min(1e12,1e11*m['arithmetic_intensity']['value']))
    assert m['bottleneck']['status']==m['hardware_headroom']['status']=='unsupported'
    bad=copy.copy(ref);object.__setattr__(bad,'profile_sha256','0'*64)
    with pytest.raises(ValueError,match='another profile'):analyze_roofline(captured,reference=bad)


def test_no_traffic_estimate_no_arithmetic_intensity(captured):
    m=analyze_roofline(captured,protocol=RooflineProtocol(traffic='unsupported')).to_dict()['measurements']
    assert m['traffic']['value'] is m['arithmetic_intensity']['value'] is None
    assert m['achieved_compute']['value']>0


def test_measure_wrapper_no_default_quota_or_extra_capture_and_errors_propagate(captured,monkeypatch):
    import hyso.performance.roofline as module
    from hyso.performance import ProfilingError
    calls=[]
    def capture(source,*,protocol):
        calls.append(protocol)
        return captured
    monkeypatch.setattr(module,'profile_operators',capture)
    measure_roofline(recipe())
    assert len(calls)==1 and calls[0].warmup_iterations==0 and calls[0].measure_iterations==1
    assert calls[0].cpu_memory_limit_bytes is calls[0].input_limit_bytes is calls[0].cuda_memory_fraction is None
    failure=ProfilingError(dict(status='oom',reason='controlled error; no allocation stress'))
    def failed(*a,**kw):raise failure
    monkeypatch.setattr(module,'profile_operators',failed)
    with pytest.raises(ProfilingError) as observed:measure_roofline(recipe())
    assert observed.value is failure


def test_unrelated_latency_transfer_and_allocator_results_cannot_substitute():
    for source in (dict(format='hyso.performance.transfer'),dict(allocated_bytes=1000),object()):
        with pytest.raises(TypeError,match='same-capture'):analyze_roofline(source)
    with pytest.raises(ValueError):RooflineProtocol(traffic='P15_H2D_bandwidth')


@pytest.mark.parametrize('field',['rate','intensity','source_hash','precision','hardware','limitation'])
def test_tamper_rejected_from_raw_evidence(captured,field):
    d=analyze_roofline(captured).to_dict()
    if field=='rate':d['measurements']['achieved_compute']['value']+=1
    elif field=='intensity':d['measurements']['traffic']['value']+=1
    elif field=='source_hash':d['raw_evidence']['source_sha256']='0'*64
    elif field=='precision':d['provenance']['input']['dtype']='torch.float64'
    elif field=='hardware':d['measurements']['bottleneck'].update(value='compute_bound',status='measured')
    else:d['diagnostics']['limitations']=[]
    with pytest.raises(ValueError):RooflineResult(d)


def test_json_csv_roundtrip_immutability_and_no_clobber(captured,tmp_path):
    result=analyze_roofline(captured);before=result.to_dict()
    j=tmp_path/'roof.json';c=tmp_path/'roof.csv';result.to_json(j);result.to_csv(c)
    assert RooflineResult.from_json(j).to_dict()==RooflineResult.from_csv(c).to_dict()==before
    result.to_dict()['measurements'].clear();assert result.to_dict()==before
    with pytest.raises((FrozenInstanceError,AttributeError)):result._payload='bad'
    with pytest.raises(FileExistsError):result.to_json(j)
    with pytest.raises(FileExistsError):result.to_csv(c)
