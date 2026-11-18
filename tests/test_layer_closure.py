"""Chronological event high-water and paired-worker uncertainty, not native proof."""
import copy
import json
import math

import pytest

from hyso.performance import LayerOverheadResult,LayerOverheadProtocol,OperatorProfileResult,profile_layers,ProfilingProtocol,measure_layer_overhead
from hyso.performance._layer_memory import observed_net_high_water,attribute_memory
from hyso.performance.layer_overhead import _paired_uncertainty,_metrics,_LIMITATIONS
from test_layer_advanced import memory_capture,overhead,_synthetic_windows,_derive


def ev(timestamp,size):return dict(start_us=timestamp,bytes=size)


def test_chronology_not_export_array_order_and_net_not_allocation_total():
    events=[ev(3,-32),ev(1,64),ev(2,32)]
    p=observed_net_high_water(events)
    assert p['value_bytes']==96 and p['status']=='derived'
    assert observed_net_high_water(list(reversed(events)))==p
    # An initial free of unknown pre-existing storage is not clamped away.
    assert observed_net_high_water([ev(1,-100),ev(2,64)])['value_bytes']==0
    assert p['absolute_allocator_peak_bytes'] is p['physical_peak_bytes'] is p['retained_bytes'] is None
    assert observed_net_high_water([])['value_bytes']==0


def test_tied_timestamp_peak_is_a_bound_not_an_invented_order():
    p=observed_net_high_water([ev(1,64),ev(1,-64)])
    assert p['status']=='partial' and p['value_bytes'] is None
    assert p['lower_bound_bytes']==0 and p['upper_bound_bytes']==64
    assert observed_net_high_water([ev(1,-64),ev(1,64)])==p
    q=observed_net_high_water([ev(0,100),ev(1,-100),ev(1,16)])
    assert q['lower_bound_bytes']==100 and q['upper_bound_bytes']==116
    assert observed_net_high_water([ev(1,10),ev(1,20)])['value_bytes']==30


@pytest.mark.parametrize('bad',[[ev(float('nan'),1)],[ev(0,True)],[ev(-1,1)],[ev(0,0)]])
def test_invalid_chronology_cannot_create_a_peak(bad):
    with pytest.raises(ValueError):observed_net_high_water(bad)


def test_real_event_peak_matches_independent_prefix_and_legacy_v4(memory_capture):
    d=memory_capture.to_dict();raw=d['raw_evidence'];m=profile_layers(memory_capture).to_dict()['measurements']['layer_memory']
    for row in m['invocations']:
        summary=row['domains']['cpu']['inclusive']
        if summary is None:continue
        selected=[e for e,a in zip(raw['memory_events'],m['assignments']) if row['invocation_id'] in a['inclusive_invocation_ids']]
        balance=peak=0
        for e in sorted(selected,key=lambda e:e['start_us']):balance+=e['bytes'];peak=max(peak,balance)
        p=summary['observed_net_high_water']
        assert p['lower_bound_bytes']<=peak<=p['upper_bound_bytes']
        assert balance==summary['net_change_bytes']
    for module in m['modules']:
        h=module['domains']['cpu']['inclusive']['max_invocation_observed_net_high_water']
        calls=[r for r in m['invocations'] if raw['invocations'][r['invocation_id']]['module_id']==module['module_id']]
        assert h['upper_bound_bytes']==max(r['domains']['cpu']['inclusive']['observed_net_high_water']['upper_bound_bytes'] for r in calls)
    legacy=copy.deepcopy(d);legacy['protocol']['instrumentation']['evidence_version']=4
    legacy['measurements']['layer_memory']=attribute_memory(raw['memory_events'],raw['trace_events'],raw['invocations'],raw['module_inventory'],'cpu')
    restored=OperatorProfileResult(legacy);layer=profile_layers(restored).to_dict()
    assert layer['protocol']['profile_evidence_version']==4
    assert 'max_invocation_observed_net_high_water' not in layer['measurements']['modules'][0]['memory']['cpu']['inclusive']
    upgraded=profile_layers(restored,memory_analysis_version=2).to_dict()
    assert upgraded['protocol']['profile_evidence_version']==5
    assert upgraded['identity']['source_result_id']==legacy['result_id']
    assert upgraded['measurements']['layer_memory']==m


def test_serialized_new_high_water_cannot_be_forged(memory_capture):
    d=memory_capture.to_dict()
    d['measurements']['layer_memory']['invocations'][0]['domains']['cpu']['inclusive']['observed_net_high_water']['value_bytes']=999999
    with pytest.raises(ValueError):OperatorProfileResult(d)


def test_student_t_uses_worker_pairs_and_small_sample_uncertainty():
    u=_paired_uncertainty([1.,2.,3.])
    assert u['mean_difference_ms']==2 and u['degrees_of_freedom']==2
    assert u['standard_error_ms']==pytest.approx(1/math.sqrt(3))
    assert u['half_width_ms']==pytest.approx(4.30265272975/math.sqrt(3))
    assert _paired_uncertainty([1.,1.])['half_width_ms']==0
    assert _paired_uncertainty([-1.,1.])['relative_half_width'] is None


def test_priming_is_declared_retained_and_does_not_change_effect_math(overhead):
    d=_synthetic_windows(overhead,[2.,2.,2.],[3.,3.,3.]);m=_derive(d)
    assert m['signed_overhead_percent']==50 and m['observed_mean_difference_ms']==1
    assert m['selection']==dict(start_index=2,stop_index=5,forward_windows_per_worker=3)
    assert m['priming']['windows_ms']['instrumented']==[[100.,99.],[100.,99.]]
    assert m['uncertainty']['pairs']==2 and m['uncertainty']['half_width_ms']==0
    assert m['arms']['control']['samples']==6
    assert all(len(p['control']['windows_ms'])==5 for p in d['raw_evidence']['pairs'])


def test_low_forward_variance_does_not_hide_between_pair_uncertainty(overhead):
    d=_synthetic_windows(overhead,[2.,2.,2.],[3.,3.,3.]);p=d['raw_evidence']['pairs'][1]['instrumented']
    p['raw_evidence']['overhead_experiment']['windows_ms']=[100.,99.,3.1,3.1,3.1]
    p['measurements']['overhead']['instrumented_host_forward_windows_ms']=[100.,99.,3.1,3.1,3.1]
    m=_derive(d)
    assert m['variance_accepted'] and not m['precision_accepted']
    assert m['status']=='unsupported' and m['signed_overhead_percent'] is None
    assert m['observed_mean_difference_ms']==pytest.approx(1.05)
    assert m['uncertainty']['half_width_ms']>0


def test_legacy_overhead_result_reopens_without_new_analysis(overhead,tmp_path):
    d=overhead.to_dict();p=d['protocol'];p.pop('analysis_version');p.pop('requested_capture_configuration')
    p['configuration'].pop('priming_iterations');p['configuration'].pop('max_relative_half_width')
    d['measurements']=_metrics(d['raw_evidence']['pairs'],p['recipe'],p['capture_configuration'],LayerOverheadProtocol(**p['configuration']))
    d['diagnostics']['limitations']=_LIMITATIONS
    old=LayerOverheadResult(d);j=tmp_path/'legacy.json';c=tmp_path/'legacy.csv';old.to_json(j);old.to_csv(c)
    assert LayerOverheadResult.from_json(j).to_dict()==LayerOverheadResult.from_csv(c).to_dict()==d
    assert 'uncertainty' not in d['measurements']


def test_declared_priming_and_uncertainty_tamper_are_rejected(overhead):
    for field in ('prefix','bound'):
        d=overhead.to_dict()
        if field=='prefix':d['protocol']['configuration']['priming_iterations']=1
        else:d['measurements']['uncertainty']['lower_ms']-=1
        with pytest.raises(ValueError):LayerOverheadResult(d)


@pytest.mark.parametrize('kw',[dict(priming_iterations=0),dict(priming_iterations=11),
    dict(max_relative_half_width=0),dict(max_relative_half_width=2)])
def test_invalid_method_policy_is_rejected(kw):
    with pytest.raises(ValueError):LayerOverheadProtocol(**kw)


def test_extra_priming_respects_existing_forward_bound_before_worker_launch(monkeypatch):
    import hyso.performance.layer_overhead as module
    from test_layer_advanced import recipe
    def forbidden(*a,**kw):raise AssertionError('must reject before launching worker')
    monkeypatch.setattr(module,'_execute_profile',forbidden)
    with pytest.raises(ValueError,match='100 forwards'):
        measure_layer_overhead(recipe(),capture_protocol=ProfilingProtocol(warmup_iterations=2,measure_iterations=98))
