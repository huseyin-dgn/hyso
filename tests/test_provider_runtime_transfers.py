"""ORT optimized transfer graph reader regressions; these are protocol tests."""
import base64
import copy
import hashlib

import onnx
import pytest
from onnx import helper

from hyso.performance import RuntimeResult
from hyso.performance import providers as module
from hyso.performance.runtime_result import _result
from test_provider_protocol import simulated, rebind


def transfer_graph(encoded):
    model=onnx.load_model_from_string(base64.b64decode(encoded))
    model.graph.node[0].output[0]='Y_device'
    model.graph.node.append(helper.make_node('MemcpyToHost',['Y_device'],['Y'],name='copy'))
    return model


def test_transfer_is_accepted_only_in_optimized_graph():
    from deployment_support import onnx_model
    raw,_=onnx_model();model=transfer_graph(base64.b64encode(raw).decode())
    encoded=base64.b64encode(model.SerializeToString()).decode()
    with pytest.raises(onnx.checker.ValidationError):module._decode_model(encoded)
    decoded,graph=module._decode_model(encoded,optimized=True)
    assert decoded==model.SerializeToString() and graph.SerializeToString()==decoded


def test_transfer_graph_maps_original_node_without_erasing_runtime_event(monkeypatch,tmp_path):
    result,_=simulated(monkeypatch,tmp_path,providers=['CPUExecutionProvider'],expected={'relu':'CPUExecutionProvider'},optimization='disabled')
    e=result.to_dict()['evidence'];e['graph_analysis_version']=2
    r=e['report'];graph=transfer_graph(r['model_base64']);raw=graph.SerializeToString()
    r['optimized_base64']=base64.b64encode(raw).decode();r['optimized_sha256']=hashlib.sha256(raw).hexdigest()
    event=copy.deepcopy(next(x for x in r['trace'] if x.get('cat')=='Node'))
    event['name']='copy_kernel_time';event['args']['op_name']='MemcpyToHost';r['trace'].append(event);rebind(e)
    output=_result('P35',e);d=output.provider_diagnostics()
    assert any(x['source_node']=='relu' and x['mapping']=='mapped' for x in d['events'])
    assert any(x['event_name']=='copy_kernel_time' and x['source_node'] is None for x in d['events'])
    assert output.to_dict()['evidence']['report']['optimized_base64']==r['optimized_base64']
    assert d['execution_fallback']['detected'] is None
    output.to_json(tmp_path/'result.json');output.to_csv(tmp_path/'result.csv')
    assert RuntimeResult.from_json(tmp_path/'result.json').to_dict()==output.to_dict()
    assert RuntimeResult.from_csv(tmp_path/'result.csv').to_dict()==output.to_dict()


@pytest.mark.parametrize('damage',['cycle','attribute','arity','duplicate_output'])
def test_malformed_transfer_graph_is_rejected(damage):
    from deployment_support import onnx_model
    raw,_=onnx_model();model=transfer_graph(base64.b64encode(raw).decode())
    node=model.graph.node[-1]
    if damage=='cycle':node.input[0]='Y'
    elif damage=='attribute':node.attribute.append(helper.make_attribute('unsupported',1))
    elif damage=='arity':node.input.append('X')
    else:model.graph.node.append(copy.deepcopy(node))
    encoded=base64.b64encode(model.SerializeToString()).decode()
    with pytest.raises(ValueError):module._decode_model(encoded,optimized=True)


def test_changed_nontransfer_definition_remains_unmapped(monkeypatch,tmp_path):
    result,_=simulated(monkeypatch,tmp_path,providers=['CPUExecutionProvider'],expected={'relu':'CPUExecutionProvider'},optimization='disabled')
    e=result.to_dict()['evidence'];e['graph_analysis_version']=2
    r=e['report'];graph=transfer_graph(r['model_base64'])
    graph.graph.node[0].op_type='Sigmoid';raw=graph.SerializeToString()
    r['optimized_base64']=base64.b64encode(raw).decode();r['optimized_sha256']=hashlib.sha256(raw).hexdigest()
    d=_result('P35',e).provider_diagnostics()
    assert d['source_node_mapping']['status']=='unknown'
    assert all(x['source_node'] is None for x in d['events'])


def test_legacy_custom_domain_transfer_does_not_gain_new_attribution(monkeypatch,tmp_path):
    result,_=simulated(monkeypatch,tmp_path,providers=['CPUExecutionProvider'],expected={'relu':'CPUExecutionProvider'},optimization='disabled')
    e=result.to_dict()['evidence'];r=e['report'];graph=transfer_graph(r['model_base64'])
    graph.graph.node[-1].domain='com.microsoft'
    graph.opset_import.append(helper.make_opsetid('com.microsoft',1))
    raw=graph.SerializeToString();r['optimized_base64']=base64.b64encode(raw).decode()
    r['optimized_sha256']=hashlib.sha256(raw).hexdigest()
    legacy=_result('P35',e)
    assert legacy.to_dict()['analysis']['status']=='unknown'
    legacy.to_json(tmp_path/'legacy.json');legacy.to_csv(tmp_path/'legacy.csv')
    assert RuntimeResult.from_json(tmp_path/'legacy.json').to_dict()==legacy.to_dict()
    assert RuntimeResult.from_csv(tmp_path/'legacy.csv').to_dict()==legacy.to_dict()
    updated=copy.deepcopy(e);updated['graph_analysis_version']=2
    assert _result('P35',updated).to_dict()['analysis']['status']=='expected_execution_observed'


@pytest.mark.parametrize('version',[True,0,3])
def test_graph_analysis_version_is_explicit_and_validated(monkeypatch,tmp_path,version):
    result,_=simulated(monkeypatch,tmp_path)
    e=result.to_dict()['evidence'];e['graph_analysis_version']=version
    with pytest.raises(ValueError):_result('P35',e)
