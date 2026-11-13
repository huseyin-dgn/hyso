"""Graph contracts written before the implementation, bounded deterministic CPU."""
import copy
import json
import pytest
import torch
from torch import nn
from hyso import GraphModelBuilder,ModelConfigError
from hyso.cli import main


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    torch.manual_seed(1355)
    def forbidden(*a,**kw):pytest.fail('graph attempted CUDA work')
    for name in ('_lazy_init','is_available','device_count','synchronize'):
        monkeypatch.setattr(torch.cuda,name,forbidden)


def graph(fusion='add',**params):
    return (GraphModelBuilder({'image':[4,5,7]})
        .add('left','conv2d',['image'],in_channels=4,out_channels=4)
        .add('right','pointwise_conv2d',['image'],in_channels=4,out_channels=4)
        .add('merge',fusion,['left','right'],**params)
        .outputs(features='merge'))


@pytest.mark.parametrize('fusion,params',[('add',{}),('multiply',{}),('concat',dict(axis=1)),
    ('weighted_sum',dict(weights=[.25,.75],learnable=False)),('weighted_sum',dict(weights=[1.,-1.],learnable=True))])
@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
def test_branches_fusion_gradients_state_json(fusion,params,dtype,tmp_path):
    b=graph(fusion,**params);report=b.preflight(b.to_config());assert report['valid'],report['diagnostics']
    path=tmp_path/'graph.json';b.to_json(path);clone_b=GraphModelBuilder.from_json(path)
    assert clone_b.to_config()==b.to_config()
    m=b.build().to(dtype);optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    for training in (True,False):
        m.train(training);optimizer.zero_grad(set_to_none=True)
        x=torch.randn(2,4,5,7,dtype=dtype,requires_grad=True);out=m(x)['features']
        assert list(out.shape)==[2,*report['outputs']['features']['shape'][1:]]
        out.square().mean().backward();assert torch.isfinite(x.grad).all()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
        optimizer.step()
    clone=clone_b.build().to(dtype);clone.load_state_dict(copy.deepcopy(m.state_dict()),strict=True)
    m.eval();clone.eval();assert torch.equal(m(x)['features'],clone(x)['features'])


def test_multioutput_three_branches_flatten_and_explicit_projection():
    b=(graph().add('skip','add',['image','merge'])
        .add('pool','global_avg_pool2d',['skip']).add('flat','flatten',['pool'])
        .add('classification','classification_head',['flat'],in_features=4,num_classes=3,pooling='none')
        .add('regression','regression_head',['flat'],in_features=4,output_features=2,pooling='none')
        .add('embedding','embedding_head',['flat'],in_features=4,embedding_dim=5,pooling='none',normalize=True)
        .outputs(classification='classification',regression='regression',embedding='embedding',features='merge'))
    r=b.preflight(b.to_config());assert r['valid'],r['diagnostics']
    m=b.build();calls={}
    hooks=[module.register_forward_hook(lambda mod,args,out,key=key:calls.__setitem__(key,calls.get(key,0)+1))
           for key,module in m.nodes.items()]
    outputs=m(torch.randn(2,4,5,7));assert list(outputs)==sorted(outputs)
    assert outputs['classification'].shape==(2,3) and outputs['regression'].shape==(2,2) and outputs['embedding'].shape==(2,5)
    optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    loss=sum((value-torch.randn_like(value)).square().mean() for value in outputs.values())
    loss.backward();optimizer.step()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    assert set(calls.values())=={1}
    for h in hooks:h.remove()
    projection=(GraphModelBuilder({'x':[4,5,7]}).add('conv','conv2d',['x'],in_channels=4,out_channels=8,stride=2)
        .add('project','pointwise_conv2d',['x'],in_channels=4,out_channels=8,stride=2)
        .add('sum','add',['conv','project']).outputs(result='sum'))
    assert projection.build()(torch.zeros(2,4,5,7))['result'].shape==(2,8,3,4)


def test_physical_order_stable_ids_and_allocation_fence(monkeypatch):
    b=graph();config=b.to_config();config['nodes'].reverse()
    other=GraphModelBuilder.from_config(config)
    assert b.preflight(b.to_config())['topological_order']==other.preflight(config)['topological_order']
    assert list(b.build().state_dict())==list(other.build().state_dict())
    def forbidden(*a,**kw):pytest.fail('preview allocated weights')
    monkeypatch.setattr(nn.Module,'__init__',forbidden);monkeypatch.setattr(torch,'empty',forbidden)
    r=other.preview();assert r['valid'] and r['edges'] and r['nodes'] and r['activation_estimate']['peak_live_elements']>0
    assert r['nodes']['merge']['input_references']==['left','right']


@pytest.mark.parametrize('mutation',[
    'duplicate','missing','self_loop','cycle','disconnected','output','wrong_count','component','rank','shape','axis','weights','id',
])
def test_structural_and_shape_failures(mutation):
    c=graph().to_config()
    rows={row['id']:row for row in c['nodes']}
    if mutation=='duplicate':c['nodes'].append(copy.deepcopy(c['nodes'][0]))
    elif mutation=='missing':c['nodes'][0]['inputs']=['absent']
    elif mutation=='self_loop':c['nodes'][0]['inputs']=['left']
    elif mutation=='cycle':c['nodes'][0]['inputs']=['merge']
    elif mutation=='disconnected':c['nodes'].append(dict(id='unused',component='identity',params={},inputs=['image']))
    elif mutation=='output':c['outputs']={'out':'absent'}
    elif mutation=='wrong_count':c['nodes'][0]['inputs']=['image','image']
    elif mutation=='component':c['nodes'][0]['component']='os.system'
    elif mutation=='rank':c['nodes'][0].update(component='flatten',params={})
    elif mutation=='shape':c['nodes'][0]['params']['out_channels']=8
    elif mutation=='axis':rows['merge'].update(component='concat',params=dict(axis=0))
    elif mutation=='weights':rows['merge'].update(component='weighted_sum',params=dict(weights=[1.],learnable=True))
    elif mutation=='id':c['nodes'][0]['id']='bad.id'
    r=GraphModelBuilder.preflight(c);assert not r['valid'] and r['diagnostics']
    assert all(d['message_tr'] and d['message_en'] and d['suggestion'] for d in r['diagnostics'])
    with pytest.raises(ModelConfigError):GraphModelBuilder.from_config(c)


def test_independent_errors_and_no_downstream_invention():
    c=graph().to_config();rows={row['id']:row for row in c['nodes']}
    rows['left']['params']['in_channels']=3;rows['right']['params']['out_channels']=0
    r=GraphModelBuilder.preflight(c)
    assert not r['valid'] and {'left','right'} <= {d['layer_id'] for d in r['diagnostics']}
    assert r['nodes']['merge']['output_shape'] is None
    assert r['parameter_count'] is None and r['parameter_estimate'] is None
    assert r['parameter_estimate_scope']=='validated_nodes_only'


def test_nc_inputs_and_runtime_guards():
    b=(GraphModelBuilder({'a':[4],'b':[4]}).add('sum','add',['a','b'])
       .add('head','regression_head',['sum'],in_features=4,output_features=2,pooling='none').outputs(out='head'))
    m=b.build();assert m({'a':torch.ones(2,4),'b':torch.zeros(2,4)})['out'].shape==(2,2)
    for bad in ({'a':torch.ones(2,4)}, {'a':torch.ones(2,4),'b':torch.ones(1,4)},
                {'a':torch.ones(2,4),'b':torch.ones(2,4,dtype=torch.float64)}):
        with pytest.raises(ModelConfigError):m(bad)
    with pytest.raises(ModelConfigError):m({'a':torch.ones(2,3),'b':torch.ones(2,4)})


def test_sampling_fusion_attention_and_resource_limits():
    b=(GraphModelBuilder({'x':[4,4,4]}).add('resize','resize',['x'],output_size=[8,8])
        .add('other','resize',['x'],scale_factor=2).add('sum','add',['resize','other'])
        .add('attn','multihead_spatial_attention',['sum'],channels=4,num_heads=2).outputs(out='attn'))
    assert b.preflight(b.to_config())['attention_cost']['attn']['matrix_elements']==8192
    assert b.build()(torch.zeros(2,4,4,4))['out'].shape==(2,4,8,8)
    assert not GraphModelBuilder.preflight(b.to_config(),batch_size=513)['valid']
    with pytest.raises(ModelConfigError) as error:b.build()(torch.zeros(513,4,4,4))
    assert error.value.code=='ATTENTION_RESOURCE_LIMIT' and error.value.layer_id=='attn'
    c=b.to_config();c['inputs']['x']['shape']=[4,64,64];rows={row['id']:row for row in c['nodes']}
    rows['resize']['params']['output_size']=[64,64];rows['other']['params']['scale_factor']=1
    assert 'ATTENTION_RESOURCE_LIMIT' in [d['code'] for d in GraphModelBuilder.preflight(c)['diagnostics']]
    c=graph().to_config();c['nodes']=c['nodes']*22
    assert not GraphModelBuilder.preflight(c)['valid']
    c=graph().to_config();c['nodes'][0]['params'].update(in_channels=8192,out_channels=8192);c['inputs']['image']['shape'][0]=8192
    assert not GraphModelBuilder.preflight(c)['valid']


def test_cli_json_preview_and_duplicate_keys(tmp_path,capsys):
    b=graph();path=tmp_path/'graph.json';b.to_json(path)
    for action in ('validate','summary','preview'):
        assert main(['models',action,str(path),'--json'])==0
        assert json.loads(capsys.readouterr().out)['result']['validation']['valid']
    path.write_text('{"format":"x","format":"y"}')
    assert main(['models','preview',str(path),'--json'])==2
    assert json.loads(capsys.readouterr().out)['diagnostics']


@pytest.mark.parametrize('axis',[1,2,3])
def test_concat_axis_numeric_parity(axis):
    b=(GraphModelBuilder({'x':[4,3,5]}).add('join','concat',['x','x','x'],axis=axis).outputs(out='join'))
    x=torch.randn(2,4,3,5,requires_grad=True)
    result=b.build()(x)['out'];assert torch.equal(result,torch.cat((x,x,x),dim=axis))
    result.sum().backward();assert torch.equal(x.grad,torch.full_like(x,3.))


def test_shared_cost_lifetimes_weights_and_parameterless_graph():
    b=(GraphModelBuilder({'x':[4,3,5]}).add('conv','pointwise_conv2d',['x'],in_channels=4,out_channels=4)
       .add('sum','weighted_sum',['conv','conv','conv'],weights=[1.,2.,-1.],learnable=True).outputs(out='sum',shared='conv'))
    r=b.preview();assert r['parameter_count']==23 and r['parameter_estimate']>=23
    assert r['activation_estimate']['peak_live_elements']>=120
    assert all(edge['tensor']['layout']=='NCHW' for edge in r['edges'])
    m=b.build();x=torch.ones(2,4,3,5);out=m(x)
    torch.testing.assert_close(out['out'],2*out['shared'])
    assert 'nodes.node_sum.weights' in m.state_dict()
    fixed=GraphModelBuilder({'x':[4]}).add('sum','weighted_sum',['x','x'],weights=[.25,.75]).outputs(out='sum').build()
    assert sum(p.numel() for p in fixed.parameters())==0 and 'nodes.node_sum.weights' in fixed.state_dict()
    assert torch.equal(fixed(torch.ones(2,4))['out'],torch.ones(2,4))
    direct=GraphModelBuilder({'x':[4]}).outputs(out='x')
    assert direct.preview()['parameter_count']==0
    assert torch.equal(direct.build().double()(torch.ones(2,4,dtype=torch.float64))['out'],torch.ones(2,4,dtype=torch.float64))


@pytest.mark.parametrize('mutation',['fanin','fanout','edges','budget','schema','layout','input_rank','head','fusion_rank'])
def test_precise_graph_limits_and_contracts(mutation):
    c=graph().to_config();rows={row['id']:row for row in c['nodes']}
    if mutation=='fanin':rows['merge']['inputs']=['image']*9
    elif mutation=='fanout':
        c['nodes']=[dict(id='n'+str(i),component='identity',params={},inputs=['image']) for i in range(17)];c['outputs']={'out':'n0'}
    elif mutation=='edges':
        c['nodes']=[dict(id='n'+str(i),component='add',params={},inputs=['image']*8) for i in range(33)];c['outputs']={'out':'n0'}
    elif mutation=='budget':
        c['inputs']['image']['shape'][0]=2048
        rows['left']['params'].update(in_channels=2048,out_channels=2048)
        rows['right']['params'].update(in_channels=2048,out_channels=2048)
    elif mutation=='schema':c['schema_version']=2
    elif mutation=='layout':c['inputs']['image']['layout']='NHWC'
    elif mutation=='input_rank':c['inputs']['image']['shape']=[4,3]
    elif mutation=='head':rows['left'].update(component='classification_head',params=dict(in_features=3,num_classes=2))
    elif mutation=='fusion_rank':rows['left'].update(component='flatten',params={})
    r=GraphModelBuilder.preflight(c);assert not r['valid']
    if mutation in ('fanout','edges'):assert 'GRAPH_LIMIT_ERROR' in [d['code'] for d in r['diagnostics']]
    if mutation=='budget':assert 'PARAMETER_BUDGET_EXCEEDED' in [d['code'] for d in r['diagnostics']]


def test_safety_boundaries_runtime_node_checks_and_json(tmp_path,monkeypatch):
    b=GraphModelBuilder({'x':[4]})
    ref='x'
    for i in range(64):
        node='n'+str(i);b.add(node,'identity',[ref]);ref=node
    b.outputs(out=ref);assert b.preview()['valid']
    with pytest.raises(ModelConfigError):b.add('extra','identity',[ref])
    path=tmp_path/'oversized.json';path.write_bytes(b' '*(2*1024**2+1))
    with pytest.raises(ModelConfigError):GraphModelBuilder.from_json(path)
    c=graph().to_config();c['nodes'][0]['params']['weights']='os.system'
    assert not GraphModelBuilder.preflight(c)['valid']
    m=graph().build()
    monkeypatch.setattr(m.nodes['node_left'],'forward',lambda x:torch.zeros(2,1,1,1))
    with pytest.raises(ModelConfigError) as err:m(torch.ones(2,4,5,7))
    assert err.value.layer_id=='left'
    assert not GraphModelBuilder.preflight(graph().to_config(),batch_size=0)['valid']
    with pytest.raises(ModelConfigError):GraphModelBuilder.from_config({})


def test_nc_activation_and_flatten_head_paths():
    b=(GraphModelBuilder({'x':[4,3,5]}).add('flat','flatten',['x'])
       .add('activation','prelu',['flat'],num_parameters=60)
       .add('head','embedding_head',['activation'],in_features=60,embedding_dim=3,pooling='none').outputs(out='head'))
    r=b.preview();assert r['valid'] and r['nodes']['flat']['layout']=='NC'
    assert b.build()(torch.ones(2,4,3,5))['out'].shape==(2,3)


def test_single_branch_classification_cross_entropy_update():
    b=(GraphModelBuilder({'image':[3,5,7]})
       .add('features','conv2d',['image'],in_channels=3,out_channels=4)
       .add('prediction','classification_head',['features'],in_features=4,num_classes=3)
       .outputs(classification='prediction'))
    m=b.build();optimizer=torch.optim.SGD(m.parameters(),lr=.01)
    x=torch.randn(2,3,5,7,requires_grad=True)
    loss=nn.CrossEntropyLoss()(m(x)['classification'],torch.tensor([0,1]))
    loss.backward();optimizer.step()
    assert torch.isfinite(x.grad).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
