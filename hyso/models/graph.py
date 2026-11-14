"""Versioned HYSo DAG planning and execution using the closed Registry, root MIT."""
from __future__ import annotations
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import re
import torch
from torch import nn
from .registry import ModelRegistry,ModelConfigError,_plain
from .builder import _shape,_cost,_pairs
from . import advanced,components,sampling,graph_ops

MAX_NODES=64
MAX_EDGES=256
MAX_FAN_IN=8
MAX_FAN_OUT=16
MAX_BYTES=2*1024**2
ID_PATTERN=re.compile(r'[A-Za-z][A-Za-z0-9_]{0,47}\Z')


def _identifier(value):
    return type(value) is str and ID_PATTERN.fullmatch(value) is not None


def _error(message,**context):
    return ModelConfigError(message,code='GRAPH_STRUCTURE_ERROR',**context)


def _node_error(error,node_id):
    if error.layer_id is None:error.layer_id=node_id
    return error.to_dict()


def _metadata(shape,meaning='features'):
    return dict(shape=['N',*shape],rank=len(shape)+1,layout='NCHW' if len(shape)==3 else 'NC',
                feature_meaning=meaning,dtype='floating',device='model')


def _inputs(specs):
    if type(specs) is not dict or not 1<=len(specs)<=8:
        raise _error('graph requires 1..8 named inputs',expected='1..8 inputs',actual=specs)
    normalized={}
    for name,spec in sorted(specs.items()):
        if not _identifier(name):raise _error('invalid input ID',actual=name)
        if type(spec) is not dict or set(spec)!={'shape','layout','dtype','device'}:
            raise _error('input requires shape/layout/dtype/device',actual=spec)
        shape=spec['shape']
        if type(shape) is not list or len(shape) not in (1,3) or any(type(v) is not int or not 1<=v<=8192 for v in shape):
            raise _error('input shape must be bounded C or CHW without batch',actual=shape)
        expected='NCHW' if len(shape)==3 else 'NC'
        if spec['layout']!=expected or spec['dtype']!='floating' or spec['device']!='model':
            raise _error('input layout/dtype/device contract mismatch',expected=dict(layout=expected,dtype='floating',device='model'),actual=spec)
        normalized[name]=spec
    return normalized


def _structure(config,report):
    if type(config) is dict and type(config.get('nodes')) is list and len(config['nodes'])>MAX_NODES:
        raise ModelConfigError('graph node count exceeds 64',code='GRAPH_LIMIT_ERROR',expected=64,actual=len(config['nodes']))
    c=_plain(config)
    if (type(c) is not dict or set(c)!={'format','schema_version','kind','inputs','nodes','outputs'}
            or c['format']!='hyso.model_config' or type(c['schema_version']) is not int
            or c['schema_version']!=3 or c['kind']!='graph'):
        raise _error('graph requires format=hyso.model_config, schema_version=3, kind=graph and closed fields',actual=c)
    c['inputs']=_inputs(c['inputs'])
    if type(c['nodes']) is not list:raise _error('nodes must be a list')
    if type(c['outputs']) is not dict or not 1<=len(c['outputs'])<=8 or any(not _identifier(k) or type(v) is not str for k,v in c['outputs'].items()):
        raise _error('outputs require 1..8 safe names and reference strings',actual=c['outputs'])
    nodes={}
    for row in c['nodes']:
        if type(row) is not dict or set(row)!={'id','component','params','inputs'}:
            raise _error('node requires id/component/params/inputs',actual=row)
        node_id=row['id']
        if not _identifier(node_id):raise _error('invalid stable node ID',actual=node_id)
        if node_id in nodes or node_id in c['inputs']:
            report['diagnostics'].append(_node_error(_error('duplicate node/input ID',actual=node_id),node_id))
            continue
        if type(row['inputs']) is not list or not 1<=len(row['inputs'])<=MAX_FAN_IN or any(type(v) is not str for v in row['inputs']):
            raise _error('node inputs must contain 1..8 ordered reference strings',layer_id=node_id,actual=row['inputs'])
        nodes[node_id]=row
    references=set(nodes)|set(c['inputs'])
    edges=[dict(source=source,target=node_id,port=port) for node_id,row in sorted(nodes.items()) for port,source in enumerate(row['inputs'])]
    report['edges']=edges
    fanout=Counter(edge['source'] for edge in edges)
    if len(edges)>MAX_EDGES or any(v>MAX_FAN_OUT for v in fanout.values()):
        report['diagnostics'].append(ModelConfigError('graph edge/fan-out limit exceeded',code='GRAPH_LIMIT_ERROR',
            expected=dict(edges=MAX_EDGES,fan_out=MAX_FAN_OUT),actual=dict(edges=len(edges),fan_out=dict(fanout))).to_dict())
    for node_id,row in sorted(nodes.items()):
        for ref in row['inputs']:
            if ref not in references or ref==node_id:
                report['diagnostics'].append(_node_error(_error('missing reference or self-loop',expected='existing distinct input/node ID',actual=ref),node_id))
    for name,ref in c['outputs'].items():
        if ref not in references:
            report['diagnostics'].append(_error('invalid named output reference',layer_id=name,actual=ref).to_dict())
    dependencies={key:set(row['inputs'])&set(nodes) for key,row in nodes.items()}
    ready=[key for key,deps in dependencies.items() if not deps];heapq.heapify(ready)
    order=[]
    while ready:
        key=heapq.heappop(ready);order.append(key)
        for other in sorted(dependencies):
            if key in dependencies[other]:
                dependencies[other].remove(key)
                if not dependencies[other]:heapq.heappush(ready,other)
    if len(order)!=len(nodes):
        unresolved=set(nodes)-set(order)
        cycles=_cycles(nodes)
        members=set().union(*map(set,cycles))
        report['cycles']=cycles
        report['blocked_nodes']=sorted(unresolved-members)
        for cycle in cycles:
            report['diagnostics'].append(_error('cycle contains nodes '+', '.join(cycle),
                layer_id=cycle[0],actual=dict(cycle_members=cycle)).to_dict())
    reachable=set();pending=list(c['outputs'].values())
    while pending:
        key=pending.pop()
        if key in reachable:continue
        reachable.add(key)
        if key in nodes:pending.extend(nodes[key]['inputs'])
    for key in sorted(set(nodes)-reachable):
        report['diagnostics'].append(_error('disconnected node does not contribute to a named output',layer_id=key,actual=key).to_dict())
    report['topological_order']=order
    c['nodes']=[nodes[key] for key in sorted(nodes)]
    c['outputs']=dict(sorted(c['outputs'].items()))
    return c,nodes


def _cycles(nodes):
    """Deterministic Tarjan SCCs. Graph admission bounds recursion to 64 nodes."""
    indices={};low={};stack=[];active=set();cycles=[]
    def visit(key):
        indices[key]=low[key]=len(indices)
        stack.append(key);active.add(key)
        for ref in sorted(set(nodes[key]['inputs'])&set(nodes)):
            if ref not in indices:
                visit(ref);low[key]=min(low[key],low[ref])
            elif ref in active:
                low[key]=min(low[key],indices[ref])
        if low[key]==indices[key]:
            component=[]
            while True:
                member=stack.pop();active.remove(member);component.append(member)
                if member==key:break
            if len(component)>1 or key in nodes[key]['inputs']:
                cycles.append(sorted(component))
    for key in sorted(nodes):
        if key not in indices:visit(key)
    return sorted(cycles)


def _describe(name,p,sources,batch_size):
    if name in graph_ops.NAMES:
        output=graph_ops.fusion_shape(name,p,sources,ModelConfigError)
        count=len(p['weights']) if name=='weighted_sum' and p['learnable'] else 0
        return output,count,count,[output]*(2*len(sources)),[],None
    if len(sources)!=1:raise _error('unary component requires exactly one input',expected=1,actual=len(sources))
    shape=sources[0]
    if name in sampling.HEADS:
        info=sampling.head_geometry(name,p,shape,ModelConfigError)
        output=(info['output_features'],)
        return output,info['parameter_count'],info['parameter_count'],[(info['in_features'],),output],[],None
    if name=='identity':return shape,0,0,[shape],[],None
    if len(shape)==1 and name in {'relu','silu','hardswish','leaky_relu','prelu','gelu','mish','hardsigmoid','dropout'}:
        if name=='prelu' and p['num_parameters'] not in (1,shape[0]):
            raise ModelConfigError('PReLU slope count must be 1 or NC feature count',code='CHANNEL_MISMATCH',expected=[1,shape[0]],actual=p['num_parameters'])
        count=components.parameter_count(name,p)
        return shape,count,count,[shape],[],None
    if len(shape)!=3:
        raise ModelConfigError('component requires NCHW, received NC features',code='GRAPH_SHAPE_ERROR',expected='NCHW',actual=_metadata(shape))
    output=_shape(name,p,shape)
    estimate=_cost(name,p)
    if name in advanced.NEW_NAMES:
        d=advanced.describe(name,p,shape,batch_size,ModelConfigError)
        return output,d['parameter_count'],estimate,d['stages'],d['requirements'],d['attention']
    stages=components.stages(name,p,shape) if name in components.CONVOLUTIONS|components.POOLING else [output]
    exact_names=components.NEW_NAMES|{'conv2d','batch_norm','group_norm','relu','silu','hardswish','identity'}
    count=components.parameter_count(name,p) if name in exact_names else 0 if name in sampling.NEW_NAMES else None
    requirements=components.statistics_requirements(name,p,shape)
    return output,count,estimate,stages,requirements,None


def _analyze(config,batch_size=1,activation_limit=16_777_216):
    report=dict(valid=False,diagnostics=[],warnings=[],cycles=[],blocked_nodes=[],topological_order=[],nodes={},edges=[],inputs={},outputs={},
                parameter_count=None,parameter_estimate=None,activation_estimate=None,attention_cost={})
    try:
        if type(batch_size) is not int or batch_size<1 or type(activation_limit) is not int or activation_limit<1:
            raise _error('batch_size and activation_limit must be positive integers')
        c,nodes=_structure(config,report)
    except ModelConfigError as error:
        report['diagnostics'].append(error.to_dict());return report,None
    shapes={name:tuple(spec['shape']) for name,spec in c['inputs'].items()}
    report['inputs']={name:_metadata(shape) for name,shape in shapes.items()}
    consumers=Counter(edge['source'] for edge in report['edges']);consumers.update(c['outputs'].values())
    live={name:batch_size*math.prod(shape) for name,shape in shapes.items()}
    peak=sum(live.values());summed=peak
    exact=0;estimate=0
    registry=ModelRegistry()
    for key,row in sorted(nodes.items()):
        report['nodes'][key]=dict(id=key,component=row['component'],input_references=row['inputs'],input_shapes=[],
                                  output_shape=None,rank=None,layout=None,parameter_count=None,parameter_estimate=None,warnings=[],
                                  ports=dict(inputs=[dict(index=i,reference=r) for i,r in enumerate(row['inputs'])],output='tensor'))
        report['nodes'][key]['cycle_member']=any(key in cycle for cycle in report['cycles'])
        report['nodes'][key]['blocked_by_cycle']=key in report['blocked_nodes']
    for key in report['topological_order']:
        row=nodes[key];node=report['nodes'][key]
        try:
            if type(row['component']) is str and row['component'] in graph_ops.NAMES:
                params=graph_ops.parameters(row['component'],row['params'])
                node['component_metadata']=graph_ops.metadata(row['component'])
            else:
                spec=registry.get(row['component'])
                if spec.category=='model':raise _error('complete model entries are unsupported graph nodes')
                params=registry.validate(row['component'],row['params'])
                node['component_metadata']=spec.to_dict()
            row['params']=params
            if any(ref not in shapes for ref in row['inputs']):
                node['blocked_by']=[ref for ref in row['inputs'] if ref not in shapes]
                continue
            sources=[shapes[ref] for ref in row['inputs']]
            node['input_shapes']=[['N',*s] for s in sources]
            output,count,cost,stages,requirements,attention=_describe(row['component'],params,sources,batch_size)
            shapes[key]=output
            meta=_metadata(output, row['component'].removesuffix('_head') if row['component'] in sampling.HEADS else 'features')
            node.update(output_shape=meta['shape'],rank=meta['rank'],layout=meta['layout'],tensor=meta,
                        parameters=params,parameter_count=count,parameter_estimate=cost,
                        intermediate_shapes=[['N',*s] for s in stages[:-1]])
            estimate+=cost
            exact=exact+count if exact is not None and count is not None else None
            elements=[batch_size*math.prod(s) for s in stages]
            node['activation_elements']=elements;summed+=sum(elements)
            matrix=0
            if attention is not None:
                report['attention_cost'][key]=attention;matrix=3*attention['matrix_elements']
            peak=max(peak,sum(live.values())+sum(elements)+matrix)
            live[key]=batch_size*math.prod(output)
            for ref in row['inputs']:
                consumers[ref]-=1
                if consumers[ref]==0:live.pop(ref,None)
            for text in requirements:
                node['warnings'].append(ModelConfigError(text,code='STATISTICS_REQUIREMENT',layer_id=key,
                                        expected=text,actual=dict(batch='symbolic')).to_dict())
            if max(elements+[matrix])>activation_limit:
                node['warnings'].append(ModelConfigError('node advisory activation threshold exceeded',code='RESOURCE_LIMIT_WARNING',
                    layer_id=key,expected=activation_limit,actual=max(elements+[matrix])).to_dict())
            report['warnings'].extend(node['warnings'])
        except ModelConfigError as error:
            report['diagnostics'].append(_node_error(error,key))
    if estimate>10_000_000:
        report['diagnostics'].append(ModelConfigError('graph conservative parameter budget exceeds 10000000',
            code='PARAMETER_BUDGET_EXCEEDED',expected=10_000_000,actual=estimate).to_dict())
    complete=all(node['parameter_estimate'] is not None for node in report['nodes'].values())
    report['parameter_count']=exact if complete else None
    report['parameter_estimate']=estimate if complete else None
    report['validated_parameter_estimate']=estimate
    report['parameter_estimate_scope']='all_nodes' if complete else 'validated_nodes_only'
    report['activation_estimate']=dict(batch_size=batch_size,peak_live_elements=peak,summed_tensor_elements=summed,
        warning_threshold=activation_limit,scope='forward reference lifetimes plus all declared internal boundaries and 3 attention matrices',
        excludes='autograd retention, allocator/workspace and unknown legacy block internals',memory_guarantee=False)
    if peak>activation_limit:
        report['warnings'].append(ModelConfigError('graph peak-live estimate exceeds advisory threshold',code='RESOURCE_LIMIT_WARNING',
            expected=activation_limit,actual=peak).to_dict())
    for name,ref in c['outputs'].items():
        if ref in shapes:
            report['outputs'][name]=dict(reference=ref,**_metadata(shapes[ref],report['nodes'].get(ref,{}).get('tensor',{}).get('feature_meaning','features')))
    for key,node in report['nodes'].items():
        node['diagnostics']=[d for d in report['diagnostics'] if d['layer_id']==key]
        node['status']='error' if node['diagnostics'] else 'blocked' if node['output_shape'] is None else 'valid'
        node['ports']['output']=dict(name='tensor',shape=node['output_shape'],layout=node['layout'])
    for edge in report['edges']:
        edge['tensor']=(report['inputs'].get(edge['source']) or report['nodes'].get(edge['source'],{}).get('tensor'))
    report['valid']=not report['diagnostics']
    return report,c


class GraphModel(nn.Module):
    """Execute each registered node once and return sorted named tensor outputs."""
    def __init__(self,config,report):
        super().__init__()
        self._architecture=json.dumps(config,sort_keys=True,allow_nan=False)
        self._report=report
        registry=ModelRegistry()
        by_id={row['id']:row for row in config['nodes']}
        self.nodes=nn.ModuleDict({'node_'+key:(graph_ops.CONSTRUCTORS[by_id[key]['component']](**by_id[key]['params'])
                                 if by_id[key]['component'] in graph_ops.NAMES else registry.create(by_id[key]['component'],**by_id[key]['params']))
                                 for key in report['topological_order']})
        self.register_buffer('_contract',torch.empty(0),persistent=False)

    def to_config(self):return json.loads(self._architecture)

    def forward(self,inputs):
        c=self.to_config()
        if isinstance(inputs,torch.Tensor) and len(c['inputs'])==1:inputs={next(iter(c['inputs'])):inputs}
        if type(inputs) is not dict or set(inputs)!=set(c['inputs']):
            raise _error('graph input names mismatch',expected=sorted(c['inputs']),actual=list(inputs) if type(inputs) is dict else type(inputs).__name__)
        values={};batch=None
        for key,spec in c['inputs'].items():
            value=inputs[key]
            if (not isinstance(value,torch.Tensor) or value.ndim!=len(spec['shape'])+1 or min(value.shape)<1
                    or list(value.shape[1:])!=spec['shape'] or value.dtype!=self._contract.dtype or value.device!=self._contract.device):
                raise ModelConfigError('graph input shape/device/dtype mismatch',code='INVALID_INPUT_SHAPE',layer_id=key,
                    expected=dict(shape=['N',*spec['shape']],dtype=str(self._contract.dtype),device=str(self._contract.device)),
                    actual=dict(shape=list(value.shape),dtype=str(value.dtype),device=str(value.device)) if isinstance(value,torch.Tensor) else type(value).__name__)
            if batch is not None and batch!=value.shape[0]:raise _error('all graph inputs must share batch N',layer_id=key)
            batch=value.shape[0];values[key]=value
        consumers=Counter(edge['source'] for edge in self._report['edges']);consumers.update(c['outputs'].values())
        by_id={row['id']:row for row in c['nodes']}
        for key in self._report['topological_order']:
            row=by_id[key];args=[values[ref] for ref in row['inputs']]
            try:
                output=self.nodes['node_'+key](*args)
                expected=self._report['nodes'][key]['output_shape'][1:]
                if (not isinstance(output,torch.Tensor) or list(output.shape)!=[batch,*expected]
                        or output.dtype!=self._contract.dtype or output.device!=self._contract.device):
                    raise ModelConfigError('node output contract mismatch',code='GRAPH_SHAPE_ERROR',expected=[batch,*expected],
                                           actual=list(output.shape) if isinstance(output,torch.Tensor) else type(output).__name__)
            except (ValueError,RuntimeError) as exc:
                if isinstance(exc,ModelConfigError):
                    exc.layer_id=key
                    raise
                error=ModelConfigError(str(exc),code='GRAPH_SHAPE_ERROR')
                error.layer_id=key
                raise error from exc
            values[key]=output
            for ref in row['inputs']:
                consumers[ref]-=1
                if consumers[ref]==0:values.pop(ref,None)
        return {name:values[ref] for name,ref in c['outputs'].items()}


class GraphModelBuilder:
    """Closed schema3 DAG builder with allocation-free planning and preview."""
    def __init__(self,inputs):
        specs={}
        if type(inputs) is not dict:raise _error('inputs must be a named dictionary')
        for key,value in inputs.items():
            specs[key]=(dict(shape=list(value),layout='NCHW' if len(value)==3 else 'NC',dtype='floating',device='model')
                        if type(value) in (list,tuple) else value)
        self._config=dict(format='hyso.model_config',schema_version=3,kind='graph',inputs=_inputs(_plain(specs)),nodes=[],outputs={})

    def add(self,node_id,component,inputs,**params):
        if not _identifier(node_id) or node_id in self._config['inputs'] or any(row['id']==node_id for row in self._config['nodes']):
            raise _error('invalid or duplicate node ID',layer_id=node_id if type(node_id) is str else None,actual=node_id)
        if len(self._config['nodes'])>=MAX_NODES:raise ModelConfigError('node limit exceeded',code='GRAPH_LIMIT_ERROR')
        normalized=graph_ops.parameters(component,params) if type(component) is str and component in graph_ops.NAMES else ModelRegistry().validate(component,params)
        row=dict(id=node_id,component=component,inputs=_plain(inputs),params=normalized)
        if type(row['inputs']) is not list or not 1<=len(row['inputs'])<=8 or any(type(v) is not str for v in row['inputs']):
            raise _error('node inputs require 1..8 ordered references',layer_id=node_id)
        self._config['nodes'].append(row)
        return self

    def outputs(self,**references):
        if not 1<=len(references)<=8 or any(not _identifier(k) or type(v) is not str for k,v in references.items()):
            raise _error('named outputs require safe names and reference strings')
        self._config['outputs']=dict(sorted(references.items()));return self

    def to_config(self):
        c=_plain(self._config);c['nodes']=sorted(c['nodes'],key=lambda row:row['id']);return c

    @classmethod
    def from_config(cls,config):
        report,c=_analyze(config)
        if not report['valid']:
            diagnostic=report['diagnostics'][0]
            raise ModelConfigError(diagnostic['message_en'],code=diagnostic['code'],layer_id=diagnostic['layer_id'],
                expected=diagnostic['expected'],actual=diagnostic['actual'],suggestion=diagnostic['suggestion'])
        result=cls.__new__(cls);result._config=c;return result

    @staticmethod
    def preflight(config,*,batch_size=1,activation_limit=16_777_216):
        return _analyze(config,batch_size,activation_limit)[0]

    def preview(self,**options):return self.preflight(self.to_config(),**options)

    def build(self):
        config=self.from_config(self.to_config()).to_config();report=self.preflight(config)
        with torch.device('cpu'):return GraphModel(config,report)

    def to_json(self,path,*,overwrite=False):
        if type(overwrite) is not bool:raise _error('overwrite must be bool')
        raw=json.dumps(self.to_config(),indent=2,allow_nan=False)+'\n'
        if len(raw.encode('utf-8'))>MAX_BYTES:raise _error('graph JSON exceeds 2 MiB')
        with Path(path).open('w' if overwrite else 'x',encoding='utf-8',newline='\n') as stream:stream.write(raw)

    @classmethod
    def from_json(cls,path):
        with Path(path).open('rb') as stream:raw=stream.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise _error('graph JSON exceeds 2 MiB')
        def invalid(value):raise _error('non-finite JSON number',actual=value)
        try:return cls.from_config(json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=invalid))
        except ModelConfigError:raise
        except (ValueError,TypeError,RecursionError) as exc:raise _error(str(exc)) from exc
