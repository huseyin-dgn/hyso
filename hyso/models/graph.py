"""Versioned HYSo DAG planning and execution using the closed Registry, root MIT."""
from __future__ import annotations
import typing as _typing
from collections import Counter
import heapq
import json
import math
from pathlib import Path
import re
import torch
from torch import nn
from .registry import ModelRegistry,ModelConfigError,_plain
from ._profile import DEVELOPMENT
from .builder import _shape,_cost,_pairs
from . import advanced,components,sampling,graph_ops

MAX_NODES=64
MAX_EDGES=256
MAX_FAN_IN=8
MAX_FAN_OUT=16
MAX_BYTES=2*1024**2
ID_PATTERN=re.compile(r'[A-Za-z][A-Za-z0-9_]{0,47}\Z')


class _GraphInputSpec(_typing.TypedDict):
    shape: list[int]
    layout: _typing.Literal['NC', 'NCHW']
    dtype: _typing.Literal['floating']
    device: _typing.Literal['model']


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
    """Execute each registered node once and return sorted named tensor outputs.

    Args:
        config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained
            state. Required.
        report (_typing.Any): Validated retained analysis/advisory report. Required.

    Returns:
        An instance of GraphModel with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        GraphModel(config, report)

    HYSo Guide: models.graph-builder
    Offline: hyso docs show models.graph-builder --lang en
    Open: hyso docs open models.graph-builder --lang en
    """
    def __init__(self,config: 'dict[str, _typing.Any]',report: '_typing.Any') -> 'None':
        """Initialize GraphModel with the supplied settings.

        Args:
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained
                state. Required.
            report (_typing.Any): Validated retained analysis/advisory report. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            GraphModel(config, report)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        super().__init__()
        self._architecture=json.dumps(config,sort_keys=True,allow_nan=False)
        self._report=report
        registry=ModelRegistry()
        by_id={row['id']:row for row in config['nodes']}
        self.nodes=nn.ModuleDict({'node_'+key:(graph_ops.CONSTRUCTORS[by_id[key]['component']](**by_id[key]['params'])
                                 if by_id[key]['component'] in graph_ops.NAMES else registry.create(by_id[key]['component'],**by_id[key]['params']))
                                 for key in report['topological_order']})
        self.register_buffer('_contract',torch.empty(0),persistent=False)

    def to_config(self) -> 'dict[str, _typing.Any]':
        """Copy the versioned architecture configuration without model weights.

        Returns:
            Detached JSON-compatible architecture metadata including normalized defaults; no weights.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_config()

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        return json.loads(self._architecture)

    def forward(self,inputs: torch.Tensor | dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            inputs: Exact named tensor mapping; a single tensor is accepted only
                for a graph with one input. Shapes exclude the shared batch N and
                must match the configured per-sample shape, model dtype and device.

        Returns:
            Named output tensors following outputs(); this call executes the DAG.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.forward(inputs)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
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
                target=graph_ops.active_autocast_dtype(self._contract.device)
                allowed=(self._contract.dtype,)
                if self._contract.dtype==torch.float32 and target is not None:allowed+=(target,)
                if (not isinstance(output,torch.Tensor) or list(output.shape)!=[batch,*expected]
                        or output.dtype not in allowed or output.device!=self._contract.device):
                    raise ModelConfigError('node output contract mismatch',code='GRAPH_SHAPE_ERROR',
                        expected=dict(shape=[batch,*expected],dtypes=[str(d) for d in allowed],device=str(self._contract.device)),
                        actual=dict(shape=list(output.shape),dtype=str(output.dtype),device=str(output.device))
                        if isinstance(output,torch.Tensor) else type(output).__name__)
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
    """Closed schema3 DAG builder with allocation-free planning and preview.

    Args:
        inputs: Mapping of 1..8 safe names to per-sample C or CHW sequences, or dictionaries with
            exactly shape/layout/dtype/device. Dimensions are integers in 1..8192; layout is NC for C or
            NCHW for CHW, dtype is floating and device is model. Required.

    Returns:
        An instance of GraphModelBuilder with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        ModelConfigError: Input names, shapes, layout, dtype or device metadata violate the closed graph
            contract.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        builder = GraphModelBuilder({"image": [1, 4, 4]})

    HYSo Guide: models.graph-builder
    Offline: hyso docs show models.graph-builder --lang en
    Open: hyso docs open models.graph-builder --lang en
    """
    def __init__(self,inputs: dict[str, tuple[int, ...] | list[int] | _GraphInputSpec]) -> None:
        """Initialize GraphModelBuilder with the supplied settings.

        Args:
            inputs: Mapping of 1..8 safe names to per-sample C or CHW sequences, or dictionaries with
                exactly shape/layout/dtype/device. Dimensions are integers in 1..8192; layout is NC for C or
                NCHW for CHW, dtype is floating and device is model. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            GraphModelBuilder(inputs)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        specs={}
        if type(inputs) is not dict:raise _error('inputs must be a named dictionary')
        for key,value in inputs.items():
            specs[key]=(dict(shape=list(value),layout='NCHW' if len(value)==3 else 'NC',dtype='floating',device='model')
                        if type(value) in (list,tuple) else value)
        self._config=dict(format='hyso.model_config',schema_version=3,kind='graph',inputs=_inputs(_plain(specs)),nodes=[],outputs={})

    def _inference_shape(self, node_id: str, inputs: list[str], parameter: str) -> tuple[int, ...]:
        """Analyze only the selected source's ancestors, without building modules."""
        if type(inputs) is not list or len(inputs) != 1 or type(inputs[0]) is not str:
            raise ModelConfigError(f'Cannot infer {parameter}: exactly one explicit source reference is required',
                                   code='SHAPE_INFERENCE_UNAVAILABLE', layer_id=node_id,
                                   expected='one named source', actual=inputs,
                                   suggestion=f'Keep node connections explicit and supply {parameter} when source shape is unavailable.')
        candidate = self.to_config()
        rows = {row['id']: row for row in candidate['nodes']}
        needed = set()
        pending = list(inputs)
        while pending:
            reference = pending.pop()
            if reference in needed:
                continue
            needed.add(reference)
            if reference in rows:
                pending.extend(rows[reference]['inputs'])
        candidate['nodes'] = [row for row in candidate['nodes'] if row['id'] in needed]
        candidate['outputs'] = {'inference_source': inputs[0]}
        report, _ = _analyze(candidate)
        if not report['valid']:
            detail = report['diagnostics'][0]['message_en'] if report['diagnostics'] else 'source shape is unavailable'
            raise ModelConfigError(f'Cannot infer {parameter} from {inputs[0]!r}: {detail}',
                                   code='SHAPE_INFERENCE_UNAVAILABLE', layer_id=node_id,
                                   expected='a validated source shape', actual=inputs[0],
                                   suggestion=f'Resolve source references/shapes or supply an explicit {parameter}; native graph validation still applies.')
        source = inputs[0]
        shape = (report['inputs'][source]['shape'] if source in report['inputs']
                 else report['nodes'][source]['output_shape'])
        return tuple(shape[1:])

    def _add_with_inference(self, node_id: str, component: str, inputs: list[str],
                            parameter: str, value: int | None, params: dict[str, _typing.Any]) -> GraphModelBuilder:
        """Resolve one omitted size, then retain native scalar and shape checks."""
        registry = ModelRegistry()
        if value is not None:
            # Explicit values are never corrected, including bool/zero/string inputs.
            params = registry.validate(component, dict(params, **{parameter: value}))
        try:
            shape = self._inference_shape(node_id, inputs, parameter)
        except ModelConfigError:
            if value is None:
                raise
            # Preserve the existing explicit forward-reference/deferred-validation API.
            return self.add(node_id, component, inputs, **params)
        if value is None:
            if component == 'classification_head':
                value = math.prod(shape) if params['pooling'] == 'none' else shape[0]
            elif len(shape) == 3:
                value = shape[0]
            else:
                raise ModelConfigError(f'Cannot infer {parameter}: source is not NCHW; provide an explicit value and a compatible shape',
                                       code='SHAPE_INFERENCE_UNAVAILABLE', layer_id=node_id,
                                       expected='known CHW features', actual=list(shape),
                                       suggestion=f'Restore a compatible NCHW source or supply {parameter} explicitly.')
            params = registry.validate(component, dict(params, **{parameter: value}))
        try:
            _describe(component, params, [shape], 1)
        except ModelConfigError as error:
            diagnostic = _node_error(error, node_id)
            raise ModelConfigError(diagnostic['message_en'], code=diagnostic['code'], layer_id=node_id,
                                   expected=diagnostic['expected'], actual=diagnostic['actual'],
                                   suggestion=diagnostic['suggestion']) from error
        return self.add(node_id, component, inputs, **params)

    def conv2d(
        self, node_id: str, inputs: list[str],
        *,
        in_channels: int | None = None,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        groups: int = 1,
        bias: bool = True,
        dilation: int | tuple[int, int] | list[int] = 1,
    ) -> GraphModelBuilder:
        """Append a registered 2D convolution.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int | None): Input feature channels; must match the incoming tensor C dimension.
                Integer in 1..8192; bool is rejected. None infers C from the current validated feature
                shape; ambiguous/non-NCHW shapes are rejected. Default: None.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups divides in_channels and out_channels
            dilation changes the effective kernel
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.conv2d("node", ["image"], in_channels=1, out_channels=4)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self._add_with_inference(node_id, 'conv2d', inputs, 'in_channels', in_channels, dict(out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=bias, dilation=dilation))

    def conv_norm_act(
        self, node_id: str, inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        groups: int = 1,
        norm: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['silu', 'relu', None] = 'silu',
    ) -> GraphModelBuilder:
        """Append convolution, optional normalization and activation.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.
            norm (_typing.Literal['batch', 'group', None]): Optional post-convolution normalization: batch,
                group or None. Group normalization chooses the largest channel divisor <=8; convolution bias
                is enabled only without normalization. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['silu', 'relu', None]): Nonlinearity applied inside this block; only
                the declared registry choices are accepted. Choices: 'silu', 'relu', None. Default: 'silu'.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups divides both channel counts
            normalization determines convolution bias
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.conv_norm_act("node", ["image"], in_channels=1, out_channels=4)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(
            node_id,
            'conv_norm_act',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            norm=norm,
            activation=activation,
        )

    def relu(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append out-of-place ReLU activation.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.relu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'relu', inputs)

    def silu(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append out-of-place SiLU activation.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.silu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'silu', inputs)

    def gelu(
        self, node_id: str, inputs: list[str],
        *,
        approximate: _typing.Literal['none', 'tanh'] = 'none',
    ) -> GraphModelBuilder:
        """Append GELU activation with the selected approximation.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            approximate (_typing.Literal['none', 'tanh']): GELU calculation: none uses the exact error
                function; tanh uses the approximation. Choices: 'none', 'tanh'. Default: 'none'.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.gelu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'gelu', inputs, approximate=approximate)

    def batch_norm(
        self, node_id: str, inputs: list[str],
        *,
        num_features: int | None = None,
        eps: float = 1e-05,
        momentum: float | None = 0.1,
        affine: bool = True,
        track_running_stats: bool = True,
    ) -> GraphModelBuilder:
        """Append registered 2D batch normalization.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            num_features (int | None): Normalization channel count; must match input C. Integer in 1..8192;
                bool is rejected. Default: None.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            momentum (float | None): Running-statistic update factor; None selects cumulative averaging for
                BatchNorm. None or number in [0,1]. Default: 0.1.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: True.
            track_running_stats (bool): Track running mean/variance for evaluation. Without tracking,
                input-statistic sample requirements also apply in eval. Boolean only. Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.batch_norm("node", ["image"], num_features=4)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self._add_with_inference(node_id, 'batch_norm', inputs, 'num_features', num_features, dict(eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats))

    def group_norm(
        self, node_id: str, inputs: list[str],
        *,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-05,
        affine: bool = True,
    ) -> GraphModelBuilder:
        """Append normalization across explicit channel groups.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            num_groups (int): GroupNorm channel partitions; must divide num_channels. Integer in 1..8192;
                bool is rejected. Required.
            num_channels (int): Total GroupNorm channels C; must be divisible by num_groups. Integer in
                1..8192; bool is rejected. Required.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            num_groups divides num_channels
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.group_norm("node", ["image"], num_groups=1, num_channels=4)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(
            node_id,
            'group_norm',
            inputs,
            num_groups=num_groups,
            num_channels=num_channels,
            eps=eps,
            affine=affine,
        )

    def max_pool2d(
        self, node_id: str, inputs: list[str],
        *,
        kernel_size: int | tuple[int, int] | list[int],
        stride: int | tuple[int, int] | list[int] | None = None,
        padding: int | tuple[int, int] | list[int] = 0,
        dilation: int | tuple[int, int] | list[int] = 1,
        ceil_mode: bool = False,
    ) -> GraphModelBuilder:
        """Append spatial maximum pooling.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Required.
            stride (int | tuple[int, int] | list[int] | None): Spatial sampling step on each axis; larger
                values reduce ordinary convolution/pooling resolution. None or integer/HW pair, each in
                1..8192. Default: None.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 0.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            ceil_mode (bool): Round pooling output sizes upward, then discard windows beginning wholly in
                right/bottom padding. Boolean only. Default: False.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            stride=None uses kernel_size
            padding[i] <= floor(kernel_size[i]/2)
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.max_pool2d("node", ["image"], kernel_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(
            node_id,
            'max_pool2d',
            inputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            ceil_mode=ceil_mode,
        )

    def avg_pool2d(
        self, node_id: str, inputs: list[str],
        *,
        kernel_size: int | tuple[int, int] | list[int],
        stride: int | tuple[int, int] | list[int] | None = None,
        padding: int | tuple[int, int] | list[int] = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
    ) -> GraphModelBuilder:
        """Append spatial average pooling.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Required.
            stride (int | tuple[int, int] | list[int] | None): Spatial sampling step on each axis; larger
                values reduce ordinary convolution/pooling resolution. None or integer/HW pair, each in
                1..8192. Default: None.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 0.
            ceil_mode (bool): Round pooling output sizes upward, then discard windows beginning wholly in
                right/bottom padding. Boolean only. Default: False.
            count_include_pad (bool): Include zero-padded elements in the average-pooling divisor. Boolean
                only. Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            stride=None uses kernel_size
            padding[i] <= floor(kernel_size[i]/2)
            count_include_pad controls the mean divisor
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.avg_pool2d("node", ["image"], kernel_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(
            node_id,
            'avg_pool2d',
            inputs,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            ceil_mode=ceil_mode,
            count_include_pad=count_include_pad,
        )

    def adaptive_avg_pool2d(
        self, node_id: str, inputs: list[str],
        *,
        output_size: int | tuple[int, int] | list[int],
    ) -> GraphModelBuilder:
        """Append average pooling to a fixed spatial size.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            output_size (int | tuple[int, int] | list[int]): Target H/W grid; integer means a square. Resize
                requires exactly one of output_size and scale_factor. Integer or H/W pair, each in 1..8192.
                Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.adaptive_avg_pool2d("node", ["image"], output_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'adaptive_avg_pool2d', inputs, output_size=output_size)

    def dropout(self, node_id: str, inputs: list[str], *, p: float = 0.5) -> GraphModelBuilder:
        """Append elementwise dropout; evaluation leaves values unchanged.

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.5.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.dropout("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'dropout', inputs, p=p)

    def residual(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] | list[int] = 1,
        norm: _typing.Literal['batch', 'group', None] = 'batch',
        attention: dict[str, _typing.Any] | None = None,
    ) -> GraphModelBuilder:
        """Append the registered residual node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            norm (_typing.Literal['batch', 'group', None]): Optional post-convolution normalization: batch,
                group or None. Group normalization chooses the largest channel divisor <=8; convolution bias
                is enabled only without normalization. Choices: 'batch', 'group', None. Default: 'batch'.
            attention (dict[str, _typing.Any] | None): Optional attention applied after the residual branch;
                nested legacy attention must use matching out_channels. Default: None.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.residual("node", ["image"], in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'residual',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            norm=norm,
            attention=attention,
        )

    def se(self, node_id: str, inputs: list[str], *, channels: int, reduction: int = 8) -> GraphModelBuilder:
        """Append the registered se node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                8.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.se("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(node_id, 'se', inputs, channels=channels, reduction=reduction)

    def cbam(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7,
    ) -> GraphModelBuilder:
        """Append the registered cbam node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                16.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 7.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.cbam("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(
            node_id,
            'cbam',
            inputs,
            channels=channels,
            reduction=reduction,
            kernel_size=kernel_size,
        )

    def eca(self, node_id: str, inputs: list[str], *, channels: int, kernel_size: int = 3) -> GraphModelBuilder:
        """Append the registered eca node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.eca("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(node_id, 'eca', inputs, channels=channels, kernel_size=kernel_size)

    def coordinate(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        reduction: int = 32,
    ) -> GraphModelBuilder:
        """Append the registered coordinate node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                32.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.coordinate("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(node_id, 'coordinate', inputs, channels=channels, reduction=reduction)

    def hardswish(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered hardswish node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.hardswish("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'hardswish', inputs)

    def identity(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered identity node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same tensor/shape; no parameters.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.identity("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'identity', inputs)

    def channel_shuffle(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        groups: int = 2,
    ) -> GraphModelBuilder:
        """Append the registered channel_shuffle node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 2.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with matching channels.
            Same NCHW shape; validated channel partition.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.channel_shuffle("node", ["image"], channels=channels)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'channel_shuffle', inputs, channels=channels, groups=groups)

    def dropblock2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        p: float = 0.1,
        block_size: int = 3,
    ) -> GraphModelBuilder:
        """Append the registered dropblock2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.1.
            block_size (int): Requested spatial square side for contiguous DropBlock masks; runtime clips it
                to min(block_size,H,W). Integer in 1..8192; bool is rejected. Default: 3.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.dropblock2d("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add(node_id, 'dropblock2d', inputs, p=p, block_size=block_size)

    def pointwise_conv2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        stride: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered pointwise_conv2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.pointwise_conv2d("node", ["image"], in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            node_id,
            'pointwise_conv2d',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            stride=stride,
            bias=bias,
        )

    def depthwise_conv2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        depth_multiplier: int = 1,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered depthwise_conv2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            depth_multiplier (int): Independent filters per input channel; depthwise output/intermediate C
                equals in_channels*depth_multiplier and must not exceed 8192. Integer in 1..8192; bool is
                rejected. Default: 1.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups=in_channels
            out_channels=in_channels*depth_multiplier
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.depthwise_conv2d("node", ["image"], in_channels=in_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            node_id,
            'depthwise_conv2d',
            inputs,
            in_channels=in_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def depthwise_separable_conv2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        depth_multiplier: int = 1,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered depthwise_separable_conv2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            depth_multiplier (int): Independent filters per input channel; depthwise output/intermediate C
                equals in_channels*depth_multiplier and must not exceed 8192. Integer in 1..8192; bool is
                rejected. Default: 1.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            intermediate_channels=in_channels*depth_multiplier
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.depthwise_separable_conv2d("node", ["image"], in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            node_id,
            'depthwise_separable_conv2d',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            depth_multiplier=depth_multiplier,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def spatial_separable_conv2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered spatial_separable_conv2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            H settings apply to the vertical stage, W settings to the horizontal stage
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.spatial_separable_conv2d("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            node_id,
            'spatial_separable_conv2d',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def factorized_conv2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered factorized_conv2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            intermediate_channels <= min(in_channels,out_channels)
            1x1 reduction then KxK convolution
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.factorized_conv2d("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            node_id,
            'factorized_conv2d',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

    def conv_transpose2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int] | list[int] = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        padding: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        bias: bool = True,
        groups: int = 1,
        output_padding: int | tuple[int, int] | list[int] = 0,
    ) -> GraphModelBuilder:
        """Append the registered conv_transpose2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            kernel_size (int | tuple[int, int] | list[int]): Spatial filter/window extent in H and W; an
                integer uses the same extent on both axes. Integer or H/W pair, each in 1..8192. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            padding (int | tuple[int, int] | list[int]): Symmetric padding on each spatial axis; changes
                output geometry without changing channel count. Integer or H/W pair, each in 0..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            bias (bool): Include a learned additive convolution bias for each output channel. Boolean only.
                Default: True.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.
            output_padding (int | tuple[int, int] | list[int]): Additional output extent for transposed
                convolution, not input padding; each axis must be less than stride or dilation on that axis.
                Integer or H/W pair, each in 0..8192. Default: 0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; groups divides both channel counts.
            Floating NCHW; dilation-aware convolution or explicit transposed/multi-stage geometry. See
            parameter_dependencies and preflight.
            groups divides both channel counts
            output_padding[i] < stride[i] or dilation[i]
            Ordinary convolution: out=floor((in+2*padding-dilation*(kernel-1)-1)/stride)+1 per axis;
            transposed convolution: out=(in-1)*stride-2*padding+dilation*(kernel-1)+output_padding+1.
            Multi-stage blocks validate every intermediate shape.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.conv_transpose2d("node", ["image"], in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.convolution
        Offline: hyso docs show components.convolution --lang en
        Open: hyso docs open components.convolution --lang en
        """
        return self.add(
            node_id,
            'conv_transpose2d',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
            groups=groups,
            output_padding=output_padding,
        )

    def channel_layer_norm(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        eps: float = 1e-05,
        affine: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered channel_layer_norm node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            normalizes C at each N,H,W position, external layout stays NCHW
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.channel_layer_norm("node", ["image"], channels=channels)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        return self.add(node_id, 'channel_layer_norm', inputs, channels=channels, eps=eps, affine=affine)

    def instance_norm(
        self,
        node_id: str,
        inputs: list[str],
        *,
        num_features: int,
        eps: float = 1e-05,
        momentum: float = 0.1,
        affine: bool = False,
        track_running_stats: bool = False,
    ) -> GraphModelBuilder:
        """Append the registered instance_norm node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            num_features (int): Normalization channel count; must match input C. Integer in 1..8192; bool is
                rejected. Required.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-05.
            momentum (float): Running-statistic update factor; None selects cumulative averaging for
                BatchNorm. Number in [0,1]. Default: 0.1.
            affine (bool): Learn per-channel scale and bias when True; otherwise use normalization without
                learned affine parameters. Boolean only. Default: False.
            track_running_stats (bool): Track running mean/variance for evaluation. Without tracking,
                input-statistic sample requirements also apply in eval. Boolean only. Default: False.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW; C matches num_features, num_channels or channels. GroupNorm groups divide C.
            Same NCHW shape; channel LayerNorm normalizes C per spatial position. Native statistics
            constraints apply.
            spatial sample count H*W > 1 when using input statistics
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.instance_norm("node", ["image"], num_features=num_features)

        HYSo Guide: components.normalization
        Offline: hyso docs show components.normalization --lang en
        Open: hyso docs open components.normalization --lang en
        """
        return self.add(
            node_id,
            'instance_norm',
            inputs,
            num_features=num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def leaky_relu(self, node_id: str, inputs: list[str], *, negative_slope: float = 0.01) -> GraphModelBuilder:
        """Append the registered leaky_relu node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            negative_slope (float): Nonnegative multiplier for negative LeakyReLU inputs. Finite nonnegative
                number. Default: 0.01.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.leaky_relu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'leaky_relu', inputs, negative_slope=negative_slope)

    def prelu(
        self,
        node_id: str,
        inputs: list[str],
        *,
        num_parameters: int = 1,
        init: float = 0.25,
    ) -> GraphModelBuilder:
        """Append the registered prelu node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            num_parameters (int): PReLU slope count; must be 1 (shared) or equal to incoming C. Integer in
                1..8192; bool is rejected. Default: 1.
            init (float): Initial learned PReLU negative slope; finite and representable as float32. Finite
                number. Default: 0.25.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            num_parameters is 1 or the incoming channel count
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.prelu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'prelu', inputs, num_parameters=num_parameters, init=init)

    def mish(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered mish node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.mish("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'mish', inputs)

    def hardsigmoid(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered hardsigmoid node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW in Builder v1.
            Same NCHW shape; out-of-place activation.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.hardsigmoid("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'hardsigmoid', inputs)

    def adaptive_max_pool2d(
        self,
        node_id: str,
        inputs: list[str],
        *,
        output_size: int | tuple[int, int] | list[int],
    ) -> GraphModelBuilder:
        """Append the registered adaptive_max_pool2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            output_size (int | tuple[int, int] | list[int]): Target H/W grid; integer means a square. Resize
                requires exactly one of output_size and scale_factor. Integer or H/W pair, each in 1..8192.
                Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.adaptive_max_pool2d("node", ["image"], output_size=output_size)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add(node_id, 'adaptive_max_pool2d', inputs, output_size=output_size)

    def global_avg_pool2d(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered global_avg_pool2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW; legal window/padding or positive adaptive output_size.
            Floating NCHW with unchanged channels; global average returns [N,C,1,1].
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.global_avg_pool2d("node", ["image"])

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add(node_id, 'global_avg_pool2d', inputs)

    def bottleneck_residual(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered bottleneck_residual node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.bottleneck_residual("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'bottleneck_residual',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def preactivation_residual(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered preactivation_residual node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.preactivation_residual("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'preactivation_residual',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def wide_residual_block(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered wide_residual_block node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.wide_residual_block("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'wide_residual_block',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def grouped_bottleneck(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
        groups: int = 1,
    ) -> GraphModelBuilder:
        """Append the registered grouped_bottleneck node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.
            groups (int): Channel partitions; must divide both convolution channel counts. ChannelShuffle
                requires channels % groups == 0. Integer in 1..8192; bool is rejected. Default: 1.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.grouped_bottleneck("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'grouped_bottleneck',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
            groups=groups,
        )

    def inverted_bottleneck(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        expansion_ratio: int = 2,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered inverted_bottleneck node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            expansion_ratio (int): Inverted-bottleneck channel multiplier; intermediate
                C=in_channels*expansion_ratio must not exceed 8192. Integer in 1..8192; bool is rejected.
                Default: 2.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.inverted_bottleneck("node", ["image"], in_channels=in_channels, out_channels=out_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'inverted_bottleneck',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            expansion_ratio=expansion_ratio,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
        )

    def gated_residual(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_channels: int,
        out_channels: int,
        intermediate_channels: int,
        kernel_size: int = 3,
        stride: int | tuple[int, int] | list[int] = 1,
        dilation: int | tuple[int, int] | list[int] = 1,
        normalization: _typing.Literal['batch', 'group', None] = 'batch',
        activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu',
        attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none',
        dropout: float = 0.0,
        drop_path: float = 0.0,
        gate_init: float = 1.0,
        learnable_gate: bool = True,
    ) -> GraphModelBuilder:
        """Append the registered gated_residual node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_channels (int): Input feature channels; must match the incoming tensor C dimension. Integer
                in 1..8192; bool is rejected. Required.
            out_channels (int): Output feature channels after this convolution or block. Integer in 1..8192;
                bool is rejected. Required.
            intermediate_channels (int): Internal feature width. Factorized convolution requires this <=
                min(in_channels,out_channels); grouped residuals require divisibility by groups. Integer in
                1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 3.
            stride (int | tuple[int, int] | list[int]): Spatial sampling step on each axis; larger values
                reduce ordinary convolution/pooling resolution. Integer or H/W pair, each in 1..8192.
                Default: 1.
            dilation (int | tuple[int, int] | list[int]): Spacing between kernel taps; effective kernel size
                is dilation*(kernel_size-1)+1 per axis. Integer or H/W pair, each in 1..8192. Default: 1.
            normalization (_typing.Literal['batch', 'group', None]): Normalization of residual main-branch
                features; batch, group or None. Choices: 'batch', 'group', None. Default: 'batch'.
            activation (_typing.Literal['relu', 'silu', 'gelu']): Nonlinearity applied inside this block;
                only the declared registry choices are accepted. Choices: 'relu', 'silu', 'gelu'. Default:
                'silu'.
            attention (_typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention']): Attention
                selector inside the block; applies channel/spatial gates without resizing. Choices: 'none',
                'eca', 'spatial_attention', 'channel_attention'. Default: 'none'.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.
            drop_path (float): Training-only per-sample residual path drop probability; disabled in eval.
                Finite number in [0,1). Default: 0.0.
            gate_init (float): Initial scalar residual contribution in [0,1]; zero suppresses the residual
                main branch initially. Number in [0,1]. Default: 1.0.
            learnable_gate (bool): Store the residual gate as a learned scalar when True, otherwise a fixed
                scalar. Boolean only. Default: True.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,in_channels,H,W]; attention channels equal out_channels.
            [N,out_channels,ceil(H/strideH),ceil(W/strideW)]; internal shortcut uses the same output shape.
            Odd spatial kernels use dilation-aware symmetric padding. Channel/stride changes use an explicit
            projection shortcut; intermediate channels remain bounded to 8192.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.gated_residual("node", ["image"], in_channels=in_channels, out_channels=out_channels, intermediate_channels=intermediate_channels)

        HYSo Guide: components.residual
        Offline: hyso docs show components.residual --lang en
        Open: hyso docs open components.residual --lang en
        """
        return self.add(
            node_id,
            'gated_residual',
            inputs,
            in_channels=in_channels,
            out_channels=out_channels,
            intermediate_channels=intermediate_channels,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            normalization=normalization,
            activation=activation,
            attention=attention,
            dropout=dropout,
            drop_path=drop_path,
            gate_init=gate_init,
            learnable_gate=learnable_gate,
        )

    def spatial_attention(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        kernel_size: int = 7,
    ) -> GraphModelBuilder:
        """Append the registered spatial_attention node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            kernel_size (int): Spatial filter/window extent in H and W; an integer uses the same extent on
                both axes. Odd integer in 1..8192; bool is rejected. Default: 7.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.spatial_attention("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(node_id, 'spatial_attention', inputs, channels=channels, kernel_size=kernel_size)

    def channel_attention(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        reduction: int = 8,
    ) -> GraphModelBuilder:
        """Append the registered channel_attention node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            reduction (int): Channel-gate reduction divisor; hidden width is max(1,C//reduction), or
                max(8,C//reduction) for coordinate attention. Integer in 1..8192; bool is rejected. Default:
                8.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.channel_attention("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(node_id, 'channel_attention', inputs, channels=channels, reduction=reduction)

    def spatial_self_attention(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        embedding_dim: int | None = None,
        dropout: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered spatial_self_attention node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            embedding_dim (int | None): Attention projection width or output embedding width, according to
                the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192. None or
                integer in 1..8192. Default: None.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            QKV channels <=8192, H*W<=1024 and N*heads*(H*W)^2<=4194304 are mandatory resource bounds.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.spatial_self_attention("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(
            node_id,
            'spatial_self_attention',
            inputs,
            channels=channels,
            embedding_dim=embedding_dim,
            dropout=dropout,
        )

    def multihead_spatial_attention(
        self,
        node_id: str,
        inputs: list[str],
        *,
        channels: int,
        embedding_dim: int | None = None,
        num_heads: int = 4,
        dropout: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered multihead_spatial_attention node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            channels (int): Feature channels C; gates and normalization preserve this channel count. Integer
                in 1..8192; bool is rejected. Required.
            embedding_dim (int | None): Attention projection width or output embedding width, according to
                the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192. None or
                integer in 1..8192. Default: None.
            num_heads (int): Parallel attention heads; must divide embedding_dim. Spatial attention is
                limited to 1024 tokens and N*heads*tokens^2<=4194304. Integer in 1..8192; bool is rejected.
                Default: 4.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating [N,channels,H,W], all dimensions positive.
            Same NCHW shape; multiplicative gates, no channel or spatial resizing.
            QKV channels <=8192, H*W<=1024 and N*heads*(H*W)^2<=4194304 are mandatory resource bounds.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.multihead_spatial_attention("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(
            node_id,
            'multihead_spatial_attention',
            inputs,
            channels=channels,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

    def dropout2d(self, node_id: str, inputs: list[str], *, p: float = 0.5) -> GraphModelBuilder:
        """Append the registered dropout2d node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Number in [0,1]. Default: 0.5.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.dropout2d("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add(node_id, 'dropout2d', inputs, p=p)

    def drop_path(self, node_id: str, inputs: list[str], *, p: float = 0.0) -> GraphModelBuilder:
        """Append the registered drop_path node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            p (float): Probability of dropping elements, channels or paths during training; evaluation
                disables masking. Finite number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.drop_path("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add(node_id, 'drop_path', inputs, p=p)

    def gaussian_noise(self, node_id: str, inputs: list[str], *, std: float = 0.1) -> GraphModelBuilder:
        """Append the registered gaussian_noise node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            std (float): Standard deviation of training-only zero-mean Gaussian noise; finite, nonnegative
                and at most 1e6. Finite nonnegative number. Default: 0.1.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Nonempty floating NCHW.
            Same shape and dtype; explicit training-only masking.
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.gaussian_noise("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add(node_id, 'gaussian_noise', inputs, std=std)

    def resize(
        self,
        node_id: str,
        inputs: list[str],
        *,
        output_size: int | tuple[int, int] | list[int] | None = None,
        scale_factor: float | tuple[float, float] | list[float] | None = None,
        mode: _typing.Literal['nearest', 'bilinear', 'bicubic', 'area'] = 'nearest',
        align_corners: bool | None = None,
    ) -> GraphModelBuilder:
        """Append the registered resize node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            output_size (int | tuple[int, int] | list[int] | None): Target H/W grid; integer means a square.
                Resize requires exactly one of output_size and scale_factor. None or integer/HW pair, each
                in 1..8192. Default: None.
            scale_factor (float | tuple[float, float] | list[float] | None): Positive H/W resize
                multipliers, each <=8192. Exactly one of this and output_size must be provided. None or
                positive scalar/HW factors, each <=8192. Default: None.
            mode (_typing.Literal['nearest', 'bilinear', 'bicubic', 'area']): Resize interpolation method;
                nearest/area require align_corners=None. Choices: 'nearest', 'bilinear', 'bicubic', 'area'.
                Default: 'nearest'.
            align_corners (bool | None): Align endpoint pixel centers for bilinear/bicubic resizing; must be
                None for nearest/area. Boolean or None. Default: None.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.resize("node", ["image"], output_size=(4, 4))

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(
            node_id,
            'resize',
            inputs,
            output_size=output_size,
            scale_factor=scale_factor,
            mode=mode,
            align_corners=align_corners,
        )

    def pixel_shuffle(self, node_id: str, inputs: list[str], *, upscale_factor: int) -> GraphModelBuilder:
        """Append the registered pixel_shuffle node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            upscale_factor (int): PixelShuffle factor r: C must be divisible by r^2; output is
                (C/r^2,H*r,W*r). Integer in 1..8192; bool is rejected. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.pixel_shuffle("node", ["image"], upscale_factor=upscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'pixel_shuffle', inputs, upscale_factor=upscale_factor)

    def pixel_unshuffle(self, node_id: str, inputs: list[str], *, downscale_factor: int) -> GraphModelBuilder:
        """Append the registered pixel_unshuffle node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            downscale_factor (int): PixelUnshuffle factor r: H and W must be divisible by r; output is
                (C*r^2,H/r,W/r). Integer in 1..8192; bool is rejected. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating nonempty NCHW.
            Floating NCHW with explicit resize or pixel rearrangement geometry.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.pixel_unshuffle("node", ["image"], downscale_factor=downscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'pixel_unshuffle', inputs, downscale_factor=downscale_factor)

    def flatten(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered flatten node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.flatten("node", ["image"])

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'flatten', inputs)

    def permute(
        self,
        node_id: str,
        inputs: list[str],
        *,
        dims: tuple[int, ...] | list[int] = (0, 1, 3, 2),
    ) -> GraphModelBuilder:
        """Append the registered permute node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            dims (tuple[int, ...] | list[int]): Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are
                accepted, preserving batch/channel axes. Only identity or spatial-axis swap is supported.
                Default: (0, 1, 3, 2).

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.permute("node", ["image"])

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'permute', inputs, dims=dims)

    def reshape(
        self,
        node_id: str,
        inputs: list[str],
        *,
        shape: tuple[int, ...] | list[int],
    ) -> GraphModelBuilder:
        """Append the registered reshape node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            shape (tuple[int, ...] | list[int]): Target per-sample CHW dimensions; reshape must preserve the
                number of sample elements and every dimension is 1..8192. Three integers in 1..8192.
                Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW with preserved batch axis.
            NCHW or terminal NF for flatten, see shape_contract.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.reshape("node", ["image"], shape=shape)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'reshape', inputs, shape=shape)

    def classification_head(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_features: int | None = None,
        pooling: _typing.Literal['avg', 'max', 'none'] = 'avg',
        num_classes: int,
        dropout: float = 0.0,
    ) -> GraphModelBuilder:
        """Append the registered classification_head node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_features (int | None): Linear-head input width after pooling/flattening; must match the
                inferred feature width. Integer in 1..8192; bool is rejected. Default: None.
            pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
                none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
                Default: 'avg'.
            num_classes (int): Number of output class logits. Outputs are unnormalized scores, not
                probabilities. Integer in 1..8192; bool is rejected. Required.
            dropout (float): Training-only activation/head dropout probability; disabled in eval. Finite
                number in [0,1). Default: 0.0.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW or NF with pooling=none.
            Floating NF prediction, head type controls output semantics.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.classification_head("node", ["image"], in_features=in_features, num_classes=num_classes)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        return self._add_with_inference(node_id, 'classification_head', inputs, 'in_features', in_features, dict(pooling=pooling, num_classes=num_classes, dropout=dropout))

    def regression_head(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_features: int,
        pooling: _typing.Literal['avg', 'max', 'none'] = 'avg',
        output_features: int,
    ) -> GraphModelBuilder:
        """Append the registered regression_head node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_features (int): Linear-head input width after pooling/flattening; must match the inferred
                feature width. Integer in 1..8192; bool is rejected. Required.
            pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
                none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
                Default: 'avg'.
            output_features (int): Width of the unrestricted numeric regression output; no classification
                objective is inferred. Integer in 1..8192; bool is rejected. Required.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW or NF with pooling=none.
            Floating NF prediction, head type controls output semantics.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.regression_head("node", ["image"], in_features=in_features, output_features=output_features)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        return self.add(
            node_id,
            'regression_head',
            inputs,
            in_features=in_features,
            pooling=pooling,
            output_features=output_features,
        )

    def embedding_head(
        self,
        node_id: str,
        inputs: list[str],
        *,
        in_features: int,
        pooling: _typing.Literal['avg', 'max', 'none'] = 'avg',
        embedding_dim: int,
        normalize: bool = False,
        eps: float = 1e-12,
    ) -> GraphModelBuilder:
        """Append the registered embedding_head node via add().

        Args:
            node_id (str): Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ
                from graph input names. Required.
            inputs (list[str]): Ordered existing graph input/node references; component nodes require
                exactly one incoming tensor at preflight. Required.
            in_features (int): Linear-head input width after pooling/flattening; must match the inferred
                feature width. Integer in 1..8192; bool is rejected. Required.
            pooling (_typing.Literal['avg', 'max', 'none']): avg/max globally pool NCHW before flattening;
                none flattens NCHW or consumes flat NF. NF requires none. Choices: 'avg', 'max', 'none'.
                Default: 'avg'.
            embedding_dim (int): Output embedding width for a head, or internal attention projection width
                as declared by this component. Integer in 1..8192; bool is rejected. Required.
            normalize (bool): Apply safe per-sample L2 normalization to output embeddings. Boolean only.
                Default: False.
            eps (float): Positive numerical stabilizer used in normalization. Embedding normalization also
                floors it at the dtype tiny value. Finite positive number. Default: 1e-12.

        Returns:
            This GraphModelBuilder for fluent chaining; no model is allocated.

        Raises:
            ModelConfigError: Component type/range, channel, shape or cross-parameter validation fails.

        Notes:
            Floating NCHW or NF with pooling=none.
            Floating NF prediction, head type controls output semantics.
            resize requires exactly one target
            pixel factors require divisibility
            reshape preserves sample element count
            heads are terminal, in_features is inferred by Builder
            Integer/HW options follow the closed registry schema; unknown keywords, invalid groups and
            nonpositive output geometry raise ModelConfigError. build() allocates the model separately.

        Examples:
            builder.embedding_head("node", ["image"], in_features=in_features, embedding_dim=embedding_dim)

        HYSo Guide: components.heads
        Offline: hyso docs show components.heads --lang en
        Open: hyso docs open components.heads --lang en
        """
        return self.add(
            node_id,
            'embedding_head',
            inputs,
            in_features=in_features,
            pooling=pooling,
            embedding_dim=embedding_dim,
            normalize=normalize,
            eps=eps,
        )

    def add_inputs(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append explicit graph 'add' fusion; no broadcasting or topology repair.

        Args:
            node_id: Unique node name.
            inputs: Ordered references to 2..8 graph inputs/nodes.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native fusion parameters or graph validation reject the configuration.
            TypeError: Missing required arguments or unsupported keywords.

        Examples:
            graph.add_inputs("merge", ["left", "right"])

        HYSo Guide: components.fusion
        Offline: hyso docs show components.fusion --lang en
        Open: hyso docs open components.fusion --lang en
        """
        return self.add(node_id, 'add', inputs)

    def concat(self, node_id: str, inputs: list[str], *, axis: int = 1) -> GraphModelBuilder:
        """Append explicit graph 'concat' fusion; no broadcasting or topology repair.

        Args:
            node_id: Unique node name.
            inputs: Ordered references to 2..8 graph inputs/nodes.
            axis: Non-batch concatenation axis in 1..3. Default: 1.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native fusion parameters or graph validation reject the configuration.
            TypeError: Missing required arguments or unsupported keywords.

        Examples:
            graph.concat("merge", ["left", "right"])

        HYSo Guide: components.fusion
        Offline: hyso docs show components.fusion --lang en
        Open: hyso docs open components.fusion --lang en
        """
        return self.add(node_id, 'concat', inputs, axis=axis)

    def multiply(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append explicit graph 'multiply' fusion; no broadcasting or topology repair.

        Args:
            node_id: Unique node name.
            inputs: Ordered references to 2..8 graph inputs/nodes.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native fusion parameters or graph validation reject the configuration.
            TypeError: Missing required arguments or unsupported keywords.

        Examples:
            graph.multiply("merge", ["left", "right"])

        HYSo Guide: components.fusion
        Offline: hyso docs show components.fusion --lang en
        Open: hyso docs open components.fusion --lang en
        """
        return self.add(node_id, 'multiply', inputs)

    def weighted_sum(self, node_id: str, inputs: list[str], *, weights: list[float] | tuple[float, ...], learnable: bool = False) -> GraphModelBuilder:
        """Append explicit graph 'weighted_sum' fusion; no broadcasting or topology repair.

        Args:
            node_id: Unique node name.
            inputs: Ordered references to 2..8 graph inputs/nodes.
            weights: One finite weight per ordered input, each in [-1e6, 1e6].
            learnable: Make the weights trainable. Default: False.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native fusion parameters or graph validation reject the configuration.
            TypeError: Missing required arguments or unsupported keywords.

        Examples:
            graph.weighted_sum("merge", ["left", "right"], weights=[0.5, 0.5])

        HYSo Guide: components.fusion
        Offline: hyso docs show components.fusion --lang en
        Open: hyso docs open components.fusion --lang en
        """
        return self.add(node_id, 'weighted_sum', inputs, weights=weights, learnable=learnable)

    # BEGIN DEVELOPMENT REGISTRY WRAPPERS
    if DEVELOPMENT:
        def simam(
            self,
            node_id: str,
            inputs: list[str],
            *,
            channels: int,
            epsilon: float = 0.0001,
        ) -> GraphModelBuilder:
            """Append the registered simam node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                channels: channels Required.
                epsilon: Numerical tolerance/stabilizer used by this operation. Default: 0.0001.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.simam("node", ["image"], channels=channels)

            HYSo Guide: components.attention
            Offline: hyso docs show components.attention --lang en
            Open: hyso docs open components.attention --lang en
            """
            return self.add(node_id, 'simam', inputs, channels=channels, epsilon=epsilon)
        def triplet(
            self,
            node_id: str,
            inputs: list[str],
            *,
            channels: int,
            kernel_size: int = 7,
            spatial: bool = True,
        ) -> GraphModelBuilder:
            """Append the registered triplet node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                channels: channels Required.
                kernel_size: kernel size Default: 7.
                spatial: Include spatial attention where supported. Default: True.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.triplet("node", ["image"], channels=channels)

            HYSo Guide: components.attention
            Offline: hyso docs show components.attention --lang en
            Open: hyso docs open components.attention --lang en
            """
            return self.add(node_id, 'triplet', inputs, channels=channels, kernel_size=kernel_size, spatial=spatial)
        def selective_kernel(
            self,
            node_id: str,
            inputs: list[str],
            *,
            channels: int,
            branches: int = 2,
            groups: int = 1,
            reduction: int = 16,
            minimum: int = 32,
        ) -> GraphModelBuilder:
            """Append the registered selective_kernel node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                channels: channels Required.
                branches: Explicit branch modules/configurations. Default: 2.
                groups: groups Default: 1.
                reduction: reduction Default: 16.
                minimum: Lower bound applied by this operation. Default: 32.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.selective_kernel("node", ["image"], channels=channels)

            HYSo Guide: components.attention
            Offline: hyso docs show components.attention --lang en
            Open: hyso docs open components.attention --lang en
            """
            return self.add(
                node_id,
                'selective_kernel',
                inputs,
                channels=channels,
                branches=branches,
                groups=groups,
                reduction=reduction,
                minimum=minimum,
            )
        def basic_residual(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
        ) -> GraphModelBuilder:
            """Append the registered basic_residual node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.basic_residual("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                node_id,
                'basic_residual',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
            )
        def bottleneck(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            groups: int = 1,
            width_per_group: int = 64,
        ) -> GraphModelBuilder:
            """Append the registered bottleneck node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                groups: groups Default: 1.
                width_per_group: Feature width within each convolution group. Default: 64.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.bottleneck("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                node_id,
                'bottleneck',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                groups=groups,
                width_per_group=width_per_group,
            )
        def preactivation(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            bottleneck: bool = False,
            dropout: float = 0.0,
        ) -> GraphModelBuilder:
            """Append the registered preactivation node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                bottleneck: Select the supported bottleneck block form. Default: False.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.preactivation("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                node_id,
                'preactivation',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                bottleneck=bottleneck,
                dropout=dropout,
            )
        def wide_residual(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            dropout: float = 0.0,
        ) -> GraphModelBuilder:
            """Append the registered wide_residual node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.wide_residual("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                node_id,
                'wide_residual',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                dropout=dropout,
            )
        def resnext_bottleneck(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            norm: _typing.Literal['batch', 'group', None] = 'batch',
            groups: int = 32,
            width_per_group: int = 4,
        ) -> GraphModelBuilder:
            """Append the registered resnext_bottleneck node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                norm: M3 normalization; group uses gcd(8, channels). Default: 'batch'.
                groups: groups Default: 32.
                width_per_group: Feature width within each convolution group. Default: 4.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.resnext_bottleneck("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.residual
            Offline: hyso docs show components.residual --lang en
            Open: hyso docs open components.residual --lang en
            """
            return self.add(
                node_id,
                'resnext_bottleneck',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                norm=norm,
                groups=groups,
                width_per_group=width_per_group,
            )
        def inverted_residual(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 6.0,
        ) -> GraphModelBuilder:
            """Append the registered inverted_residual node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                expand_ratio: Internal channel expansion ratio. Default: 6.0.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.inverted_residual("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'inverted_residual',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
            )
        def mbconv(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 6.0,
            kernel_size: int = 3,
            stochastic_depth: float = 0.0,
        ) -> GraphModelBuilder:
            """Append the registered mbconv node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                expand_ratio: Internal channel expansion ratio. Default: 6.0.
                kernel_size: kernel size Default: 3.
                stochastic_depth: Probability/schedule for dropping residual paths during training. Default: 0.0.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.mbconv("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'mbconv',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                stochastic_depth=stochastic_depth,
            )
        def fused_mbconv(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            expand_ratio: float = 4.0,
            kernel_size: int = 3,
            stochastic_depth: float = 0.0,
        ) -> GraphModelBuilder:
            """Append the registered fused_mbconv node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                expand_ratio: Internal channel expansion ratio. Default: 4.0.
                kernel_size: kernel size Default: 3.
                stochastic_depth: Probability/schedule for dropping residual paths during training. Default: 0.0.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.fused_mbconv("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'fused_mbconv',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                expand_ratio=expand_ratio,
                kernel_size=kernel_size,
                stochastic_depth=stochastic_depth,
            )
        def ghost_conv(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            kernel_size: int = 1,
            ratio: int = 2,
            cheap_kernel: int = 3,
            relu: bool = True,
        ) -> GraphModelBuilder:
            """Append the registered ghost_conv node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                kernel_size: kernel size Default: 1.
                ratio: Feature expansion/split ratio used by this block. Default: 2.
                cheap_kernel: Spatial kernel of the inexpensive feature-generation branch. Default: 3.
                relu: Enable the block's ReLU activation. Default: True.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.ghost_conv("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'ghost_conv',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                ratio=ratio,
                cheap_kernel=cheap_kernel,
                relu=relu,
            )
        def ghost_bottleneck(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            hidden_channels: int,
            kernel_size: int = 3,
            se_ratio: float = 0.0,
        ) -> GraphModelBuilder:
            """Append the registered ghost_bottleneck node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                hidden_channels: Internal feature width. Required.
                kernel_size: kernel size Default: 3.
                se_ratio: Squeeze-and-excitation reduction setting. Default: 0.0.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.ghost_bottleneck("node", ["image"], in_channels=in_channels, out_channels=out_channels, hidden_channels=hidden_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'ghost_bottleneck',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                hidden_channels=hidden_channels,
                kernel_size=kernel_size,
                se_ratio=se_ratio,
            )
        def shuffle_unit(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
        ) -> GraphModelBuilder:
            """Append the registered shuffle_unit node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.shuffle_unit("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'shuffle_unit',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
            )
        def repvgg_block(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            groups: int = 1,
            deploy: bool = False,
        ) -> GraphModelBuilder:
            """Append the registered repvgg_block node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                groups: groups Default: 1.
                deploy: Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.repvgg_block("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'repvgg_block',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                groups=groups,
                deploy=deploy,
            )
        def mobileone_block(
            self,
            node_id: str,
            inputs: list[str],
            *,
            in_channels: int,
            out_channels: int,
            stride: int = 1,
            kernel_size: int = 3,
            groups: int = 1,
            num_conv_branches: int = 1,
            use_se: bool = False,
            deploy: bool = False,
        ) -> GraphModelBuilder:
            """Append the registered mobileone_block node via add().

            Development profile only; absent from commercial installations.
            No weights, forward pass or training are requested.

            Args:
                node_id: Unique node name.
                inputs: Ordered explicit graph input/node references.
                in_channels: in channels Required.
                out_channels: out channels Required.
                stride: stride Default: 1.
                kernel_size: kernel size Default: 3.
                groups: groups Default: 1.
                num_conv_branches: Parallel convolution branch count in the training form. Default: 1.
                use_se: Enable squeeze-and-excitation attention. Default: False.
                deploy: Use the deployment form of this reparameterized block; conversion support remains explicit. Default: False.

            Returns:
                This GraphModelBuilder for chaining.

            Raises:
                ModelConfigError: Native Registry, profile, shape or configuration validation fails.
                TypeError: A required argument is missing or an unknown keyword is supplied.

            Example:
                builder.mobileone_block("node", ["image"], in_channels=in_channels, out_channels=out_channels)

            HYSo Guide: components.mobile
            Offline: hyso docs show components.mobile --lang en
            Open: hyso docs open components.mobile --lang en
            """
            return self.add(
                node_id,
                'mobileone_block',
                inputs,
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                kernel_size=kernel_size,
                groups=groups,
                num_conv_branches=num_conv_branches,
                use_se=use_se,
                deploy=deploy,
            )
    # END DEVELOPMENT REGISTRY WRAPPERS

    # Static keyword contracts mirror the closed runtime registry.
    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['conv_norm_act'], inputs: list[str], *, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, groups: int = 1, norm: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['silu', 'relu', None] = 'silu') -> GraphModelBuilder:
        """Keyword contract for conv_norm_act; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'conv_norm_act'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.
            norm: Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['conv2d'], inputs: list[str], *, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, groups: int = 1, bias: bool = True, dilation: int | tuple[int, int] | list[int] = 1) -> GraphModelBuilder:
        """Keyword contract for conv2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'conv2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.
            bias: Include a learned additive convolution bias for each output channel.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['residual'], inputs: list[str], *, in_channels: int, out_channels: int, stride: int | tuple[int, int] | list[int] = 1, norm: _typing.Literal['batch', 'group', None] = 'batch', attention: dict[str, _typing.Any] | None = None) -> GraphModelBuilder:
        """Keyword contract for residual; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'residual'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            norm: Optional post-convolution normalization: batch, group or None. Group normalization chooses the largest channel divisor <=8; convolution bias is enabled only without normalization.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['se'], inputs: list[str], *, channels: int, reduction: int = 8) -> GraphModelBuilder:
        """Keyword contract for se; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'se'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['cbam'], inputs: list[str], *, channels: int, reduction: int = 16, kernel_size: int = 7) -> GraphModelBuilder:
        """Keyword contract for cbam; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'cbam'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['eca'], inputs: list[str], *, channels: int, kernel_size: int = 3) -> GraphModelBuilder:
        """Keyword contract for eca; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'eca'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['coordinate'], inputs: list[str], *, channels: int, reduction: int = 32) -> GraphModelBuilder:
        """Keyword contract for coordinate; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'coordinate'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['batch_norm'], inputs: list[str], *, num_features: int, eps: float = 1e-05, momentum: float | None = 0.1, affine: bool = True, track_running_stats: bool = True) -> GraphModelBuilder:
        """Keyword contract for batch_norm; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'batch_norm'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            num_features: Normalization channel count; must match input C.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            momentum: Running-statistic update factor; None selects cumulative averaging for BatchNorm.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.
            track_running_stats: Track running mean/variance for evaluation. Without tracking, input-statistic sample requirements also apply in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['group_norm'], inputs: list[str], *, num_groups: int, num_channels: int, eps: float = 1e-05, affine: bool = True) -> GraphModelBuilder:
        """Keyword contract for group_norm; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'group_norm'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            num_groups: GroupNorm channel partitions; must divide num_channels.
            num_channels: Total GroupNorm channels C; must be divisible by num_groups.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['relu'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for relu; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'relu'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['silu'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for silu; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'silu'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['hardswish'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for hardswish; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'hardswish'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['identity'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for identity; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'identity'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['channel_shuffle'], inputs: list[str], *, channels: int, groups: int = 2) -> GraphModelBuilder:
        """Keyword contract for channel_shuffle; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'channel_shuffle'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['dropblock2d'], inputs: list[str], *, p: float = 0.1, block_size: int = 3) -> GraphModelBuilder:
        """Keyword contract for dropblock2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'dropblock2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.
            block_size: Requested mask side, clipped at runtime to min(block_size,H,W).

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['pointwise_conv2d'], inputs: list[str], *, in_channels: int, out_channels: int, stride: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> GraphModelBuilder:
        """Keyword contract for pointwise_conv2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'pointwise_conv2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['depthwise_conv2d'], inputs: list[str], *, in_channels: int, depth_multiplier: int = 1, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> GraphModelBuilder:
        """Keyword contract for depthwise_conv2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'depthwise_conv2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            depth_multiplier: Independent filters per input channel; depthwise output/intermediate C equals in_channels*depth_multiplier and must not exceed 8192.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['depthwise_separable_conv2d'], inputs: list[str], *, in_channels: int, out_channels: int, depth_multiplier: int = 1, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> GraphModelBuilder:
        """Keyword contract for depthwise_separable_conv2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'depthwise_separable_conv2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            depth_multiplier: Independent filters per input channel; depthwise output/intermediate C equals in_channels*depth_multiplier and must not exceed 8192.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['spatial_separable_conv2d'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> GraphModelBuilder:
        """Keyword contract for spatial_separable_conv2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'spatial_separable_conv2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['factorized_conv2d'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True) -> GraphModelBuilder:
        """Keyword contract for factorized_conv2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'factorized_conv2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['conv_transpose2d'], inputs: list[str], *, in_channels: int, out_channels: int, kernel_size: int | tuple[int, int] | list[int] = 3, stride: int | tuple[int, int] | list[int] = 1, padding: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, bias: bool = True, groups: int = 1, output_padding: int | tuple[int, int] | list[int] = 0) -> GraphModelBuilder:
        """Keyword contract for conv_transpose2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'conv_transpose2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            bias: Include a learned additive convolution bias for each output channel.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.
            output_padding: Additional output extent for transposed convolution, not input padding; each axis must be less than stride or dilation on that axis.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['channel_layer_norm'], inputs: list[str], *, channels: int, eps: float = 1e-05, affine: bool = True) -> GraphModelBuilder:
        """Keyword contract for channel_layer_norm; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'channel_layer_norm'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['instance_norm'], inputs: list[str], *, num_features: int, eps: float = 1e-05, momentum: float = 0.1, affine: bool = False, track_running_stats: bool = False) -> GraphModelBuilder:
        """Keyword contract for instance_norm; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'instance_norm'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            num_features: Normalization channel count; must match input C.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.
            momentum: Running-statistic update factor; None selects cumulative averaging for BatchNorm.
            affine: Learn per-channel scale and bias when True; otherwise use normalization without learned affine parameters.
            track_running_stats: Track running mean/variance for evaluation. Without tracking, input-statistic sample requirements also apply in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['leaky_relu'], inputs: list[str], *, negative_slope: float = 0.01) -> GraphModelBuilder:
        """Keyword contract for leaky_relu; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'leaky_relu'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            negative_slope: Nonnegative multiplier for negative LeakyReLU inputs.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['prelu'], inputs: list[str], *, num_parameters: int = 1, init: float = 0.25) -> GraphModelBuilder:
        """Keyword contract for prelu; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'prelu'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            num_parameters: PReLU slope count; must be 1 (shared) or equal to incoming C.
            init: Initial learned PReLU negative slope; finite and representable as float32.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['gelu'], inputs: list[str], *, approximate: _typing.Literal['none', 'tanh'] = 'none') -> GraphModelBuilder:
        """Keyword contract for gelu; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'gelu'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            approximate: GELU calculation: none uses the exact error function; tanh uses the approximation.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['mish'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for mish; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'mish'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['hardsigmoid'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for hardsigmoid; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'hardsigmoid'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['max_pool2d'], inputs: list[str], *, kernel_size: int | tuple[int, int] | list[int], stride: int | tuple[int, int] | list[int] | None = None, padding: int | tuple[int, int] | list[int] = 0, dilation: int | tuple[int, int] | list[int] = 1, ceil_mode: bool = False) -> GraphModelBuilder:
        """Keyword contract for max_pool2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'max_pool2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            ceil_mode: Round pooling output sizes upward, then discard windows beginning wholly in right/bottom padding.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['avg_pool2d'], inputs: list[str], *, kernel_size: int | tuple[int, int] | list[int], stride: int | tuple[int, int] | list[int] | None = None, padding: int | tuple[int, int] | list[int] = 0, ceil_mode: bool = False, count_include_pad: bool = True) -> GraphModelBuilder:
        """Keyword contract for avg_pool2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'avg_pool2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            padding: Symmetric padding on each spatial axis; changes output geometry without changing channel count.
            ceil_mode: Round pooling output sizes upward, then discard windows beginning wholly in right/bottom padding.
            count_include_pad: Include zero-padded elements in the average-pooling divisor.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['adaptive_avg_pool2d'], inputs: list[str], *, output_size: int | tuple[int, int] | list[int]) -> GraphModelBuilder:
        """Keyword contract for adaptive_avg_pool2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'adaptive_avg_pool2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            output_size: Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['adaptive_max_pool2d'], inputs: list[str], *, output_size: int | tuple[int, int] | list[int]) -> GraphModelBuilder:
        """Keyword contract for adaptive_max_pool2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'adaptive_max_pool2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            output_size: Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['global_avg_pool2d'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for global_avg_pool2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'global_avg_pool2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['bottleneck_residual'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for bottleneck_residual; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'bottleneck_residual'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['preactivation_residual'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for preactivation_residual; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'preactivation_residual'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['wide_residual_block'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for wide_residual_block; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'wide_residual_block'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['grouped_bottleneck'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0, groups: int = 1) -> GraphModelBuilder:
        """Keyword contract for grouped_bottleneck; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'grouped_bottleneck'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.
            groups: Channel partitions; must divide both convolution channel counts. ChannelShuffle requires channels % groups == 0.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['inverted_bottleneck'], inputs: list[str], *, in_channels: int, out_channels: int, expansion_ratio: int = 2, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for inverted_bottleneck; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'inverted_bottleneck'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            expansion_ratio: Inverted-bottleneck channel multiplier; intermediate C=in_channels*expansion_ratio must not exceed 8192.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['gated_residual'], inputs: list[str], *, in_channels: int, out_channels: int, intermediate_channels: int, kernel_size: int = 3, stride: int | tuple[int, int] | list[int] = 1, dilation: int | tuple[int, int] | list[int] = 1, normalization: _typing.Literal['batch', 'group', None] = 'batch', activation: _typing.Literal['relu', 'silu', 'gelu'] = 'silu', attention: _typing.Literal['none', 'eca', 'spatial_attention', 'channel_attention'] = 'none', dropout: float = 0.0, drop_path: float = 0.0, gate_init: float = 1.0, learnable_gate: bool = True) -> GraphModelBuilder:
        """Keyword contract for gated_residual; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'gated_residual'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_channels: Input feature channels; must match the incoming tensor C dimension.
            out_channels: Output feature channels after this convolution or block.
            intermediate_channels: Internal feature width. Factorized convolution requires this <= min(in_channels,out_channels); grouped residuals require divisibility by groups.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.
            stride: Spatial sampling step on each axis; larger values reduce ordinary convolution/pooling resolution.
            dilation: Spacing between kernel taps; effective kernel size is dilation*(kernel_size-1)+1 per axis.
            normalization: Normalization of residual main-branch features; batch, group or None.
            activation: Nonlinearity applied inside this block; only the declared registry choices are accepted.
            attention: Optional attention applied after the residual branch; nested legacy attention must use matching out_channels.
            dropout: Training-only activation/head dropout probability; disabled in eval.
            drop_path: Training-only per-sample residual path drop probability; disabled in eval.
            gate_init: Initial scalar residual contribution in [0,1]; zero suppresses the residual main branch initially.
            learnable_gate: Store the residual gate as a learned scalar when True, otherwise a fixed scalar.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['spatial_attention'], inputs: list[str], *, channels: int, kernel_size: int = 7) -> GraphModelBuilder:
        """Keyword contract for spatial_attention; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'spatial_attention'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            kernel_size: Spatial filter/window extent in H and W; an integer uses the same extent on both axes.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['channel_attention'], inputs: list[str], *, channels: int, reduction: int = 8) -> GraphModelBuilder:
        """Keyword contract for channel_attention; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'channel_attention'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            reduction: Channel-gate reduction divisor; hidden width is max(1,C//reduction), or max(8,C//reduction) for coordinate attention.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['spatial_self_attention'], inputs: list[str], *, channels: int, embedding_dim: int | None = None, dropout: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for spatial_self_attention; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'spatial_self_attention'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            embedding_dim: Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192.
            dropout: Training-only activation/head dropout probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['multihead_spatial_attention'], inputs: list[str], *, channels: int, embedding_dim: int | None = None, num_heads: int = 4, dropout: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for multihead_spatial_attention; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'multihead_spatial_attention'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            channels: Feature channels C; gates and normalization preserve this channel count.
            embedding_dim: Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192.
            num_heads: Parallel attention heads; must divide embedding_dim. Spatial attention is limited to 1024 tokens and N*heads*tokens^2<=4194304.
            dropout: Training-only activation/head dropout probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['dropout'], inputs: list[str], *, p: float = 0.5) -> GraphModelBuilder:
        """Keyword contract for dropout; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'dropout'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['dropout2d'], inputs: list[str], *, p: float = 0.5) -> GraphModelBuilder:
        """Keyword contract for dropout2d; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'dropout2d'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['drop_path'], inputs: list[str], *, p: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for drop_path; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'drop_path'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            p: Probability of dropping elements, channels or paths during training; evaluation disables masking.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['gaussian_noise'], inputs: list[str], *, std: float = 0.1) -> GraphModelBuilder:
        """Keyword contract for gaussian_noise; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'gaussian_noise'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            std: Standard deviation of training-only zero-mean Gaussian noise; finite, nonnegative and at most 1e6.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['resize'], inputs: list[str], *, output_size: int | tuple[int, int] | list[int] | None = None, scale_factor: float | tuple[float, float] | list[float] | None = None, mode: _typing.Literal['nearest', 'bilinear', 'bicubic', 'area'] = 'nearest', align_corners: bool | None = None) -> GraphModelBuilder:
        """Keyword contract for resize; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'resize'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            output_size: Target H/W grid; integer means a square. Resize requires exactly one of output_size and scale_factor.
            scale_factor: Positive H/W resize multipliers, each <=8192. Exactly one of this and output_size must be provided.
            mode: Interpolation or transformation strategy; the closed choices are specific to this operation.
            align_corners: Align endpoint pixel centers for bilinear/bicubic resizing; must be None for nearest/area.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['pixel_shuffle'], inputs: list[str], *, upscale_factor: int) -> GraphModelBuilder:
        """Keyword contract for pixel_shuffle; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'pixel_shuffle'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            upscale_factor: PixelShuffle factor r: C must be divisible by r^2; output is (C/r^2,H*r,W*r).

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['pixel_unshuffle'], inputs: list[str], *, downscale_factor: int) -> GraphModelBuilder:
        """Keyword contract for pixel_unshuffle; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'pixel_unshuffle'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            downscale_factor: PixelUnshuffle factor r: H and W must be divisible by r; output is (C*r^2,H/r,W/r).

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['flatten'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for flatten; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'flatten'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['permute'], inputs: list[str], *, dims: tuple[int, ...] | list[int] = [0, 1, 3, 2]) -> GraphModelBuilder:
        """Keyword contract for permute; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'permute'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            dims: Permutation of NCHW axes; only (0,1,2,3) and (0,1,3,2) are accepted, preserving batch/channel axes.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['reshape'], inputs: list[str], *, shape: tuple[int, int, int] | list[int]) -> GraphModelBuilder:
        """Keyword contract for reshape; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'reshape'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            shape: Target per-sample CHW dimensions; reshape must preserve the number of sample elements and every dimension is 1..8192.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['classification_head'], inputs: list[str], *, in_features: int, pooling: _typing.Literal['avg', 'max', 'none'] = 'avg', num_classes: int, dropout: float = 0.0) -> GraphModelBuilder:
        """Keyword contract for classification_head; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'classification_head'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_features: Linear-head input width after pooling/flattening; must match the inferred feature width.
            pooling: avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF.
            num_classes: Number of output class logits. Outputs are unnormalized scores, not probabilities.
            dropout: Training-only activation/head dropout probability; disabled in eval.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['regression_head'], inputs: list[str], *, in_features: int, pooling: _typing.Literal['avg', 'max', 'none'] = 'avg', output_features: int) -> GraphModelBuilder:
        """Keyword contract for regression_head; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'regression_head'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_features: Linear-head input width after pooling/flattening; must match the inferred feature width.
            pooling: avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF.
            output_features: Width of the unrestricted numeric regression output; no classification objective is inferred.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['embedding_head'], inputs: list[str], *, in_features: int, pooling: _typing.Literal['avg', 'max', 'none'] = 'avg', embedding_dim: int, normalize: bool = False, eps: float = 1e-12) -> GraphModelBuilder:
        """Keyword contract for embedding_head; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'embedding_head'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            in_features: Linear-head input width after pooling/flattening; must match the inferred feature width.
            pooling: avg/max globally pool NCHW before flattening; none flattens NCHW or consumes flat NF.
            embedding_dim: Attention projection width or output embedding width, according to the component. None in self-attention uses C; QKV requires 3*embedding_dim<=8192.
            normalize: Apply safe per-sample L2 normalization to output embeddings.
            eps: Positive numerical stabilizer used in normalization. Embedding normalization also floors it at the dtype tiny value.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['add'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for add; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'add'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['multiply'], inputs: list[str]) -> GraphModelBuilder:
        """Keyword contract for multiply; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'multiply'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['concat'], inputs: list[str], *, axis: int = 1) -> GraphModelBuilder:
        """Keyword contract for concat; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'concat'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            axis: Non-batch concatenation axis 1..3; other dimensions must match.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    @_typing.overload
    def add(self, node_id: str, component: _typing.Literal['weighted_sum'], inputs: list[str], *, weights: list[float] | tuple[float, ...], learnable: bool = False) -> GraphModelBuilder:
        """Keyword contract for weighted_sum; execution uses the validated implementation.

        Args:
            node_id: Unique graph node identifier matching [A-Za-z][A-Za-z0-9_]{0,47}; must differ from graph input names.
            component: Exact component identifier 'weighted_sum'.
            inputs: Ordered graph input/node references; topology is validated by preflight/build.
            weights: One finite coefficient per ordered input, 2..8 values in [-1e6,1e6]; tensors must have identical shape.
            learnable: Register fusion coefficients as trainable parameters when True; otherwise retain fixed buffers.

        Returns:
            This GraphModelBuilder for chaining; build() allocates separately.
        """
        ...

    def add(self,node_id: 'str',component: 'str',inputs: 'list[str]',**params: '_typing.Any') -> 'GraphModelBuilder':
        """Append a validated component and its ordered input references to the graph.

        Args:
            node_id (str): Unique stable Graph node name, distinct from graph inputs. Required.
            component (str): Closed registered component or supported Graph fusion name. Required.
            inputs (list[str]): Ordered list of 1..8 Graph input/node reference strings. Required.
            **params (_typing.Any): Component keyword parameters; registered defaults apply to omitted
            optional keys.

        Returns:
            This builder after component parameter and input-list validation.
            Forward references and graph topology are checked by preflight/build.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            builder.add("stem", "conv2d", ["image"], in_channels=1, out_channels=4, kernel_size=3,
            padding=1)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        if not _identifier(node_id) or node_id in self._config['inputs'] or any(row['id']==node_id for row in self._config['nodes']):
            raise _error('invalid or duplicate node ID',layer_id=node_id if type(node_id) is str else None,actual=node_id)
        if len(self._config['nodes'])>=MAX_NODES:raise ModelConfigError('node limit exceeded',code='GRAPH_LIMIT_ERROR')
        normalized=graph_ops.parameters(component,params) if type(component) is str and component in graph_ops.NAMES else ModelRegistry().validate(component,params)
        row=dict(id=node_id,component=component,inputs=_plain(inputs),params=normalized)
        if type(row['inputs']) is not list or not 1<=len(row['inputs'])<=8 or any(type(v) is not str for v in row['inputs']):
            raise _error('node inputs require 1..8 ordered references',layer_id=node_id)
        self._config['nodes'].append(row)
        return self

    def outputs(self,**references: 'str') -> 'GraphModelBuilder':
        """Assign names to 1..8 graph output references; this does not prune nodes.

        Args:
            **references (str): 1..8 output_name=node_id pairs; selecting an output does not prune other
            nodes.

        Returns:
            This builder with the explicit named outputs configured; other graph nodes are not pruned.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.outputs(logits="head")  # head is an existing node ID

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        if not 1<=len(references)<=8 or any(not _identifier(k) or type(v) is not str for k,v in references.items()):
            raise _error('named outputs require safe names and reference strings')
        self._config['outputs']=dict(sorted(references.items()));return self

    def to_config(self) -> 'dict[str, _typing.Any]':
        """Copy the versioned architecture configuration without model weights.

        Returns:
            Detached JSON-compatible architecture metadata including normalized defaults; no weights.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_config()

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        c=_plain(self._config);c['nodes']=sorted(c['nodes'],key=lambda row:row['id']);return c

    @classmethod
    def from_config(cls,config: 'dict[str, _typing.Any]') -> 'GraphModelBuilder':
        """Validate a versioned configuration and reconstruct its builder.

        Args:
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained
                state. Required.

        Returns:
            A validated builder/configuration object; no executable names are imported.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            GraphModelBuilder.from_config(config)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        report,c=_analyze(config)
        if not report['valid']:
            diagnostic=report['diagnostics'][0]
            raise ModelConfigError(diagnostic['message_en'],code=diagnostic['code'],layer_id=diagnostic['layer_id'],
                expected=diagnostic['expected'],actual=diagnostic['actual'],suggestion=diagnostic['suggestion'])
        result=cls.__new__(cls);result._config=c;return result

    @staticmethod
    def preflight(config: 'dict[str, _typing.Any]',*,batch_size: 'int'=1,activation_limit: 'int'=16_777_216) -> 'dict[str, _typing.Any]':
        """Validate graph structure, infer shapes and estimate bounded resource needs without constructing a
        model.

        Args:
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained
                state. Required.
            batch_size (int): Positive batch-size setting or metadata assumption; no implicit batch search.
                Default: 1.
            activation_limit (int): Advisory per-tensor element threshold used by metadata preflight, not a
                physical-memory quota. Default: 16777216.

        Returns:
            Metadata validation, shapes, diagnostics and advisory resource estimates; no model is built.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.preflight(config)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        return _analyze(config,batch_size,activation_limit)[0]

    def preview(self, *, batch_size: int = 1, activation_limit: int = 16777216) -> 'dict[str, _typing.Any]':
        """Inspect this graph using the allocation-free preflight checks.

        Args:
            batch_size (int): Positive metadata preview batch size. Default: 1.
            activation_limit (int): Native preflight activation element limit. Default: 16777216.

        Returns:
            Allocation-free Graph metadata preview, preserving unknown/unsupported conditions.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.preview()

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        return self.preflight(self.to_config(), batch_size=batch_size, activation_limit=activation_limit)

    def build(self) -> 'GraphModel':
        """Construct the validated graph on CPU without running a forward pass.

        Returns:
            A fresh CPU model. No forward, training, CUDA move or checkpoint load is requested.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.build()

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        config=self.from_config(self.to_config()).to_config();report=self.preflight(config)
        with torch.device('cpu'):return GraphModel(config,report)

    def to_json(self,path: '_typing.Any',*,overwrite: 'bool'=False) -> 'None':
        """Serialize this object using its native JSON format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain
                protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json(path)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        if type(overwrite) is not bool:raise _error('overwrite must be bool')
        raw=json.dumps(self.to_config(),indent=2,allow_nan=False)+'\n'
        if len(raw.encode('utf-8'))>MAX_BYTES:raise _error('graph JSON exceeds 2 MiB')
        with Path(path).open('w' if overwrite else 'x',encoding='utf-8',newline='\n') as stream:stream.write(raw)

    @classmethod
    def from_json(cls,path: '_typing.Any') -> 'GraphModelBuilder':
        """Read and validate this type from a local JSON file.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated GraphModelBuilder instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            GraphModelBuilder.from_json(path)

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        with Path(path).open('rb') as stream:raw=stream.read(MAX_BYTES+1)
        if len(raw)>MAX_BYTES:raise _error('graph JSON exceeds 2 MiB')
        def invalid(value):raise _error('non-finite JSON number',actual=value)
        try:return cls.from_config(json.loads(raw.decode('utf-8'),object_pairs_hook=_pairs,parse_constant=invalid))
        except ModelConfigError:raise
        except (ValueError,TypeError,RecursionError) as exc:raise _error(str(exc)) from exc
