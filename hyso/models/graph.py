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
        config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained state. Required.
        report (_typing.Any): Validated retained analysis/advisory report. Required.

    Returns:
        An instance of GraphModel with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        RuntimeError: A requested native construction/forward operation fails.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        GraphModel(config, report)

    HYSo Guide: models.graph-builder
    Offline: hyso docs show models.graph-builder --lang en
    Open: hyso docs open models.graph-builder --lang en
    """
    def __init__(self,config: 'dict[str, _typing.Any]',report: '_typing.Any') -> 'None':
        """Initialize GraphModel with the supplied settings.

        Args:
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained state. Required.
            report (_typing.Any): Validated retained analysis/advisory report. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
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
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_config()

        HYSo Guide: models.graph-builder
        Offline: hyso docs show models.graph-builder --lang en
        Open: hyso docs open models.graph-builder --lang en
        """
        return json.loads(self._architecture)

    def forward(self,inputs: '_typing.Any') -> '_typing.Any':
        """Apply this module to inputs with its declared shape, dtype and device.

        Args:
            inputs (_typing.Any): Named input specifications for the builder, ordered references for add(), or supported runtime inputs for forward(). Required.

        Returns:
            Native Tensor output(s) of this model/block; this call performs the forward computation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
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
    """Closed schema3 DAG builder with allocation-free planning and preview.

    Args:
        inputs (dict[str, list[int] | tuple[int, ...] | dict[str, object]]): Named input shapes without batch, or closed shape/layout/dtype/device specifications. Required.

    Returns:
        An instance of GraphModelBuilder with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: Native configuration, shape or input validation rejects the request.
        ModelConfigError: Input names, shapes, layout, dtype or device metadata violate the closed graph contract.

    Example:
        # Caller-supplied values; explicitly requested operations only.
        builder = GraphModelBuilder({"image": [1, 4, 4]})

    HYSo Guide: models.graph-builder
    Offline: hyso docs show models.graph-builder --lang en
    Open: hyso docs open models.graph-builder --lang en
    """
    def __init__(self,inputs: 'dict[str, _typing.Any]') -> 'None':
        """Initialize GraphModelBuilder with the supplied settings.

        Args:
            inputs (dict[str, _typing.Any]): Dictionary of named concrete shape sequences or closed input-specification dictionaries. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            in_channels: Expected input channel count. Required.
            out_channels: Output channel count. Required.
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Default: 3.
            stride: Positive spatial sampling step. Default: 1.
            padding: Nonnegative spatial padding. Default: 1.
            groups: Group count; input and output channels must be divisible by it. Default: 1.
            bias: Include the convolution bias. Default: True.
            dilation: Positive spacing between filter/window taps. Default: 1.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            in_channels: Expected input channel count. Required.
            out_channels: Output channel count. Required.
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Default: 3.
            stride: Positive spatial sampling step. Default: 1.
            padding: Nonnegative spatial padding. Default: 1.
            groups: Group count; input and output channels must be divisible by it. Default: 1.
            norm: Convolution normalization: batch, group, or None. Default: 'batch'.
            activation: Convolution activation: silu, relu, or None. Default: 'silu'.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.relu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'relu', inputs)

    def silu(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append out-of-place SiLU activation.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            approximate: GELU approximation: none (exact) or tanh. Default: 'none'.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            num_features: Expected feature-channel count. Required.
            eps: Positive normalization stabilizer. Default: 1e-05.
            momentum: Running-statistic factor in [0, 1]; None uses the cumulative average. Default: 0.1.
            affine: Learn normalization scale and offset. Default: True.
            track_running_stats: Keep running mean/variance for evaluation. Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            num_groups: Positive normalization group count dividing num_channels. Required.
            num_channels: Expected channel count for group normalization. Required.
            eps: Positive normalization stabilizer. Default: 1e-05.
            affine: Learn normalization scale and offset. Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Required.
            stride: Spatial sampling step; None in pooling uses the kernel size. Default: None.
            padding: Nonnegative spatial padding; pooling limits still apply. Default: 0.
            dilation: Positive spacing between filter/window taps. Default: 1.
            ceil_mode: Use the native ceil-window pooling rule instead of floor. Default: False.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            kernel_size: Positive spatial filter/window size, as an integer or H/W pair. Required.
            stride: Spatial sampling step; None in pooling uses the kernel size. Default: None.
            padding: Nonnegative spatial padding; pooling limits still apply. Default: 0.
            ceil_mode: Use the native ceil-window pooling rule instead of floor. Default: False.
            count_include_pad: Include padded positions in the average divisor. Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            output_size: Positive target spatial H/W size. Required.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
            builder.adaptive_avg_pool2d("node", ["image"], output_size=2)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        Graph topology: hyso docs show models.graph-builder --lang en
        """
        return self.add(node_id, 'adaptive_avg_pool2d', inputs, output_size=output_size)

    def dropout(self, node_id: str, inputs: list[str], *, p: float = 0.5) -> GraphModelBuilder:
        """Append elementwise dropout; evaluation leaves values unchanged.

        Delegates to add(); Registry defaults and native validation are unchanged.
        This configures a node without building or training a model.

        Args:
            node_id: Unique stable node name; never generated automatically.
            inputs: Ordered graph input/node references; topology is explicit.
            p: Element drop probability in [0, 1]; active only during training. Default: 0.5.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: The existing Registry or builder rejects a value or configuration.
            TypeError: A required argument is missing or an unsupported keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            stride: stride Default: 1.
            norm: Convolution normalization; group uses largest divisor <=8. Default: 'batch'.
            attention: None or {name, params} for a registered attention with matching output channels. Default: None.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            reduction: reduction Default: 8.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            reduction: reduction Default: 16.
            kernel_size: kernel size Default: 7.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            kernel_size: kernel size Default: 3.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            reduction: reduction Default: 32.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.coordinate("node", ["image"], channels=channels)

        HYSo Guide: components.attention
        Offline: hyso docs show components.attention --lang en
        Open: hyso docs open components.attention --lang en
        """
        return self.add(node_id, 'coordinate', inputs, channels=channels, reduction=reduction)

    def hardswish(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered hardswish node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.hardswish("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'hardswish', inputs)

    def identity(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered identity node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            groups: groups Default: 2.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            p: Probability of dropping an element/path during training. Default: 0.1.
            block_size: Spatial side length of the dropped block. Default: 3.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            stride: stride Default: 1.
            bias: bias Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            depth_multiplier: Depthwise filters per input channel. Default: 1.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            depth_multiplier: Depthwise filters per input channel. Default: 1.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            padding: padding Default: 1.
            dilation: dilation Default: 1.
            bias: bias Default: True.
            groups: groups Default: 1.
            output_padding: Extra output size on one side of each transposed-convolution axis. Default: 0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            eps: Small numerical stabilizer for normalization. Default: 1e-05.
            affine: Learn a normalization scale and offset. Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            num_features: num features Required.
            eps: Small numerical stabilizer for normalization. Default: 1e-05.
            momentum: Running-statistic update factor in normalization. Default: 0.1.
            affine: Learn a normalization scale and offset. Default: False.
            track_running_stats: Maintain running mean/variance for evaluation. Default: False.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            negative_slope: negative slope Default: 0.01.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            num_parameters: num parameters Default: 1.
            init: init Default: 0.25.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.prelu("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'prelu', inputs, num_parameters=num_parameters, init=init)

    def mish(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered mish node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.mish("node", ["image"])

        HYSo Guide: components.activation
        Offline: hyso docs show components.activation --lang en
        Open: hyso docs open components.activation --lang en
        """
        return self.add(node_id, 'mish', inputs)

    def hardsigmoid(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered hardsigmoid node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            output_size: output size Required.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.adaptive_max_pool2d("node", ["image"], output_size=output_size)

        HYSo Guide: components.pooling
        Offline: hyso docs show components.pooling --lang en
        Open: hyso docs open components.pooling --lang en
        """
        return self.add(node_id, 'adaptive_max_pool2d', inputs, output_size=output_size)

    def global_avg_pool2d(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered global_avg_pool2d node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.
            groups: groups Default: 1.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            expansion_ratio: Internal positive integer channel expansion factor. Default: 2.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_channels: in channels Required.
            out_channels: out channels Required.
            intermediate_channels: Internal feature width between projections. Required.
            kernel_size: kernel size Default: 3.
            stride: stride Default: 1.
            dilation: dilation Default: 1.
            normalization: Closed normalization choice for the residual path. Default: 'batch'.
            activation: Supported activation family. Default: 'silu'.
            attention: attention Default: 'none'.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.
            drop_path: drop path Default: 0.0.
            gate_init: Initial residual gate value in [0, 1]. Default: 1.0.
            learnable_gate: Make the residual gate trainable. Default: True.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            kernel_size: kernel size Default: 7.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            reduction: reduction Default: 8.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            embedding_dim: embedding dim Default: None.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            channels: channels Required.
            embedding_dim: embedding dim Default: None.
            num_heads: num heads Default: 4.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            p: Probability of dropping an element/path during training. Default: 0.5.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.dropout2d("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add(node_id, 'dropout2d', inputs, p=p)

    def drop_path(self, node_id: str, inputs: list[str], *, p: float = 0.0) -> GraphModelBuilder:
        """Append the registered drop_path node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            p: Probability of dropping an element/path during training. Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.drop_path("node", ["image"])

        HYSo Guide: components.regularization
        Offline: hyso docs show components.regularization --lang en
        Open: hyso docs open components.regularization --lang en
        """
        return self.add(node_id, 'drop_path', inputs, p=p)

    def gaussian_noise(self, node_id: str, inputs: list[str], *, std: float = 0.1) -> GraphModelBuilder:
        """Append the registered gaussian_noise node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            std: Standard deviation of sampled noise. Default: 0.1.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            output_size: output size Default: None.
            scale_factor: scale factor Default: None.
            mode: mode Default: 'nearest'.
            align_corners: align corners Default: None.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            upscale_factor: Positive spatial upscaling factor; channels must be divisible by its square. Required.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.pixel_shuffle("node", ["image"], upscale_factor=upscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'pixel_shuffle', inputs, upscale_factor=upscale_factor)

    def pixel_unshuffle(self, node_id: str, inputs: list[str], *, downscale_factor: int) -> GraphModelBuilder:
        """Append the registered pixel_unshuffle node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            downscale_factor: Positive spatial downscaling factor dividing both spatial dimensions. Required.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
            builder.pixel_unshuffle("node", ["image"], downscale_factor=downscale_factor)

        HYSo Guide: components.sampling
        Offline: hyso docs show components.sampling --lang en
        Open: hyso docs open components.sampling --lang en
        """
        return self.add(node_id, 'pixel_unshuffle', inputs, downscale_factor=downscale_factor)

    def flatten(self, node_id: str, inputs: list[str]) -> GraphModelBuilder:
        """Append the registered flatten node via add().

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            dims: Explicit axis permutation/order. Default: (0, 1, 3, 2).

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            shape: shape Required.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_features: in features Required.
            pooling: Head pooling: avg, max, or none; none requires compatible flat features. Default: 'avg'.
            num_classes: num classes Required.
            dropout: Drop probability; this regularization is active only in training mode. Default: 0.0.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_features: in features Required.
            pooling: Head pooling: avg, max, or none; none requires compatible flat features. Default: 'avg'.
            output_features: output features Required.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Available in commercial and development profiles.
        No weights, forward pass or training are requested.

        Args:
            node_id: Unique node name.
            inputs: Ordered explicit graph input/node references.
            in_features: in features Required.
            pooling: Head pooling: avg, max, or none; none requires compatible flat features. Default: 'avg'.
            embedding_dim: embedding dim Required.
            normalize: Apply the head's supported safe L2 embedding normalization. Default: False.
            eps: Small numerical stabilizer for normalization. Default: 1e-12.

        Returns:
            This GraphModelBuilder for chaining.

        Raises:
            ModelConfigError: Native Registry, profile, shape or configuration validation fails.
            TypeError: A required argument is missing or an unknown keyword is supplied.

        Example:
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

        Example:
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

        Example:
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

        Example:
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

        Example:
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

    def add(self,node_id: 'str',component: 'str',inputs: 'list[str]',**params: '_typing.Any') -> 'GraphModelBuilder':
        """Append a validated component and its ordered input references to the graph.

        Args:
            node_id (str): Unique stable Graph node name, distinct from graph inputs. Required.
            component (str): Closed registered component or supported Graph fusion name. Required.
            inputs (list[str]): Ordered list of 1..8 existing Graph input/node reference strings. Required.
            **params (_typing.Any): Component keyword parameters; registered defaults apply to omitted optional keys.

        Returns:
            This builder after the existing component/reference validation.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
            # Caller-supplied values; explicitly requested operations only.
            builder.add("stem", "conv2d", ["image"], in_channels=1, out_channels=4, kernel_size=3, padding=1)

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
            **references (str): 1..8 output_name=node_id pairs; selecting an output does not prune other nodes.

        Returns:
            This builder with the explicit named outputs configured; other graph nodes are not pruned.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
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
            No additional validation for a valid object; corrupt retained state or native copy/serialization failures propagate.

        Example:
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
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained state. Required.

        Returns:
            A validated builder/configuration object; no executable names are imported.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
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
        """Validate graph structure, infer shapes and estimate bounded resource needs without constructing a model.

        Args:
            config (dict[str, _typing.Any]): Plain versioned configuration; never executable code or trained state. Required.
            batch_size (int): Positive batch-size setting or metadata assumption; no implicit batch search. Default: 1.
            activation_limit (int): Advisory per-tensor element threshold used by metadata preflight, not a physical-memory quota. Default: 16777216.

        Returns:
            Metadata validation, shapes, diagnostics and advisory resource estimates; no model is built.

        Raises:
            ValueError, TypeError: Native configuration, shape or input validation rejects the request.
            RuntimeError: A requested native construction/forward operation fails.

        Example:
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

        Example:
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

        Example:
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
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.
            overwrite (bool): Whether an existing output file may be replaced explicitly; inputs remain protected. Default: False.

        Returns:
            None after the native file export succeeds.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
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
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is executed. Required.

        Returns:
            A validated GraphModelBuilder instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Example:
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
