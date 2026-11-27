"""Two explicit classification heads composed from the existing CPU evaluator.

The envelope binds retained selections, not a new native performance receipt.
Each head retains its own P26 labels, stream, metrics and declared context.
"""
import typing as _typing
from dataclasses import dataclass
import json
import weakref

import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, BatchSampler
from torch.utils.data._utils.collate import default_collate

from ._codec import canonical, digest, plain, fingerprint
from .graph_evidence import (GraphEvaluationResult, GraphEvidenceError, _Export, _body,
    _context, _receipt_statement, _contract_statement, _remember, _EVALUATIONS, evaluate_graph_classification)
from .._evaluation_evidence import _graph_contract, model_hash, STREAM_METHOD, NAMED_STREAM_METHOD
from ..checkpoint import graph_checkpoint_evidence
from ..evaluation import _metadata
from ..models.graph import GraphModel, GraphModelBuilder
from ..models.output import GraphTensorOutput
from ..training.data import NamedTargetDataset, NamedTensorDataset

from .model_comparison import ModelComparisonResult, ModelComparisonSource


class _RequiredGraphHeadSpec(_typing.TypedDict):
    target: str
    task: _typing.Literal['multiclass', 'binary']
    class_names: _typing.Sequence[str]
    split: str
    dataset_id: str
    preprocessing_id: str


class GraphHeadSpec(_RequiredGraphHeadSpec, total=False):
    """Typed plain dictionary for one existing named P26 classification head.

    target, split, dataset_id and preprocessing_id are caller-owned identifiers.
    task and class_names must satisfy the existing evaluator's label contract.
    zero_division is optional and defaults to zero in the native validator.

    Args:
        zero_division (int): Value used for undefined precision/recall; use the evaluator-supported
            choices. Required.

    Returns:
        A plain dictionary; creating it does not evaluate or modify a graph.
    Raises:
        No runtime validation here; evaluate_graph_heads validates the supplied data.
    Examples:
        head = GraphHeadSpec(target='label', task='multiclass', class_names=['a', 'b'],
                             split='test', dataset_id='sample', preprocessing_id='v1')

    HYSo Guide: analysis.p26
    Offline: hyso docs show analysis.p26 --lang en
    Open: hyso docs open analysis.p26 --lang en
    """
    zero_division: int


def _spec(value):
    value = plain(value)
    fields = {'target', 'task', 'class_names', 'split', 'dataset_id', 'preprocessing_id', 'zero_division'}
    if type(value) is not dict or set(value) - fields or fields - {'zero_division'} - set(value):
        raise GraphEvidenceError('explicit_head_target_task_labels_split_and_context_required')
    if type(value['target']) is not str or not value['target']:
        raise GraphEvidenceError('explicit_target_name_required')
    value.setdefault('zero_division', 0)
    _metadata(value['task'], value['class_names'], value['split'], value['zero_division'])
    _context({k: value[k] for k in ('dataset_id', 'split', 'preprocessing_id')}, value['dataset_id'], value['split'])
    return value


@dataclass(frozen=True, init=False)
class MultiHeadGraphEvaluationResult(_Export):
    """Separate existing GraphEvaluationResults; no aggregate accuracy or receipt.

    Args:
        data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
            version/schema validator. Required.

    Returns:
        An instance of MultiHeadGraphEvaluationResult with the documented native validation and
        defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        MultiHeadGraphEvaluationResult(data)

    HYSo Guide: analysis.p26
    Offline: hyso docs show analysis.p26 --lang en
    Open: hyso docs open analysis.p26 --lang en
    """
    _payload: str

    def __init__(self, data: 'dict[str, _typing.Any]') -> 'None':
        """Initialize MultiHeadGraphEvaluationResult with the supplied settings.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
                version/schema validator. Required.

        Returns:
            None. Initialize this object; use its explicit operation methods to request work.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            MultiHeadGraphEvaluationResult(data)

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        d = _body(data, 'hyso.graph_multi_head_evaluation',
            {'graph_contract', 'model_state_sha256', 'source_receipt', 'heads'})
        fingerprint(d['model_state_sha256'])
        contract = d['graph_contract']
        _contract_statement(contract)
        if contract['adapter_output'] is not None or not GraphModelBuilder.preflight(contract['configuration'])['valid']:
            raise GraphEvidenceError('raw_graph_contract_required')
        _receipt_statement(d['source_receipt'])
        if d['source_receipt']['status'] == 'native' and (
                d['source_receipt']['contract'] != contract
                or d['source_receipt']['model_state_sha256'] != d['model_state_sha256']):
            raise GraphEvidenceError('graph_source_state_mismatch')
        if type(d['heads']) is not dict or len(d['heads']) != 2:
            raise GraphEvidenceError('exactly_two_explicit_heads_required')
        config = contract['configuration']; nodes = {n['id']:n for n in config['nodes']}
        shared = []
        for name, row in d['heads'].items():
            if type(name) is not str or type(row) is not dict or set(row) != {'spec', 'result'}:
                raise GraphEvidenceError('invalid_head_fields')
            spec = _spec(row['spec'])
            if type(row['result']) is not dict or any(k in row['result'] for k in ('verification','relation')):
                raise GraphEvidenceError('per_head_process_local_views_must_not_be_persisted')
            node = nodes.get(config['outputs'].get(name))
            if (node is None or node['component'] != 'classification_head'
                    or node['params']['num_classes'] != (1 if spec['task']=='binary' else len(spec['class_names']))):
                raise GraphEvidenceError('head_output_label_contract_mismatch')
            result = GraphEvaluationResult(row['result']).to_dict()
            evaluation = result['evaluation']
            if evaluation['configuration'] != _metadata(spec['task'],spec['class_names'],spec['split'],spec['zero_division']):
                raise GraphEvidenceError('head_label_protocol_mismatch')
            if result['context'] != {k:spec[k] for k in ('dataset_id','split','preprocessing_id')}:
                raise GraphEvidenceError('head_declared_context_mismatch')
            # Per-head adapters have no transferred raw-Graph checkpoint receipt.
            if result['source_receipt']['status'] != 'missing':
                raise GraphEvidenceError('raw_graph_receipt_must_not_transfer_to_head_adapter')
            stream = evaluation['producer_evidence']['evaluated_stream']
            input_names = sorted(config['inputs'])
            if (len(input_names) == 2 and (stream['method'] != NAMED_STREAM_METHOD or stream.get('input_names') != input_names)
                    or len(input_names) == 1 and stream['method'] != STREAM_METHOD):
                raise GraphEvidenceError('head_native_input_contract_mismatch')
            shared.append((stream['sample_count'],stream['inputs_sha256'],evaluation['source']['model_state_sha256']))
        if shared[0] != shared[1] or len({r['spec']['target'] for r in d['heads'].values()}) != 2:
            raise GraphEvidenceError('heads_must_preserve_shared_input_state_and_distinct_targets')
        object.__setattr__(self, '_payload', canonical(d))

    def to_dict(self) -> 'dict[str, _typing.Any]':
        """Return a detached plain-data view of the stored state.

        Returns:
            A detached plain-data view of this validated object; native units and unavailable values are
            retained.

        Raises:
            No additional validation for a valid object; corrupt retained state or native copy/serialization
            failures propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_dict()

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        d = json.loads(self._payload)
        d['verification'] = GraphEvaluationResult._verification(self, d)
        return d

    def head(self, output: '_typing.Any') -> 'GraphEvaluationResult':
        """Return this exact retained head; importing never restores live trust.

        Args:
            output (_typing.Any): Explicit output payload or destination required by this API. Required.

        Returns:
            The retained head value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.head(output)

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        d = json.loads(self._payload)
        if output not in d['heads']:
            raise GraphEvidenceError('selected_head_missing')
        return GraphEvaluationResult(d['heads'][output]['result'])

    def analysis_source(self, model_id: '_typing.Any', *, head: '_typing.Any') -> 'ModelComparisonSource':
        """Create a typed retained source from this validated named-head evaluation result.

        Args:
            model_id (_typing.Any): Stable model label bound to this result/context. Required.
            head (_typing.Any): Explicit prediction-head kind, such as classification, regression or
                embedding. Required.

        Returns:
            The retained analysis source value, without recomputing measurements.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.analysis_source(model_id, head=head)

        HYSo Guide: analysis.p26
        Offline: hyso docs show analysis.p26 --lang en
        Open: hyso docs open analysis.p26 --lang en
        """
        return self.head(head).analysis_source(model_id)


def evaluate_graph_heads(model: '_typing.Any', loader: '_typing.Any', *, heads: dict[str, GraphHeadSpec], capture_structure: 'bool'=False,
                         expected_checkpoint_sha256: '_typing.Any'=None) -> 'MultiHeadGraphEvaluationResult':
    """Explicit two-head CPU fp32 evaluation, using the original evaluator twice.

    Requires the closed one- or two-input NamedTargetDataset profile. Each head spec
    names its target, task, ordered class_names, split, dataset_id and
    preprocessing_id (unknown identities may explicitly be None). No report,
    advice or comparison calls this execution function implicitly.

    Args:
        model (_typing.Any): Caller-owned model; device, dtype and task must match the native operation.
            Required.
        loader (_typing.Any): Caller-owned finite DataLoader with the required input/target batch
            format. Required.
        heads (_typing.Any): Explicit named Graph outputs and their classification evaluation contracts.
            Required.
        capture_structure (bool): Request supported structure evidence in the explicit operation.
            Default: False.
        expected_checkpoint_sha256 (_typing.Any): Exact SHA-256 digest of the expected checkpoint.
            Default: None.

    Returns:
        The retained evaluate graph heads value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        evaluate_graph_heads(model, loader, heads=heads)

    HYSo Guide: analysis.p26
    Offline: hyso docs show analysis.p26 --lang en
    Open: hyso docs open analysis.p26 --lang en
    """
    if type(model) is not GraphModel or type(heads) is not dict or len(heads) != 2:
        raise GraphEvidenceError('raw_GraphModel_and_two_explicit_heads_required')
    contract = _graph_contract(model)
    input_config = contract['configuration']['inputs']
    count = len(input_config)
    if count not in (1,2) or contract['dtype'] != 'torch.float32':
        raise GraphEvidenceError('two_head_evaluation_requires_one_or_two_inputs_CPU_fp32')
    if (type(loader) is not DataLoader or type(loader.dataset) is not NamedTargetDataset
            or type(loader.sampler) is not SequentialSampler or type(loader.batch_sampler) is not BatchSampler
            or loader.num_workers != 0 or loader.drop_last or loader.pin_memory
            or loader.collate_fn is not default_collate or loader.generator is not None):
        raise GraphEvidenceError('two_head_evaluation_requires_standard_sequential_NamedTargetDataset_loader')
    dataset = loader.dataset
    if (set(vars(dataset)) != ({'target_names','tensors','input_names'} if count == 2 else {'target_names','tensors'})
            or count == 2 and (type(dataset.input_names) is not tuple or dataset.input_names != tuple(sorted(input_config)))
            or type(dataset.target_names) is not tuple
            or dataset.target_names != tuple(sorted(dataset.target_names))
            or type(dataset.tensors) is not tuple or len(dataset.tensors) != count+2
            or loader.batch_sampler.sampler is not loader.sampler
            or loader.batch_sampler.batch_size != loader.batch_size
            or loader.batch_sampler.drop_last != loader.drop_last):
        raise GraphEvidenceError('closed_NamedTargetDataset_required')
    # Revalidate mutable dataset fields through its closed constructor.
    inputs = dict(zip(dataset.input_names,dataset.tensors[:2])) if count == 2 else dataset.tensors[0]
    NamedTargetDataset(inputs, dict(zip(dataset.target_names,dataset.tensors[count:])))
    specs = {name:_spec(value) for name,value in heads.items()}
    if {s['target'] for s in specs.values()} != set(dataset.target_names):
        raise GraphEvidenceError('explicit_head_target_mapping_mismatch')
    values = inputs if count == 2 else {next(iter(input_config)):inputs}
    if any(x.dtype != torch.float32 or list(x.shape[1:]) != input_config[name]['shape'] for name,x in values.items()):
        raise GraphEvidenceError('head_input_contract_mismatch')
    receipt = graph_checkpoint_evidence(model)
    if expected_checkpoint_sha256 is not None and (receipt['status'] != 'native'
            or receipt['checkpoint_sha256'] != expected_checkpoint_sha256):
        raise GraphEvidenceError('expected_source_identity_mismatch:checkpoint_sha256')
    state = model_hash(model); rows = {}
    for name,spec in specs.items():
        adapter = GraphTensorOutput(model,name,task=spec['task'],named_inputs=count == 2)
        target = dataset.tensors[count+dataset.target_names.index(spec['target'])]
        selected = DataLoader(NamedTensorDataset(inputs,target) if count == 2 else TensorDataset(inputs,target),
                              batch_size=loader.batch_size,shuffle=False)
        result = evaluate_graph_classification(adapter,selected,class_names=spec['class_names'],
            task=spec['task'],split=spec['split'],source_id=spec['dataset_id'],
            context={k:spec[k] for k in ('dataset_id','split','preprocessing_id')},
            zero_division=spec['zero_division'],capture_structure=capture_structure)
        rows[name] = dict(spec=spec,result=json.loads(result._payload))
    if (model_hash(model) != state or _graph_contract(model) != contract
            or graph_checkpoint_evidence(model) != receipt):
        raise GraphEvidenceError('graph_changed_during_head_evaluation')
    data = dict(format='hyso.graph_multi_head_evaluation',schema_version=1,
        graph_contract=contract,model_state_sha256=state,source_receipt=receipt,heads=rows)
    data['content_sha256'] = digest(data)
    result = MultiHeadGraphEvaluationResult(data)
    _remember(_EVALUATIONS,result,weakref.ref(model))
    return result


def compare_graph_heads(baseline: '_typing.Any', candidate: '_typing.Any', *, head_pairs: '_typing.Any', model_ids: '_typing.Any'=('A','B'),
                        features: _typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35'] | _typing.Sequence[_typing.Literal['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07', 'P08', 'P09', 'P10', 'P11', 'P12', 'P13', 'P14', 'P15', 'P16', 'P17', 'P18', 'P19', 'P20', 'P21', 'P22', 'P23', 'P24', 'P25', 'P26', 'P27', 'P28', 'P29', 'P30', 'P31', 'P32', 'P33', 'P34', 'P35']]=('P26','P12'), include_structure: 'bool'=True) -> 'dict[str, ModelComparisonResult]':
    """Compare only explicitly paired equal output names through compare_models.

    This bounded adapter does not infer semantic equivalence or rename heads.
    Matching names remain caller declarations; native label/stream/protocol and
    strict declared-context gates independently decide numeric eligibility.
    Returns separate existing ModelComparisonResults, never a model-wide winner.

    Args:
        baseline (_typing.Any): Validated retained baseline source or predictions required by this
            comparison. Required.
        candidate (_typing.Any): Validated retained candidate source or predictions required by this
            comparison. Required.
        head_pairs (_typing.Any): Explicit baseline/candidate named-output pairings. Required.
        model_ids (_typing.Any): Explicit ordered model identities for the recorded workflow. Default:
            ('A', 'B').
        features (_typing.Any): Explicit feature IDs to compare/plan; prerequisites are never silently
            executed. Default: ('P26', 'P12').
        include_structure (bool): Whether to include supported retained structure comparisons. Default:
            True.

    Returns:
        The retained compare graph heads value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        compare_graph_heads(baseline, candidate, head_pairs=head_pairs)

    HYSo Guide: analysis.p26
    Offline: hyso docs show analysis.p26 --lang en
    Open: hyso docs open analysis.p26 --lang en
    """
    from .model_comparison import compare_models
    if type(baseline) is not MultiHeadGraphEvaluationResult or type(candidate) is not MultiHeadGraphEvaluationResult:
        raise GraphEvidenceError('two_retained_multi_head_results_required')
    if (type(head_pairs) is not dict or not 1 <= len(head_pairs) <= 2
            or any(type(a) is not str or type(b) is not str or a != b for a,b in head_pairs.items())):
        raise GraphEvidenceError('explicit_equal_head_names_required;semantic_head_renaming_unsupported')
    if (type(model_ids) not in (tuple,list) or len(model_ids) != 2
            or any(type(m) is not str or not m for m in model_ids) or model_ids[0] == model_ids[1]):
        raise GraphEvidenceError('two_distinct_explicit_model_ids_required')
    return {a:compare_models(baseline.analysis_source(model_ids[0]+':'+a,head=a),
        candidate.analysis_source(model_ids[1]+':'+b,head=b),features=features,
        include_structure=include_structure,context_policy='strict') for a,b in head_pairs.items()}


def report_section(result, language: _typing.Literal['tr', 'en'], level, catalog) -> 'dict[str, _typing.Any]':
    """Use existing P26 presentation and rendering with explicit per-head labels.

    Args:
        result: Retained HYSo result/evidence record; rendering/advice does not run measurements.
            Required.
        language (_typing.Literal['tr', 'en']): Presentation language: tr, en, de or es. Technical
            keys/status codes remain stable. Required.
        level: Heading level used when embedding a report section. Required.
        catalog: Translation/metric catalog used to render stable recorded fields. Required.
    """
    from .graph_evidence import report_section as single_section, _LIMITS
    from ..reporting.report import _metric
    data = result.to_dict();metrics=[];charts=[];features=[]
    for name,row in data['heads'].items():
        head = result.head(name);part = single_section(head,language,level,catalog)
        evaluation=head.evaluation.to_dict();values=evaluation['metrics']
        scope = dict(output=name,target=row['spec']['target'],context=head.to_dict()['context'],
            evaluation_id=evaluation['evaluation_id'],source_sha256=evaluation['source_sha256'])
        for metric in part['metrics']:
            metric['key'] = name+'.'+metric['key']
            metric['label'] = name+' · '+metric['label']
            metric['scope']['head'] = scope
            metrics.append(metric)
        for chart in part['charts']:
            chart['title'] = name+' · '+chart['title'];charts.append(chart)
        features.extend(part['features'])
        for key,value,unit in [('macro_precision',values['averages']['macro']['precision'],'fraction'),
                ('macro_recall',values['averages']['macro']['recall'],'fraction'),
                ('sample_count',values['sample_count'],'sample'),
                ('confusion_matrix',values['confusion_matrix'],None),
                ('label_mapping',evaluation['configuration']['class_names'],None),
                ('evaluation_protocol',row['spec'],None)]:
            metrics.append(_metric(name+'.'+key,name+' · '+key,value,unit,'head_evaluation',
                'recorded' if value is not None else 'insufficient_evidence',scope,
                'Per-head result; no aggregate accuracy. Context identities remain declared.',_LIMITS,feature='P26'))
    for key,value in [('model_state_sha256',data['model_state_sha256']),
                      ('checkpoint_sha256',data['source_receipt']['checkpoint_sha256']),
                      ('graph_source_verification',data['verification'])]:
        metrics.append(_metric(key,key,value,None,'source_identity',
            'recorded' if value is not None else 'insufficient_evidence',{},
            'Raw Graph source association is separate from each isolated head evaluation.',_LIMITS))
    return dict(kind='graph_multi_head_evaluation',
        title='Graph head değerlendirmeleri' if language=='tr' else 'Graph head evaluations',
        raw=data,metrics=metrics,charts=charts,features=features,advisor=None,
        limits=[*_LIMITS,'no overall accuracy;head metrics and protocols remain separate'])
