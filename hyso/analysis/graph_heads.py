"""Two explicit classification heads composed from the existing CPU evaluator.

The envelope binds retained selections, not a new native performance receipt.
Each head retains its own P26 labels, stream, metrics and declared context.
"""
from dataclasses import dataclass
import json
import weakref

import torch
from torch.utils.data import DataLoader, TensorDataset, SequentialSampler, BatchSampler
from torch.utils.data._utils.collate import default_collate

from ._codec import canonical, digest, plain, fingerprint
from .graph_evidence import (GraphEvaluationResult, GraphEvidenceError, _Export, _body,
    _context, _receipt_statement, _contract_statement, _remember, _EVALUATIONS, evaluate_graph_classification)
from .._evaluation_evidence import _graph_contract, model_hash
from ..checkpoint import graph_checkpoint_evidence
from ..evaluation import _metadata
from ..models.graph import GraphModel, GraphModelBuilder
from ..models.output import GraphTensorOutput
from ..training.data import NamedTargetDataset


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
    """Separate existing GraphEvaluationResults; no aggregate accuracy or receipt."""
    _payload: str

    def __init__(self, data):
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
            shared.append((stream['sample_count'],stream['inputs_sha256'],evaluation['source']['model_state_sha256']))
        if shared[0] != shared[1] or len({r['spec']['target'] for r in d['heads'].values()}) != 2:
            raise GraphEvidenceError('heads_must_preserve_shared_input_state_and_distinct_targets')
        object.__setattr__(self, '_payload', canonical(d))

    def to_dict(self):
        d = json.loads(self._payload)
        d['verification'] = GraphEvaluationResult._verification(self, d)
        return d

    def head(self, output):
        """Return this exact retained head; importing never restores live trust."""
        d = json.loads(self._payload)
        if output not in d['heads']:
            raise GraphEvidenceError('selected_head_missing')
        return GraphEvaluationResult(d['heads'][output]['result'])

    def analysis_source(self, model_id, *, head):
        return self.head(head).analysis_source(model_id)


def evaluate_graph_heads(model, loader, *, heads, capture_structure=False,
                         expected_checkpoint_sha256=None):
    """Explicit two-head CPU fp32 evaluation, using the original evaluator twice.

    Requires the closed single-input NamedTargetDataset profile. Each head spec
    names its target, task, ordered class_names, split, dataset_id and
    preprocessing_id (unknown identities may explicitly be None). No report,
    advice or comparison calls this execution function implicitly.
    """
    if type(model) is not GraphModel or type(heads) is not dict or len(heads) != 2:
        raise GraphEvidenceError('raw_GraphModel_and_two_explicit_heads_required')
    contract = _graph_contract(model)
    if len(contract['configuration']['inputs']) != 1 or contract['dtype'] != 'torch.float32':
        raise GraphEvidenceError('two_head_evaluation_requires_single_input_CPU_fp32')
    if (type(loader) is not DataLoader or type(loader.dataset) is not NamedTargetDataset
            or type(loader.sampler) is not SequentialSampler or type(loader.batch_sampler) is not BatchSampler
            or loader.num_workers != 0 or loader.drop_last or loader.pin_memory
            or loader.collate_fn is not default_collate or loader.generator is not None):
        raise GraphEvidenceError('two_head_evaluation_requires_standard_sequential_NamedTargetDataset_loader')
    dataset = loader.dataset
    if (set(vars(dataset)) != {'target_names','tensors'} or type(dataset.target_names) is not tuple
            or dataset.target_names != tuple(sorted(dataset.target_names))
            or type(dataset.tensors) is not tuple or len(dataset.tensors) != 3
            or loader.batch_sampler.sampler is not loader.sampler
            or loader.batch_sampler.batch_size != loader.batch_size
            or loader.batch_sampler.drop_last != loader.drop_last):
        raise GraphEvidenceError('closed_NamedTargetDataset_required')
    # Revalidate mutable dataset fields through its closed constructor.
    NamedTargetDataset(dataset.tensors[0], dict(zip(dataset.target_names,dataset.tensors[1:])))
    specs = {name:_spec(value) for name,value in heads.items()}
    if {s['target'] for s in specs.values()} != set(dataset.target_names):
        raise GraphEvidenceError('explicit_head_target_mapping_mismatch')
    inputs = dataset.tensors[0]
    if inputs.dtype != torch.float32 or list(inputs.shape[1:]) != next(iter(contract['configuration']['inputs'].values()))['shape']:
        raise GraphEvidenceError('head_input_contract_mismatch')
    receipt = graph_checkpoint_evidence(model)
    if expected_checkpoint_sha256 is not None and (receipt['status'] != 'native'
            or receipt['checkpoint_sha256'] != expected_checkpoint_sha256):
        raise GraphEvidenceError('expected_source_identity_mismatch:checkpoint_sha256')
    state = model_hash(model); rows = {}
    for name,spec in specs.items():
        adapter = GraphTensorOutput(model,name,task=spec['task'])
        target = dataset.tensors[1+dataset.target_names.index(spec['target'])]
        selected = DataLoader(TensorDataset(inputs,target),batch_size=loader.batch_size,shuffle=False)
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
