"""P26 schema-3 structure binding. Metadata of the actual isolated receiver.

Live capture happens only in evaluate_classification. Reader consistency is not
cryptographic attestation of a live model, its source code or an execution graph.
"""
from .analysis._codec import canonical

_METHOD = 'p26_isolated_forward_receiver_pre_post_v1'
_SCOPE = 'registered_structure_of_forward_receiver_at_evaluation_boundaries'
_PHASES = ('after_candidate_eval_before_loader_iteration',
           'after_successful_loader_iteration_before_result')
_STREAM_FIELDS = ('method', 'scope', 'sample_count', 'inputs_sha256', 'targets_sha256', 'paired_sha256')


def _payload(snapshot, evaluation_id, source, evidence, accounting):
    d = snapshot.to_dict()
    state = source['model_state_sha256']
    sha = snapshot.content_sha256
    checkpoint = evidence['checkpoint']; stream = evidence['evaluated_stream']
    return dict(format='hyso.evaluation.structure_binding', schema_version=1, status='native',
        method=_METHOD, scope=_SCOPE, receiver='isolated_candidate_called_for_forward',
        original_structure_relation='not_captured_or_equivalence_not_certified',
        evaluation_id=evaluation_id, evaluated_model_type=d['modules'][0]['qualified_type'],
        structure_sha256=sha, pre_structure_sha256=sha, post_structure_sha256=sha,
        model_state_method=evidence['model_state']['method'], model_state_sha256=state,
        pre_model_state_sha256=state, post_model_state_sha256=state,
        checkpoint=dict(status=checkpoint['status'], sha256=checkpoint['sha256']),
        evaluated_stream={k: stream[k] for k in _STREAM_FIELDS},
        forward_batches=len(accounting['batch_sizes']), observed_phases=list(_PHASES), snapshot=d)


def build_structure_binding(observations, *, evaluation_id, source, evidence, accounting, provenance):
    """Private writer, supplied with actual pre/post observations by P26 itself."""
    from .structure import ModelStructureSnapshot
    before, after, pre_state, post_state = observations
    if type(before) is not ModelStructureSnapshot or type(after) is not ModelStructureSnapshot:
        raise ValueError('typed structure observations required')
    if before.content_sha256 != after.content_sha256:
        raise ValueError('evaluated model structure changed during evaluation')
    if pre_state != post_state or pre_state != source['model_state_sha256']:
        raise ValueError('evaluated model state does not match structure binding')
    binding = _payload(before, evaluation_id, source, evidence, accounting)
    validate_structure_binding(binding, evaluation_id=evaluation_id, source=source,
        evidence=evidence, accounting=accounting, provenance=provenance)
    return binding


def validate_structure_binding(binding, *, evaluation_id, source, evidence, accounting, provenance):
    """Revalidate snapshot, boundary equality and all native peer references.

    A valid imported assertion is internally consistent; it cannot recreate or
    authenticate the live process/object association made by the original writer.
    """
    from .structure import ModelStructureSnapshot
    try:
        if source['kind'] != 'model_inference' or accounting['expected_samples'] < 1:
            raise ValueError('structure binding requires successful nonempty model inference')
        snapshot = ModelStructureSnapshot(binding['snapshot'])
        if snapshot.to_dict()['producer']['torch_version'] != provenance['torch']:
            raise ValueError('structure/evaluation Torch provenance mismatch')
        expected = _payload(snapshot, evaluation_id, source, evidence, accounting)
        # Canonical equality also rejects bool/int substitutions and extra fields.
        if canonical(binding) != canonical(expected):
            raise ValueError('structure binding does not match native evaluation evidence')
    except (TypeError, KeyError, ValueError, AttributeError, RuntimeError, OverflowError) as exc:
        raise ValueError('invalid evaluation structure binding') from exc


def resolved_structure_context(binding):
    """Small native-context projection; original snapshot remains in the payload."""
    return dict(status=binding['status'], format=binding['format'], schema_version=binding['schema_version'],
        method=binding['method'], scope=binding['scope'], evaluation_id=binding['evaluation_id'],
        evaluated_model_type=binding['evaluated_model_type'], structure_sha256=binding['structure_sha256'],
        snapshot_path='structure_binding.snapshot', binding_path='structure_binding',
        receiver=binding['receiver'], original_structure_relation=binding['original_structure_relation'],
        pre_structure_sha256=binding['pre_structure_sha256'], post_structure_sha256=binding['post_structure_sha256'],
        model_state_sha256=binding['model_state_sha256'], checkpoint=dict(binding['checkpoint']),
        evaluated_stream=dict(binding['evaluated_stream']), forward_batches=binding['forward_batches'],
        snapshot_limitations_path='structure_binding.snapshot.limitations', observed_phases=list(binding['observed_phases']))
