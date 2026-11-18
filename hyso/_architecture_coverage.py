"""Audit policy v2: separate applicability, ruleset scope and missing evidence.

The v1 engine and its wire contract stay intact. Shape, inventory and alias
diagnostics are reused; only coverage semantics are explicitly versioned here.
"""
from collections import Counter, defaultdict

from .analysis._codec import digest, plain
from .structure import ModelStructureError
from ._architecture_rules import _MAX_ENTRIES, _POLICY, _LIMITATIONS, build_audit


_POLICY_V2 = dict(_POLICY, id='registered_static_architecture_v2', version=2,
    rules=[dict(r) for r in _POLICY['rules'][:-1]] + [
        dict(id='coverage_gaps', version=2,
             scope='native_evidence_required_by_applicable_shape_inventory_or_alias_checks'),
        dict(id='coverage_classification', version=2,
             scope='per_rule_applicability_and_unassessed_registered_areas')])
_LIMITATIONS_V2 = _LIMITATIONS + (
    'NOT_APPLICABLE describes a particular rule/type relation, not a model correctness verdict.',
    'UNASSESSED_BY_RULESET records areas outside the implemented checks without asserting missing native evidence.',
    'COVERAGE_GAP denotes native evidence needed by an applicable shape, inventory or storage-alias assessment.',
    'Coverage without evidence gaps is still scoped and may contain unassessed areas; it is not full architecture acceptance.',
)
_REQUIRED = dict(linear_shapes={'in_features', 'out_features', 'bias'},
    convolution_shapes={'in_channels', 'out_channels', 'groups', 'kernel_size', 'bias'})
_OUTSIDE_RULESET = {'STATIC_SHAPE_RULE_NOT_IMPLEMENTED', 'TRANSPOSED_CONV_RULE_NOT_IMPLEMENTED',
                    'CUSTOM_MODULE_SEMANTICS_UNSUPPORTED'}


def _fact(source, field, status='native'):
    value = source
    for key in field.split('.'):
        value = value[int(key)] if type(value) is list else value[key]
    return dict(field=field, status=status, value=value)


def _entry_limit(data, limit):
    # A separate v2 output bound includes applicability/area records and their
    # nested observations. The retained source already has its snapshot bound.
    stack = [v for k, v in data.items() if k != 'source_snapshot']
    count = 0
    while stack:
        value = stack.pop()
        if type(value) in (dict, list):
            count += 1
            if count > limit: raise ModelStructureError('architecture_audit_entry_limit')
            stack.extend(value.values() if type(value) is dict else value)


def build_audit_v2(source, max_entries=_MAX_ENTRIES):
    data = build_audit(source, max_entries)
    modules = {m['module_id']: (i, m) for i, m in enumerate(source['modules'])}
    assessments = data['coverage']['module_assessments']
    by_module = {a['module_id']: a for a in assessments}
    applicability, areas, findings, gaps = [], [], [], []

    def area(mid, path, name, reason, facts):
        areas.append(dict(module_id=mid, path=path, area=name, status='UNASSESSED_BY_RULESET',
            reason=reason, observations=facts, source_snapshot_sha256=source['content_sha256'],
            limitation='Outside the implemented checks; no consistency, defect or missing-tensor conclusion is established.'))

    for a in assessments:
        i, m = modules[a['module_id']]
        # A known invalid shape prerequisite makes the dependent expectation
        # undefined even when the parameter registration is absent. V1 checks
        # absence first; v2 must not call that unnecessary shape evidence a gap.
        invalid = {c['code'] for c in a['checks'] if c['status'] == 'inconsistent'
                   and (c['code'].endswith('_domain') or c['code'].endswith('_group_divisibility'))}
        for c in a['checks']:
            if c['status'] == 'insufficient_evidence' and (
                    c['code'] == 'weight_shape' and invalid or
                    c['code'] == 'bias_shape' and invalid & {'out_features_domain', 'out_channels_domain'}):
                c['status'] = 'not_applicable'
        facts = [_fact(source, 'modules.%d.qualified_type' % i), _fact(source, 'modules.%d.category' % i)]
        for rule in ('linear_shapes', 'convolution_shapes'):
            applicable = a['rule_id'] == rule
            reason = ('exact_standard_type' if applicable else 'custom_semantics_not_inferred'
                if m['category'] == 'custom' else 'transposed_layout_excluded'
                if m['category'].startswith('conv_transpose') else 'module_type_outside_this_rule')
            applicability.append(dict(module_id=a['module_id'], path=a['path'], rule_id=rule,
                rule_version=1, status='APPLICABLE' if applicable else 'NOT_APPLICABLE',
                reason=reason, observations=facts, source_snapshot_sha256=source['content_sha256']))
        if a['rule_id'] is None:
            a['status'] = 'unassessed_by_ruleset'
            area(a['module_id'], a['path'], 'registered_module_shape_semantics',
                 'custom_semantics_not_inferred' if m['category'] == 'custom' else
                 'transposed_shape_rule_not_implemented' if m['category'].startswith('conv_transpose') else
                 'no_shape_rule_in_current_catalog', facts)

    for f in data['findings']:
        if f['code'] in _OUTSIDE_RULESET:
            continue  # represented by applicability and unassessed area records
        if f['code'] == 'REQUIRED_PARAMETER_METADATA_MISSING':
            parameter = f['related_paths'][0].rsplit('.', 1)[-1]
            if any(c['code'] == parameter + '_shape' and c['status'] == 'not_applicable'
                   for c in by_module[f['module_id']]['checks']):
                continue
        if f['code'] == 'STATIC_CONFIGURATION_UNAVAILABLE':
            a = by_module[f['module_id']]
            # The v1 fact addresses the complete retained configuration field.
            field = f['observations'][0]['value']['name']
            if field not in _REQUIRED.get(a['rule_id'], set()):
                area(f['module_id'], f['path'], 'configuration.' + field,
                     'field_not_used_by_implemented_checks', f['observations'])
                continue
        if f['classification'] == 'COVERAGE_GAP':
            f['rule_version'] = 2
            identity = {k: f[k] for k in ('source_snapshot_sha256', 'rule_id', 'rule_version',
                                         'code', 'module_id', 'tensor_id', 'path')}
            identity['fields'] = [v['field'] for v in f['observations']]
            f['finding_id'] = digest(identity)
        findings.append(f)

    owners = defaultdict(list)
    owner_sets = defaultdict(set)
    for m in source['modules']:
        for kind in ('parameters', 'buffers'):
            for slot in m[kind]:
                if slot['tensor_id'] is not None and m['module_id'] not in owner_sets[slot['tensor_id']]:
                    owners[slot['tensor_id']].append(m['module_id'])
                    owner_sets[slot['tensor_id']].add(m['module_id'])
    for f in findings:
        if f['classification'] != 'COVERAGE_GAP': continue
        mid = f['module_id']
        required = ('registration_relationships' if f['code'] == 'STORAGE_ALIAS_UNAVAILABLE' else
                    'parameter_inventory' if f['code'] == 'TENSOR_METADATA_UNSUPPORTED' else by_module[mid]['rule_id'])
        gaps.append(dict(finding_id=f['finding_id'], status='COVERAGE_GAP', code=f['code'],
            required_by_rule_id=required, module_ids=[mid] if mid is not None else owners[f['tensor_id']],
            source_fields=[v['field'] for v in f['observations']]))

    states = Counter(a['status'] for a in assessments)
    kinds = Counter(f['classification'] for f in findings)
    check_states = Counter(c['status'] for a in assessments for c in a['checks'])
    partial_ids = [a['module_id'] for a in assessments
                   if any(c['status'] == 'insufficient_evidence' for c in a['checks'])]
    gap_modules = {mid for g in gaps for mid in g['module_ids']}
    coverage = dict(total_unique_modules=len(source['modules']), registered_module_paths=len(source['paths']),
        shape_rule_eligible_modules=sum(a['rule_id'] is not None for a in assessments),
        examined_supported_modules=states['consistent'] + states['inconsistent'],
        consistent_modules=states['consistent'], inconsistent_modules=states['inconsistent'],
        insufficient_evidence_modules=states['insufficient_evidence'],
        unassessed_by_ruleset_modules=states['unassessed_by_ruleset'],
        unassessed_module_ids=[a['module_id'] for a in assessments if a['status'] == 'unassessed_by_ruleset'],
        module_ids_with_insufficient_checks=partial_ids,
        module_ids_with_evidence_gaps=[m['module_id'] for m in source['modules'] if m['module_id'] in gap_modules],
        not_applicable_rule_evaluations=sum(r['status'] == 'NOT_APPLICABLE' for r in applicability),
        applicable_rule_evaluations=sum(r['status'] == 'APPLICABLE' for r in applicability),
        check_counts={k: check_states[k] for k in ('consistent', 'inconsistent', 'insufficient_evidence', 'not_applicable')},
        evidence_gap_count=len(gaps), unassessed_area_count=len(areas),
        status='evidence_gaps_present' if gaps else 'scoped_static_evidence_assessed',
        rule_applicability=applicability, unassessed_areas=areas, evidence_gaps=gaps,
        module_assessments=assessments,
        rule_coverage=[dict(rule_id=r['id'], rule_version=r['version'], scope=r['scope'],
            finding_count=sum(f['rule_id'] == r['id'] for f in findings),
            module_assessment_count=sum(a['rule_id'] == r['id'] for a in assessments)) for r in _POLICY_V2['rules']])
    data.update(schema_version=2, audit_policy=_POLICY_V2, findings=findings, coverage=coverage,
        summary=dict(total_findings=len(findings), finding_counts={k: kinds[k] for k in
                     ('STATIC_INCONSISTENCY', 'OBSERVATION', 'COVERAGE_GAP')}), limitations=_LIMITATIONS_V2)
    data.pop('content_sha256')
    data = plain(data)
    _entry_limit(data, max_entries)
    data['content_sha256'] = digest(data)
    return data
