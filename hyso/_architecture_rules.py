"""Closed v1 rules over validated JSON metadata. No live object inspection."""
from collections import Counter, defaultdict
from decimal import Decimal, localcontext

from .analysis._codec import digest, plain
from .structure import ModelStructureError

_LINEAR = 'torch.nn.modules.linear.Linear'
_CONVS = {'torch.nn.modules.conv.Conv%dd' % d: d for d in (1, 2, 3)}
_MAX_ENTRIES = 500000
_MAX_BYTES = 64 * 1024**2
_POLICY = dict(id='registered_static_architecture_v1', version=1,
    source_format='hyso.model_structure', source_schema_version=1,
    producer_method='registered_nn_module_metadata', producer_version=1, classification_version=1,
    rules=[
        dict(id='linear_shapes', version=1, scope='exact_standard_Linear_registered_parameter_shapes'),
        dict(id='convolution_shapes', version=1, scope='exact_standard_Conv1d_Conv2d_Conv3d_registered_parameter_shapes'),
        dict(id='parameter_inventory', version=1, scope='unique_parameter_objects_and_direct_owner_metadata'),
        dict(id='registration_relationships', version=1, scope='within_snapshot_module_tensor_and_storage_identities'),
        dict(id='coverage_gaps', version=1, scope='unassessed_semantics_configuration_tensor_and_storage_metadata'),
    ])
_LIMITATIONS = (
    'Static registered metadata only; no overall correctness, PASS/FAIL, severity score or confidence probability.',
    'Consistent means only the recorded fields checked by the named versioned rule; unassessed semantics remain explicit.',
    'Only exact standard Linear and Conv1d/2d/3d types receive shape rules; custom/subclass and transposed convolution semantics are not inferred.',
    'Registration order and module paths are not execution order or an operator graph; functional operations and dynamic forward behavior may be absent.',
    'No runtime input/output shape, latency bottleneck, FLOPs, GPU allocation, gradient, overfitting or learning diagnosis.',
    'Parameter counts and requires_grad are metadata observations, not defects or actual training behavior.',
    'Per-owner parameter shares may overlap for tied objects and must not be summed; global totals deduplicate tensor objects.',
    'Shared/tied/cyclic registrations are observations, not evidence of incorrectness or danger; storage identity does not establish byte-range overlap.',
    'Local object/storage IDs have meaning only inside the source snapshot; structure SHA is neither model-state nor checkpoint identity.',
    'Imported consistency is not cryptographic live-producer authenticity, code equivalence or evidence of an executed workload.',
    'No automatic optimization, model superiority, VERIFIED_IMPROVEMENT, P29 acceptance, P26/P11 association or YOLO/LLM qualification.',
)


class _Engine:
    def __init__(self, source, max_entries):
        self.source = source
        self.sha = source['content_sha256']
        self.remaining = max_entries
        self.findings = []
        self.ids = set()
        self.assessments = []
        self.tensors = {t['tensor_id']: (i, t) for i, t in enumerate(source['tensors'])}

    def use(self, n=1):
        self.remaining -= n
        if self.remaining < 0: raise ModelStructureError('architecture_audit_entry_limit')

    def fact(self, field, status='native'):
        value = self.source
        for key in field.split('.'):
            value = value[int(key)] if type(value) is list else value[key]
        self.use()
        return dict(field=field, status=status, value=value)

    def derived(self, method, value, facts):
        return dict(method=method, value=value, source_fields=[f['field'] for f in facts])

    def finding(self, code, kind, rule, facts, *, module=None, tensor=None, path='',
                related_paths=(), expected=None, derived=None, limitation=None, next_review=None):
        self.use()
        identity = dict(source_snapshot_sha256=self.sha, rule_id=rule, rule_version=1,
            code=code, module_id=module, tensor_id=tensor, path=path,
            fields=[f['field'] for f in facts])
        fid = digest(identity)
        if fid in self.ids: raise ModelStructureError('duplicate_architecture_finding')
        self.ids.add(fid)
        evidence = ('observed_static_contradiction' if kind == 'STATIC_INCONSISTENCY' else
                    'insufficient_evidence' if kind == 'COVERAGE_GAP' else
                    'partial_metadata_observation' if any(f['status'] != 'native' for f in facts) else
                    'native_metadata_observation')
        self.findings.append(dict(finding_id=fid, **{k: v for k, v in identity.items() if k != 'fields'},
            classification=kind, related_paths=list(related_paths), observations=facts,
            expected=expected, derived=derived, evidence_status=evidence,
            limitation=limitation or 'Only the named recorded metadata relation is assessed; runtime behavior and full model correctness are not established.',
            next_review=next_review or 'Inspect the referenced retained metadata and the trusted owner registration; capture a fresh static snapshot if clarification is needed.'))

    def gap(self, code, facts, **kwargs):
        self.finding(code, 'COVERAGE_GAP', 'coverage_gaps', facts, **kwargs)

    def module_shapes(self, i, m):
        self.use()
        path, mid, qtype = m['primary_path'], m['module_id'], m['qualified_type']
        prefix = 'modules.%d' % i
        rule = ('linear_shapes' if qtype == _LINEAR and m['category'] == 'linear' else
                'convolution_shapes' if qtype in _CONVS and m['category'] == 'conv%dd' % _CONVS[qtype] else None)
        checks = []
        def check(name, status, facts, expected=None):
            self.use()
            checks.append(dict(code=name, status=status, source_fields=[f['field'] for f in facts], expected=expected))
        if rule is None:
            code = ('CUSTOM_MODULE_SEMANTICS_UNSUPPORTED' if m['category'] == 'custom' else
                    'TRANSPOSED_CONV_RULE_NOT_IMPLEMENTED' if m['category'].startswith('conv_transpose') else
                    'STATIC_SHAPE_RULE_NOT_IMPLEMENTED')
            self.gap(code, [self.fact(prefix + '.qualified_type'), self.fact(prefix + '.category'),
                           self.fact(prefix + '.configuration_status')], module=mid, path=path,
                limitation='This module has no applicable v1 shape rule; captured configuration is not a certificate of its semantics.')
        else:
            conf = {f['name']: (j, f) for j, f in enumerate(m['configuration'])}
            config_facts = {}
            invalid = set()
            def configuration(name, predicate):
                j, f = conf[name]
                fact = self.fact(prefix + '.configuration.%d.value' % j, f['status'])
                config_facts[name] = fact
                if f['status'] != 'native': return None
                if not predicate(f['value']):
                    self.gap('CONFIGURATION_TYPE_UNSUPPORTED_BY_RULE', [fact], module=mid, path=path,
                        limitation='The recorded plain value is not a type supported by this rule; no coercion or shape expectation is inferred.')
                    return None
                return f['value']
            names = ('in_features', 'out_features') if rule == 'linear_shapes' else ('in_channels', 'out_channels', 'groups')
            values = {name: configuration(name, lambda v: type(v) is int) for name in names}
            for name, value in values.items():
                if value is None: continue
                minimum = 1 if name == 'groups' else 0
                facts = [config_facts[name]]
                expected = self.derived('integer_minimum_v1', dict(minimum=minimum), facts)
                okay = value >= minimum
                check(name + '_domain', 'consistent' if okay else 'inconsistent', facts, expected)
                if not okay:
                    invalid.add(name)
                    self.finding('CONV_GROUPS_INVALID' if name == 'groups' else 'STATIC_DIMENSION_INVALID',
                        'STATIC_INCONSISTENCY', rule, facts, module=mid, path=path, expected=expected)
            bias = configuration('bias', lambda v: type(v) is bool)
            kernel = None
            if rule == 'convolution_shapes':
                dim = _CONVS[qtype]
                kernel = configuration('kernel_size', lambda v: type(v) is list and len(v) == dim
                                       and all(type(x) is int for x in v))
                if kernel is not None:
                    facts = [config_facts['kernel_size']]
                    okay = all(x >= 0 for x in kernel)
                    expected = self.derived('kernel_dimension_minimum_v1', dict(dimensions=dim, minimum=0), facts)
                    check('kernel_size_domain', 'consistent' if okay else 'inconsistent', facts, expected)
                    if not okay:
                        invalid.add('kernel_size')
                        self.finding('CONV_KERNEL_SIZE_INVALID', 'STATIC_INCONSISTENCY', rule,
                                     facts, module=mid, path=path, expected=expected)
                groups = values['groups']
                for name in ('in_channels', 'out_channels'):
                    facts = [config_facts[name], config_facts['groups']]
                    if values[name] is None or groups is None:
                        check(name + '_group_divisibility', 'insufficient_evidence', facts)
                    elif name in invalid or 'groups' in invalid:
                        check(name + '_group_divisibility', 'not_applicable', facts)
                    else:
                        okay = values[name] % groups == 0
                        expected = self.derived('channels_divisible_by_groups_v1', dict(remainder=0), facts)
                        check(name + '_group_divisibility', 'consistent' if okay else 'inconsistent', facts, expected)
                        if not okay:
                            invalid.add(name)
                            self.finding('CONV_' + name.upper() + '_NOT_DIVISIBLE', 'STATIC_INCONSISTENCY',
                                         rule, facts, module=mid, path=path, expected=expected)
            weight_facts = [config_facts[name] for name in names]
            if rule == 'convolution_shapes': weight_facts.append(config_facts['kernel_size'])
            shape = None
            if not invalid and all(v is not None for v in values.values()) and (kernel is not None or rule == 'linear_shapes'):
                shape = ([values['out_features'], values['in_features']] if rule == 'linear_shapes' else
                         [values['out_channels'], values['in_channels'] // values['groups'], *kernel])
            self.parameter_shape(i, m, 'weight', shape, weight_facts, rule, check, invalid=bool(invalid))
            out = values['out_features' if rule == 'linear_shapes' else 'out_channels']
            out_name = 'out_features' if rule == 'linear_shapes' else 'out_channels'
            bias_facts = [config_facts[out_name], config_facts['bias']]
            if bias is True:
                self.parameter_shape(i, m, 'bias', [out] if out is not None and out >= 0 else None,
                                     bias_facts, rule, check, invalid=out is not None and out < 0)
            elif bias is False:
                # The producer observes bias presence directly, not a constructor
                # flag. Reader consistency alone cannot establish live origin.
                facts = bias_facts + [self.fact(prefix + '.parameters')]
                present = any(s['name'] == 'bias' and s['tensor_id'] is not None for s in m['parameters'])
                expected = self.derived('absent_registered_bias_v1', None, bias_facts)
                check('bias_presence', 'inconsistent' if present else 'consistent', facts, expected)
                if present:
                    self.finding('BIAS_PRESENCE_CONFLICT', 'STATIC_INCONSISTENCY', rule, facts,
                                 module=mid, path=path, expected=expected)
            else:
                check('bias_shape', 'insufficient_evidence', bias_facts)
        status = ('not_applicable' if rule is None else
                  'inconsistent' if any(c['status'] == 'inconsistent' for c in checks) else
                  'insufficient_evidence' if any(c['status'] == 'insufficient_evidence' for c in checks) else 'consistent')
        self.assessments.append(dict(module_id=mid, path=path, qualified_type=qtype,
            rule_id=rule, rule_version=None if rule is None else 1, status=status, checks=checks))

    def parameter_shape(self, i, m, name, shape, config_facts, rule, check, *, invalid):
        slots = m['parameters']; prefix = 'modules.%d.parameters' % i
        found = next(((j, s) for j, s in enumerate(slots) if s['name'] == name), None)
        facts = list(config_facts)
        if found is None or found[1]['tensor_id'] is None:
            facts.append(self.fact(prefix if found is None else prefix + '.%d' % found[0]))
            check(name + '_shape', 'insufficient_evidence', facts)
            self.gap('REQUIRED_PARAMETER_METADATA_MISSING', facts, module=m['module_id'], path=m['primary_path'],
                related_paths=[m['primary_path'] + '.' + name if m['primary_path'] else name],
                limitation='No native registered parameter shape is available for this check; absence is not replaced by an inferred tensor.')
            return
        j, slot = found
        ti, t = self.tensors[slot['tensor_id']]
        facts.extend([self.fact(prefix + '.%d.tensor_id' % j), self.fact('tensors.%d.shape' % ti, t['status'])])
        if t['status'] != 'native' or shape is None:
            check(name + '_shape', 'not_applicable' if invalid else 'insufficient_evidence', facts)
            return
        expected = self.derived('linear_%s_shape_v1' % name if rule == 'linear_shapes' else
                                'normal_convolution_%s_shape_v1' % name, shape, config_facts)
        okay = t['shape'] == shape
        check(name + '_shape', 'consistent' if okay else 'inconsistent', facts, expected)
        if not okay:
            self.finding(('LINEAR_' if rule == 'linear_shapes' else 'CONV_') + name.upper() + '_SHAPE_MISMATCH',
                'STATIC_INCONSISTENCY', rule, facts, module=m['module_id'], tensor=t['tensor_id'],
                path=m['primary_path'], related_paths=[m['primary_path'] + '.' + name if m['primary_path'] else name], expected=expected)

    def configuration_and_tensor_gaps(self):
        for i, m in enumerate(self.source['modules']):
            for j, field in enumerate(m['configuration']):
                if field['status'] != 'native':
                    self.gap('STATIC_CONFIGURATION_UNAVAILABLE', [self.fact('modules.%d.configuration.%d' % (i, j), field['status'])],
                             module=m['module_id'], path=m['primary_path'])
        for i, t in enumerate(self.source['tensors']):
            if t['status'] != 'native':
                self.gap('TENSOR_METADATA_UNSUPPORTED', [self.fact('tensors.%d' % i, t['status'])], tensor=t['tensor_id'], path='tensors.%d' % i,
                    limitation='Unsupported/lazy/custom tensor metadata is not a definite shape contradiction; known subtotals are not complete totals.')
            if t['storage_status'] != 'native':
                self.gap('STORAGE_ALIAS_UNAVAILABLE', [self.fact('tensors.%d.storage_status' % i, t['storage_status']),
                    self.fact('tensors.%d.storage_reason' % i, t['storage_status'])], tensor=t['tensor_id'],
                    path='tensors.%d' % i,
                    limitation='Missing storage identity prevents alias assessment; it does not prove independent storage or zero memory use.')

    def inventory(self):
        summary = self.source['summary']
        keys = ('parameter_elements', 'trainable_elements', 'frozen_elements', 'unique_parameter_tensors',
                'module_types', 'categories', 'parameter_dtypes', 'buffer_dtypes')
        facts = [self.fact('summary.' + k, summary[k]['status'] if type(summary[k]) is dict else 'native') for k in keys]
        quantities = [summary[k] for k in keys[:3]]
        complete = all(q['status'] == 'native' for q in quantities)
        total = quantities[0]['value']
        def ratio(n, d):
            with localcontext() as context:
                context.prec = 50
                return float(Decimal(n) / Decimal(d))
        ratios = dict(trainable_fraction=None, frozen_fraction=None,
                      unavailable_reason='incomplete_total' if not complete else 'no_parameter_elements' if total == 0 else None)
        if complete and total:
            ratios.update(trainable_fraction=ratio(quantities[1]['value'], total),
                          frozen_fraction=ratio(quantities[2]['value'], total))
        self.finding('PARAMETER_INVENTORY', 'OBSERVATION', 'parameter_inventory', facts,
            derived=self.derived('unique_parameter_element_fractions_v1', ratios, facts[:3]),
            limitation='Unique registered tensor objects only; large or fully frozen models are not inherently erroneous. requires_grad does not establish training behavior.')
        for i, m in enumerate(self.source['modules']):
            if m['category'] == 'custom' or m['configuration_status'] != 'native': continue
            if not any(s['tensor_id'] is not None for s in m['parameters']): continue
            q = m['direct_parameter_elements']
            facts = [self.fact('modules.%d.direct_parameter_elements' % i, q['status']), self.fact('summary.parameter_elements', summary['parameter_elements']['status'])]
            share = ratio(q['value'], total) if q['status'] == 'native' and summary['parameter_elements']['status'] == 'native' and total else None
            self.finding('DIRECT_PARAMETER_CONCENTRATION', 'OBSERVATION', 'parameter_inventory', facts,
                module=m['module_id'], path=m['primary_path'],
                derived=self.derived('direct_unique_parameter_element_share_v1', dict(fraction=share,
                    unavailable_reason=None if share is not None else 'incomplete_or_zero_total'), facts),
                limitation='Direct unique parameter objects for this owner, not recursive/executed layers or compute density. Shares overlap for tied tensors and must not be summed.')

    def relationships(self):
        module_paths = defaultdict(list)
        for i, p in enumerate(self.source['paths']):
            self.use()
            if p['cycle']:
                self.finding('REGISTERED_MODULE_CYCLE', 'OBSERVATION', 'registration_relationships',
                    [self.fact('paths.%d' % i)], module=p['module_id'], path=p['path'],
                    limitation='A cycle in registration is not proof of a forward execution cycle, a recursion failure or a defect.')
            else: module_paths[p['module_id']].append((i, p))
        for mid, rows in module_paths.items():
            if len(rows) > 1:
                self.finding('SHARED_MODULE_REGISTRATIONS', 'OBSERVATION', 'registration_relationships',
                    [self.fact('paths.%d' % i) for i, _ in rows], module=mid, path=rows[0][1]['path'],
                    related_paths=[p['path'] for _, p in rows],
                    limitation='Repeated registered paths identify one local module object, not repeated executions or a defect.')
        endpoints = defaultdict(list)
        for i, m in enumerate(self.source['modules']):
            for kind in ('parameters', 'buffers'):
                for j, slot in enumerate(m[kind]):
                    self.use()
                    if slot['tensor_id'] is not None:
                        endpoints[slot['tensor_id']].append((i, m, kind, j))
        for tid, rows in endpoints.items():
            if len(rows) > 1:
                only_parameters = all(kind == 'parameters' for _, _, kind, _ in rows)
                self.finding('TIED_PARAMETER_REGISTRATIONS' if only_parameters else 'SHARED_TENSOR_REGISTRATIONS',
                    'OBSERVATION', 'registration_relationships',
                    [self.fact('modules.%d.%s.%d' % (i, kind, j)) for i, _, kind, j in rows],
                    tensor=tid, path=rows[0][1]['primary_path'], related_paths=list(dict.fromkeys(m['primary_path'] for _, m, _, _ in rows)),
                    limitation='Identity is within this snapshot and each unique tensor object is counted once globally; sharing is not itself an error.')
        for i, group in enumerate(self.source['summary']['storage_groups']):
            if len(group['tensor_ids']) > 1:
                self.finding('SHARED_STORAGE_OBSERVED', 'OBSERVATION', 'registration_relationships',
                    [self.fact('summary.storage_groups.%d' % i)],
                    path='summary.storage_groups.%d' % i,
                    limitation='Distinct local tensor objects share a recorded storage handle; this does not establish byte-range overlap, physical bytes, danger or cross-snapshot identity.')


def build_audit(source, max_entries=_MAX_ENTRIES):
    engine = _Engine(source, max_entries)
    for i, m in enumerate(source['modules']): engine.module_shapes(i, m)
    engine.inventory()
    engine.relationships()
    engine.configuration_and_tensor_gaps()
    # Policy order, then source traversal/check order is part of v1 semantics.
    order = {r['id']: i for i, r in enumerate(_POLICY['rules'])}
    engine.findings.sort(key=lambda f: order[f['rule_id']])
    states = Counter(a['status'] for a in engine.assessments)
    kinds = Counter(f['classification'] for f in engine.findings)
    eligible = sum(a['rule_id'] is not None for a in engine.assessments)
    check_states = Counter(c['status'] for a in engine.assessments for c in a['checks'])
    partial_ids = [a['module_id'] for a in engine.assessments
                   if any(c['status'] == 'insufficient_evidence' for c in a['checks'])]
    partial_set = set(partial_ids)
    coverage = dict(total_unique_modules=len(source['modules']), registered_module_paths=len(source['paths']),
        shape_rule_eligible_modules=eligible, examined_supported_modules=states['consistent'] + states['inconsistent'],
        consistent_modules=states['consistent'], inconsistent_modules=states['inconsistent'],
        insufficient_evidence_modules=states['insufficient_evidence'], not_applicable_modules=states['not_applicable'],
        module_ids_with_insufficient_checks=partial_ids,
        check_counts={k: check_states[k] for k in ('consistent', 'inconsistent', 'insufficient_evidence', 'not_applicable')},
        unassessed_module_ids=[a['module_id'] for a in engine.assessments
            if a['status'] in ('insufficient_evidence', 'not_applicable') or a['module_id'] in partial_set],
        status='no_applicable_shape_modules' if not eligible else 'partial' if kinds['COVERAGE_GAP'] else 'recorded_fields_assessed_for_v1_rules',
        module_assessments=engine.assessments,
        rule_coverage=[dict(rule_id=r['id'], rule_version=r['version'],
            scope=r['scope'], finding_count=sum(f['rule_id'] == r['id'] for f in engine.findings),
            module_assessment_count=sum(a['rule_id'] == r['id'] for a in engine.assessments)) for r in _POLICY['rules']])
    data = plain(dict(format='hyso.model_architecture_audit', schema_version=1, source_snapshot=source,
        audit_policy=_POLICY, summary=dict(total_findings=len(engine.findings),
            finding_counts={k: kinds[k] for k in ('STATIC_INCONSISTENCY', 'OBSERVATION', 'COVERAGE_GAP')}),
        findings=engine.findings, coverage=coverage, limitations=_LIMITATIONS))
    data['content_sha256'] = digest(data)
    return data
