"""Typed, immutable static architecture audit. Source-based reader validation."""
from dataclasses import dataclass
from enum import Enum
import json

from .analysis._codec import canonical, export, plain, read_json
from .structure import ModelStructureError, ModelStructureSnapshot
from ._architecture_rules import _MAX_BYTES, _MAX_ENTRIES, build_audit
from ._architecture_coverage import build_audit_v2

__all__ = ['audit_model_architecture', 'ModelArchitectureAudit', 'ModelArchitectureAuditError',
    'ArchitectureFindingClass', 'ArchitectureAuditFinding', 'AuditEvidence', 'AuditDerivedValue',
    'ArchitectureAuditRule', 'ArchitectureAuditPolicy', 'ArchitectureModuleAssessment', 'ArchitectureAuditCheck',
    'ArchitectureCoverageStatus', 'ArchitectureRuleApplicability', 'ArchitectureUnassessedArea', 'ArchitectureEvidenceGap']


class ModelArchitectureAuditError(ModelStructureError):
    """Invalid source/result or explicit audit resource limit; never truncates."""


class ArchitectureFindingClass(str, Enum):
    STATIC_INCONSISTENCY = 'STATIC_INCONSISTENCY'
    OBSERVATION = 'OBSERVATION'
    COVERAGE_GAP = 'COVERAGE_GAP'


class ArchitectureCoverageStatus(str, Enum):
    APPLICABLE = 'APPLICABLE'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    UNASSESSED_BY_RULESET = 'UNASSESSED_BY_RULESET'
    COVERAGE_GAP = 'COVERAGE_GAP'


@dataclass(frozen=True)
class AuditEvidence:
    field: str
    status: str
    _value_json: str

    @property
    def value(self): return json.loads(self._value_json)


@dataclass(frozen=True)
class AuditDerivedValue:
    method: str
    source_fields: tuple
    _value_json: str

    @property
    def value(self): return json.loads(self._value_json)


@dataclass(frozen=True)
class ArchitectureAuditFinding:
    finding_id: str
    code: str
    classification: ArchitectureFindingClass
    rule_id: str
    rule_version: int
    module_id: str | None
    tensor_id: str | None
    path: str
    related_paths: tuple
    observations: tuple
    expected: AuditDerivedValue | None
    derived: AuditDerivedValue | None
    evidence_status: str
    source_snapshot_sha256: str
    limitation: str
    next_review: str


@dataclass(frozen=True)
class ArchitectureAuditRule:
    id: str
    version: int
    scope: str


@dataclass(frozen=True)
class ArchitectureAuditPolicy:
    id: str
    version: int
    source_format: str
    source_schema_version: int
    producer_method: str
    producer_version: int
    classification_version: int
    rules: tuple


@dataclass(frozen=True)
class ArchitectureAuditCheck:
    code: str
    status: str
    source_fields: tuple
    expected: AuditDerivedValue | None


@dataclass(frozen=True)
class ArchitectureModuleAssessment:
    module_id: str
    path: str
    qualified_type: str
    rule_id: str | None
    rule_version: int | None
    status: str
    checks: tuple


@dataclass(frozen=True)
class ArchitectureRuleApplicability:
    module_id: str
    path: str
    rule_id: str
    rule_version: int
    status: ArchitectureCoverageStatus
    reason: str
    observations: tuple
    source_snapshot_sha256: str


@dataclass(frozen=True)
class ArchitectureUnassessedArea:
    module_id: str
    path: str
    area: str
    status: ArchitectureCoverageStatus
    reason: str
    observations: tuple
    source_snapshot_sha256: str
    limitation: str


@dataclass(frozen=True)
class ArchitectureEvidenceGap:
    finding_id: str
    status: ArchitectureCoverageStatus
    code: str
    required_by_rule_id: str
    module_ids: tuple
    source_fields: tuple


def _derived(d):
    return None if d is None else AuditDerivedValue(d['method'], tuple(d['source_fields']), canonical(d['value']))


@dataclass(frozen=True, init=False)
class ModelArchitectureAudit:
    __slots__ = ('_payload',)
    _payload: str

    def __init__(self, data):
        try:
            d = plain(data)
            raw = canonical(d)
            if len(raw.encode('utf-8')) > _MAX_BYTES: raise ModelArchitectureAuditError('architecture_audit_byte_limit')
            # The source constructor verifies the closed producer schema, all
            # derived counts/paths and its hash before any audit assumptions.
            source = ModelStructureSnapshot(d['source_snapshot'])
            version = d.get('schema_version')
            if type(version) is not int or version not in (1, 2):
                raise ModelArchitectureAuditError('unsupported_architecture_audit_version')
            expected = (build_audit if version == 1 else build_audit_v2)(source.to_dict())
            if raw != canonical(expected):
                raise ModelArchitectureAuditError('architecture_audit_derivation_or_integrity_mismatch')
        except ModelArchitectureAuditError:
            raise
        except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
            raise ModelArchitectureAuditError('invalid_architecture_audit') from exc
        object.__setattr__(self, '_payload', raw)

    def to_dict(self): return json.loads(self._payload)

    @property
    def source_snapshot(self): return ModelStructureSnapshot(self.to_dict()['source_snapshot'])

    @property
    def audit_policy(self):
        d = self.to_dict()['audit_policy']
        return ArchitectureAuditPolicy(**dict(d, rules=tuple(ArchitectureAuditRule(**r) for r in d['rules'])))

    @property
    def findings(self):
        return tuple(ArchitectureAuditFinding(**dict(d,
            classification=ArchitectureFindingClass(d['classification']), related_paths=tuple(d['related_paths']),
            observations=tuple(AuditEvidence(f['field'], f['status'], canonical(f['value'])) for f in d['observations']),
            expected=_derived(d['expected']), derived=_derived(d['derived']))) for d in self.to_dict()['findings'])

    @property
    def module_assessments(self):
        return tuple(ArchitectureModuleAssessment(**dict(d, checks=tuple(ArchitectureAuditCheck(
            c['code'], c['status'], tuple(c['source_fields']), _derived(c['expected'])) for c in d['checks'])))
            for d in self.to_dict()['coverage']['module_assessments'])

    def _v2_coverage(self):
        if self.schema_version != 2:
            raise ModelArchitectureAuditError('v2_coverage_required')
        return self.to_dict()['coverage']

    @property
    def rule_applicability(self):
        return tuple(ArchitectureRuleApplicability(**dict(d, status=ArchitectureCoverageStatus(d['status']),
            observations=tuple(AuditEvidence(f['field'], f['status'], canonical(f['value'])) for f in d['observations'])))
            for d in self._v2_coverage()['rule_applicability'])

    @property
    def unassessed_areas(self):
        return tuple(ArchitectureUnassessedArea(**dict(d, status=ArchitectureCoverageStatus(d['status']),
            observations=tuple(AuditEvidence(f['field'], f['status'], canonical(f['value'])) for f in d['observations'])))
            for d in self._v2_coverage()['unassessed_areas'])

    @property
    def evidence_gaps(self):
        return tuple(ArchitectureEvidenceGap(**dict(d, status=ArchitectureCoverageStatus(d['status']),
            module_ids=tuple(d['module_ids']), source_fields=tuple(d['source_fields'])))
            for d in self._v2_coverage()['evidence_gaps'])

    @property
    def schema_version(self): return self.to_dict()['schema_version']

    @property
    def summary(self): return self.to_dict()['summary']

    @property
    def coverage(self): return self.to_dict()['coverage']

    @property
    def limitations(self): return tuple(self.to_dict()['limitations'])

    @property
    def content_sha256(self): return self.to_dict()['content_sha256']

    def to_json(self, path=None):
        if path is not None: export(path, self._payload + '\n')
        return self._payload + '\n'

    @classmethod
    def from_dict(cls, data): return cls(data)

    @classmethod
    def from_json(cls, path):
        d, _, _ = read_json(path, max_bytes=_MAX_BYTES + 1)
        return cls(d)


def audit_model_architecture(snapshot, *, policy_version=2, max_entries=_MAX_ENTRIES, max_bytes=_MAX_BYTES):
    """Assess one validated static snapshot; v2 is the creation default.

    policy_version=1 reproduces the original v1 bytes/semantics. Readers select
    only the explicit retained schema/policy and never migrate an old result.
    Accepts no live model, model factory, filename or checkpoint. Consistency is
    limited to named recorded fields, with explicit observations and coverage
    gaps. No runtime behavior or model-wide correctness verdict is inferred.
    """
    if type(snapshot) is not ModelStructureSnapshot:
        raise ModelArchitectureAuditError('validated_structure_snapshot_required')
    if type(policy_version) is not int or policy_version not in (1, 2):
        raise ModelArchitectureAuditError('unsupported_architecture_audit_policy')
    if (type(max_entries) is not int or not 1 <= max_entries <= _MAX_ENTRIES
            or type(max_bytes) is not int or not 1 <= max_bytes <= _MAX_BYTES):
        raise ModelArchitectureAuditError('invalid_architecture_audit_limit')
    try:
        source = ModelStructureSnapshot(snapshot.to_dict())
        data = (build_audit if policy_version == 1 else build_audit_v2)(source.to_dict(), max_entries)
        if len(canonical(data).encode('utf-8')) > max_bytes:
            raise ModelArchitectureAuditError('architecture_audit_byte_limit')
        return ModelArchitectureAudit(data)
    except ModelArchitectureAuditError:
        raise
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError, RecursionError) as exc:
        raise ModelArchitectureAuditError('invalid_architecture_audit_source_or_limit') from exc
