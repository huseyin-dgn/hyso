"""Typed, immutable static architecture audit. Source-based reader validation."""
import typing as _typing
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
    """Closed ArchitectureFindingClass labels: STATIC_INCONSISTENCY, OBSERVATION, COVERAGE_GAP.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    STATIC_INCONSISTENCY = 'STATIC_INCONSISTENCY'
    OBSERVATION = 'OBSERVATION'
    COVERAGE_GAP = 'COVERAGE_GAP'


class ArchitectureCoverageStatus(str, Enum):
    """Closed ArchitectureCoverageStatus labels: APPLICABLE, NOT_APPLICABLE, UNASSESSED_BY_RULESET,
    COVERAGE_GAP.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    APPLICABLE = 'APPLICABLE'
    NOT_APPLICABLE = 'NOT_APPLICABLE'
    UNASSESSED_BY_RULESET = 'UNASSESSED_BY_RULESET'
    COVERAGE_GAP = 'COVERAGE_GAP'


@dataclass(frozen=True)
class AuditEvidence:
    """Store audit evidence evidence as an immutable typed record.

    Args:
        field (str): Source metadata field path used by this observation. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    field: str
    status: str
    _value_json: str

    @property
    def value(self) -> '_typing.Any':
        """Decode and return the retained JSON evidence value.
        """
        return json.loads(self._value_json)


@dataclass(frozen=True)
class AuditDerivedValue:
    """Store audit derived value evidence as an immutable typed record.

    Args:
        method (str): TorchScript capture strategy: trace records the example path; script compiles
            supported source. Required.
        source_fields (tuple): Exact metadata field paths used to derive or assess this evidence.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    method: str
    source_fields: tuple
    _value_json: str

    @property
    def value(self) -> '_typing.Any':
        """Decode and return the retained JSON evidence value.
        """
        return json.loads(self._value_json)


@dataclass(frozen=True)
class ArchitectureAuditFinding:
    """Store architecture audit finding evidence as an immutable typed record.

    Args:
        finding_id (str): Stable finding identifier within this audit result. Required.
        code (str): Stable machine-readable diagnostic code; distinct from the human-readable reason.
            Required.
        classification (ArchitectureFindingClass): Static finding classification; observations are not
            measured optimization outcomes. Required.
        rule_id (str): Identifier of the audit rule that produced this finding. Required.
        rule_version (int): Version of the applied audit rule. Required.
        module_id (str | None): Snapshot-local registered module identifier. Required.
        tensor_id (str | None): Snapshot-local tensor identifier; not a cross-model object identity.
            Required.
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.
        related_paths (tuple): Other registered module paths contributing to this finding. Required.
        observations (tuple): Retained static evidence observations used by the rule. Required.
        expected (AuditDerivedValue | None): Expected value or supported constraint used by the
            diagnostic. Required.
        derived (AuditDerivedValue | None): Explicitly derived evidence value with its calculation
            provenance. Required.
        evidence_status (str): Recorded completeness/coverage status for this finding. Required.
        source_snapshot_sha256 (str): Canonical source structure-snapshot digest, not a model-state
            digest. Required.
        limitation (str): Specific scope restriction that prevents a broader conclusion. Required.
        next_review (str): Manual follow-up suggested by the static finding. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
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
    """Store architecture audit rule evidence as an immutable typed record.

    Args:
        id (str): Stable policy or rule identifier. Required.
        version (int): Version of this evidence policy or rule. Required.
        scope (str): Declared evidence coverage; does not authorize execution. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    id: str
    version: int
    scope: str


@dataclass(frozen=True)
class ArchitectureAuditPolicy:
    """Store architecture audit policy evidence as an immutable typed record.

    Args:
        id (str): Stable policy or rule identifier. Required.
        version (int): Version of this evidence policy or rule. Required.
        source_format (str): Accepted source evidence format identifier. Required.
        source_schema_version (int): Accepted source evidence schema version. Required.
        producer_method (str): Recorded evidence producer identifier. Required.
        producer_version (int): Recorded evidence producer version. Required.
        classification_version (int): Version of the exact-type structural classification policy.
            Required.
        rules (tuple): Ordered immutable audit rules used by this policy. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
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
    """Store architecture audit check evidence as an immutable typed record.

    Args:
        code (str): Stable machine-readable diagnostic code; distinct from the human-readable reason.
            Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        source_fields (tuple): Exact metadata field paths used to derive or assess this evidence.
            Required.
        expected (AuditDerivedValue | None): Expected value or supported constraint used by the
            diagnostic. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    code: str
    status: str
    source_fields: tuple
    expected: AuditDerivedValue | None


@dataclass(frozen=True)
class ArchitectureModuleAssessment:
    """Store architecture module assessment evidence as an immutable typed record.

    Args:
        module_id (str): Snapshot-local registered module identifier. Required.
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.
        qualified_type (str): Recorded Python module-qualified type name, without semantic-equivalence
            inference. Required.
        rule_id (str | None): Identifier of the audit rule that produced this finding. Required.
        rule_version (int | None): Version of the applied audit rule. Required.
        status (str): Recorded evidence/coverage status; unavailable values must not be interpreted as
            zero. Required.
        checks (tuple): Existing model-check or audit observations; no measurement is inferred.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
    module_id: str
    path: str
    qualified_type: str
    rule_id: str | None
    rule_version: int | None
    status: str
    checks: tuple


@dataclass(frozen=True)
class ArchitectureRuleApplicability:
    """Store architecture rule applicability evidence as an immutable typed record.

    Args:
        module_id (str): Snapshot-local registered module identifier. Required.
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.
        rule_id (str): Identifier of the audit rule that produced this finding. Required.
        rule_version (int): Version of the applied audit rule. Required.
        status (ArchitectureCoverageStatus): Recorded evidence/coverage status; unavailable values must
            not be interpreted as zero. Required.
        reason (str): Recorded explanation for rejection, absence or incomplete evidence; None only
            where the schema allows it. Required.
        observations (tuple): Retained static evidence observations used by the rule. Required.
        source_snapshot_sha256 (str): Canonical source structure-snapshot digest, not a model-state
            digest. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
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
    """Store architecture unassessed area evidence as an immutable typed record.

    Args:
        module_id (str): Snapshot-local registered module identifier. Required.
        path (str): Filesystem location read or written by this explicit operation; output parents must
            already exist. Required.
        area (str): Static structure area left unassessed or unsupported. Required.
        status (ArchitectureCoverageStatus): Recorded evidence/coverage status; unavailable values must
            not be interpreted as zero. Required.
        reason (str): Recorded explanation for rejection, absence or incomplete evidence; None only
            where the schema allows it. Required.
        observations (tuple): Retained static evidence observations used by the rule. Required.
        source_snapshot_sha256 (str): Canonical source structure-snapshot digest, not a model-state
            digest. Required.
        limitation (str): Specific scope restriction that prevents a broader conclusion. Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
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
    """Store architecture evidence gap evidence as an immutable typed record.

    Args:
        finding_id (str): Stable finding identifier within this audit result. Required.
        status (ArchitectureCoverageStatus): Recorded evidence/coverage status; unavailable values must
            not be interpreted as zero. Required.
        code (str): Stable machine-readable diagnostic code; distinct from the human-readable reason.
            Required.
        required_by_rule_id (str): Rule requiring the missing source fields. Required.
        module_ids (tuple): Snapshot-local modules implicated by the rule or gap. Required.
        source_fields (tuple): Exact metadata field paths used to derive or assess this evidence.
            Required.

    Notes:
        These fields retain source evidence; constructing a record does not validate a live model,
        execute a workload or certify complete coverage. Use the domain producer/reader for schema
        validation.
    """
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
    """Inspect rule findings, coverage and evidence gaps from a retained architecture audit.

    Args:
        data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
            version/schema validator. Required.

    Returns:
        An instance of ModelArchitectureAudit with the documented native validation and defaults.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        ModelArchitectureAudit(data)

    HYSo Guide: comparison.architecture
    Offline: hyso docs show comparison.architecture --lang en
    Open: hyso docs open comparison.architecture --lang en
    """
    __slots__ = ('_payload',)
    _payload: str

    def __init__(self, data: 'dict[str, _typing.Any]') -> 'None':
        """Initialize ModelArchitectureAudit with the supplied settings.

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
            ModelArchitectureAudit(data)

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
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

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return json.loads(self._payload)

    @property
    def source_snapshot(self) -> '_typing.Any':
        """Read the retained source snapshot from this ModelArchitectureAudit record.

        Returns:
            The retained source snapshot value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.source_snapshot

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return ModelStructureSnapshot(self.to_dict()['source_snapshot'])

    @property
    def audit_policy(self) -> '_typing.Any':
        """Read the retained audit policy from this ModelArchitectureAudit record.

        Returns:
            The retained audit policy value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.audit_policy

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        d = self.to_dict()['audit_policy']
        return ArchitectureAuditPolicy(**dict(d, rules=tuple(ArchitectureAuditRule(**r) for r in d['rules'])))

    @property
    def findings(self) -> '_typing.Any':
        """Read the retained findings from this ModelArchitectureAudit record.

        Returns:
            The retained findings value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.findings

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(ArchitectureAuditFinding(**dict(d,
            classification=ArchitectureFindingClass(d['classification']), related_paths=tuple(d['related_paths']),
            observations=tuple(AuditEvidence(f['field'], f['status'], canonical(f['value'])) for f in d['observations']),
            expected=_derived(d['expected']), derived=_derived(d['derived']))) for d in self.to_dict()['findings'])

    @property
    def module_assessments(self) -> '_typing.Any':
        """Read the retained module assessments from this ModelArchitectureAudit record.

        Returns:
            The retained module assessments value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.module_assessments

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(ArchitectureModuleAssessment(**dict(d, checks=tuple(ArchitectureAuditCheck(
            c['code'], c['status'], tuple(c['source_fields']), _derived(c['expected'])) for c in d['checks'])))
            for d in self.to_dict()['coverage']['module_assessments'])

    def _v2_coverage(self):
        if self.schema_version != 2:
            raise ModelArchitectureAuditError('v2_coverage_required')
        return self.to_dict()['coverage']

    @property
    def rule_applicability(self) -> '_typing.Any':
        """Read the retained rule applicability from this ModelArchitectureAudit record.

        Returns:
            The retained rule applicability value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.rule_applicability

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(ArchitectureRuleApplicability(**dict(d, status=ArchitectureCoverageStatus(d['status']),
            observations=tuple(AuditEvidence(f['field'], f['status'], canonical(f['value'])) for f in d['observations'])))
            for d in self._v2_coverage()['rule_applicability'])

    @property
    def unassessed_areas(self) -> '_typing.Any':
        """Read the retained unassessed areas from this ModelArchitectureAudit record.

        Returns:
            The retained unassessed areas value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.unassessed_areas

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(ArchitectureUnassessedArea(**dict(d, status=ArchitectureCoverageStatus(d['status']),
            observations=tuple(AuditEvidence(f['field'], f['status'], canonical(f['value'])) for f in d['observations'])))
            for d in self._v2_coverage()['unassessed_areas'])

    @property
    def evidence_gaps(self) -> '_typing.Any':
        """Read the retained evidence gaps from this ModelArchitectureAudit record.

        Returns:
            The retained evidence gaps value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.evidence_gaps

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(ArchitectureEvidenceGap(**dict(d, status=ArchitectureCoverageStatus(d['status']),
            module_ids=tuple(d['module_ids']), source_fields=tuple(d['source_fields'])))
            for d in self._v2_coverage()['evidence_gaps'])

    @property
    def schema_version(self) -> '_typing.Any':
        """Return the retained record's schema version.

        Returns:
            The integer schema version of this retained record.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.schema_version

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['schema_version']

    @property
    def summary(self) -> '_typing.Any':
        """Read the retained summary from this ModelArchitectureAudit record.

        Returns:
            The retained summary value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.summary

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['summary']

    @property
    def coverage(self) -> '_typing.Any':
        """Read the retained coverage from this ModelArchitectureAudit record.

        Returns:
            The retained coverage value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.coverage

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['coverage']

    @property
    def limitations(self) -> '_typing.Any':
        """Read the retained limitations from this ModelArchitectureAudit record.

        Returns:
            The retained limitations value, without recomputing measurements.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.limitations

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return tuple(self.to_dict()['limitations'])

    @property
    def content_sha256(self) -> '_typing.Any':
        """Return the canonical content digest of this retained record.

        Returns:
            The canonical SHA-256 digest string of this retained content.

        Raises:
            No additional input validation; corrupt retained state or native access errors propagate.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.content_sha256

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return self.to_dict()['content_sha256']

    def to_json(self, path: '_typing.Any'=None) -> '_typing.Any':
        """Serialize this object using its native JSON format and overwrite policy.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Default: None.

        Returns:
            A JSON string when path is None, otherwise None after writing the file.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            obj.to_json()

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        if path is not None: export(path, self._payload + '\n')
        return self._payload + '\n'

    @classmethod
    def from_dict(cls, data: 'dict[str, _typing.Any]') -> 'ModelArchitectureAudit':
        """Validate plain retained data and reconstruct the typed result.

        Args:
            data (dict[str, _typing.Any]): Detached plain-data payload accepted by the existing
                version/schema validator. Required.

        Returns:
            A validated ModelArchitectureAudit instance reconstructed from the supplied data.

        Raises:
            ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
            rejected.
            RuntimeError: A requested native workload fails; inspect retained state before retrying.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ModelArchitectureAudit.from_dict(data)

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        return cls(data)

    @classmethod
    def from_json(cls, path: '_typing.Any') -> 'ModelArchitectureAudit':
        """Read and validate this type from a local JSON file.

        Args:
            path (_typing.Any): Filesystem path for the specific reader/exporter; no configuration code is
                executed. Required.

        Returns:
            A validated ModelArchitectureAudit instance reconstructed from the supplied data.

        Raises:
            OSError: Filesystem access fails.
            ValueError: The native format, version or overwrite contract is rejected.

        Examples:
            # Caller-supplied values; explicitly requested operations only.
            ModelArchitectureAudit.from_json(path)

        HYSo Guide: comparison.architecture
        Offline: hyso docs show comparison.architecture --lang en
        Open: hyso docs open comparison.architecture --lang en
        """
        d, _, _ = read_json(path, max_bytes=_MAX_BYTES + 1)
        return cls(d)


def audit_model_architecture(snapshot: '_typing.Any', *, policy_version: 'int'=2, max_entries: '_typing.Any'=_MAX_ENTRIES, max_bytes: '_typing.Any'=_MAX_BYTES) -> '_typing.Any':
    """Assess one validated static snapshot; v2 is the creation default.

    policy_version=1 reproduces the original v1 bytes/semantics. Readers select
    only the explicit retained schema/policy and never migrate an old result.
    Accepts no live model, model factory, filename or checkpoint. Consistency is
    limited to named recorded fields, with explicit observations and coverage
    gaps. No runtime behavior or model-wide correctness verdict is inferred.

    Args:
        snapshot (_typing.Any): Validated retained model-structure snapshot. Required.
        policy_version (int): Exact supported audit policy version. Default: 2.
        max_entries (_typing.Any): Upper bound on entries retained/accepted by this operation. Default:
            _MAX_ENTRIES.
        max_bytes (_typing.Any): Maximum accepted or retained byte count. Default: _MAX_BYTES.

    Returns:
        The retained audit model architecture value, without recomputing measurements.

    Raises:
        ValueError, TypeError: The existing typed-source, configuration or compatibility contract is
        rejected.
        RuntimeError: A requested native workload fails; inspect retained state before retrying.

    Examples:
        # Caller-supplied values; explicitly requested operations only.
        audit_model_architecture(snapshot)

    HYSo Guide: comparison.architecture
    Offline: hyso docs show comparison.architecture --lang en
    Open: hyso docs open comparison.architecture --lang en
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
