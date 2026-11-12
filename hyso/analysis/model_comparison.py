"""Two-model, retained-evidence comparison. No capture, model or factory execution."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Sequence

from ._codec import canonical, digest, export, plain, read_json
from .evidence import EvidenceIndex, EvidenceRecord
from .report import AnalysisContext, UnifiedAnalysisReport
from .feature_plan import FeaturePlanRequest, ModelContext
from .execution import AnalysisExecutionResult
from ..benchmark.result import BenchmarkResult, compare_results
from ..structure import ModelStructureSnapshot


_MAX_BYTES = 128*1024**2
_METRICS = {
    'P01': ('resident_forward_p50','resident_forward_p95','resident_forward_p99'),
    'P02': ('completed_images_per_second',),
    'P10': ('process_rss_current','process_rss_sampled_peak','process_rss_baseline'),
    'P11': ('model_parameter_logical_bytes','model_buffer_logical_bytes',
            'model_parameter_buffer_logical_bytes','serialized_artifact_bytes'),
    'P26': ('test_accuracy','test_macro_f1'),
}
_LIMITS = (
    'descriptive_metric_differences_only;no_global_winner',
    'statistical_independence_significance_and_causality_not_certified',
    'P10_is_process_RSS_not_model_exclusive_memory;P11_is_logical_bytes_not_P09_VRAM',
    'missing_producer_comparison_metadata_is_not_backfilled_from_caller_declarations',
    'P01_native_tail_policy_and_existing_comparison_sample_requirements_retained',
    'P26_quality_is_exact_observed_stream_only;dataset_origin_and_preprocessing_not_verified',
    'structure_metadata_is_not_executed_behavior_or_an_automatic_P11_measurement',
    'no_workload_or_checkpoint_access;no_acceptance_or_VERIFIED_IMPROVEMENT_claim',
)


class ModelComparisonError(ValueError):
    """Invalid selection, model/record association or inconsistent serialized result."""


@dataclass(frozen=True, init=False)
class ModelComparisonSource:
    """Immutable existing report plus explicit model identity/optional retained sources.

    Args:
        model_id: Caller model label; never a native state/checkpoint identity.
        report: Existing, validated evidence report, including a shared two-model report.
        records: Optional feature -> exact record ID selection. Otherwise unique
            matching model_id records are chosen. Ambiguity never picks a latest run.
        structure: Optional already-captured snapshot, not a live model. Without
            native P26 binding its association with this model remains declared.
        benchmark: Optional original P01 raw result; linkage is checked before use.

    Scope declarations never repair missing producer comparison_capture metadata.
    """
    _payload: str

    def __init__(self, model_id: str, report: UnifiedAnalysisReport, *,
                 records: Mapping[str,str] | None = None,
                 structure: ModelStructureSnapshot | None = None,
                 benchmark: BenchmarkResult | None = None) -> None:
        ModelContext(model_id)
        if type(report) is not UnifiedAnalysisReport:
            raise ModelComparisonError('validated_analysis_report_required')
        if structure is not None and type(structure) is not ModelStructureSnapshot:
            raise ModelComparisonError('retained_structure_snapshot_required;no_live_model_capture')
        if benchmark is not None and type(benchmark) is not BenchmarkResult:
            raise ModelComparisonError('retained_benchmark_result_required')
        data = UnifiedAnalysisReport(report.to_dict()).to_dict()
        available = {r['record_id']:r for r in data['index']['records']}
        choices = {}
        if records is not None:
            if type(records) is not dict or any(type(k) is not str or type(v) is not str for k,v in records.items()):
                raise ModelComparisonError('feature_to_exact_record_id_mapping_required')
            for feature, identity in records.items():
                FeaturePlanRequest('single',(ModelContext(model_id),),features=(feature,))
                record = available.get(identity)
                if record is None or record['feature_id']!=feature or record['scope']['model_id'] not in (None,model_id):
                    raise ModelComparisonError('record_feature_or_model_identity_mismatch')
                choices[feature] = [identity]
        else:
            for record in available.values():
                if record['scope']['model_id']==model_id:
                    choices.setdefault(record['feature_id'],[]).append(record['record_id'])
        body = dict(model_id=model_id,report=data,records=choices,
                    structure=None if structure is None else structure.to_dict(),
                    benchmark=None if benchmark is None else benchmark.to_dict())
        raw = canonical(body)
        if len(raw.encode('utf-8'))>_MAX_BYTES:
            raise ModelComparisonError('comparison_source_size_limit')
        object.__setattr__(self,'_payload',raw)

    @property
    def model_id(self) -> str:
        return self.to_dict()['model_id']

    def to_dict(self) -> dict[str,Any]:
        return json.loads(self._payload)

    @classmethod
    def from_execution(cls, result: AnalysisExecutionResult, model_id: str) -> ModelComparisonSource:
        """Adapt retained Stage-5 output; never execute missing analyses or refresh data."""
        if type(result) is not AnalysisExecutionResult:
            raise ModelComparisonError('analysis_execution_result_required')
        rows = [r for r in result.to_dict()['features'] if r['model_id']==model_id]
        if not rows:
            raise ModelComparisonError('model_not_present_in_execution')
        index = EvidenceIndex(r for r in result.to_evidence_index().records if r.to_dict()['scope']['model_id']==model_id)
        report = index.analyze(context=AnalysisContext(workload_id='retained-execution',task='classification',
            model={'id':model_id},dataset={'identity':'not_inferred'}))
        raw = [r['auxiliary_evidence']['benchmark'] for r in rows if r['feature_id']=='P01'
               and 'benchmark' in r['auxiliary_evidence']]
        return cls(model_id,report,benchmark=BenchmarkResult(raw[0]) if len(raw)==1 else None)

    @classmethod
    def _restore(cls, data):
        d = plain(data)
        if set(d)!= {'model_id','report','records','structure','benchmark'}:
            raise ModelComparisonError('invalid_source_fields')
        report = UnifiedAnalysisReport(d['report'])
        # Explicit selections may contain multiple auto-discovered candidates.
        # Rebuild those from the original report, never silently disambiguate.
        base = cls(d['model_id'],report,
            structure=None if d['structure'] is None else ModelStructureSnapshot(d['structure']),
            benchmark=None if d['benchmark'] is None else BenchmarkResult(d['benchmark']))
        available = {r['record_id']:r for r in d['report']['index']['records']}
        if type(d['records']) is not dict:
            raise ModelComparisonError('invalid_source_record_selection')
        for feature, ids in d['records'].items():
            FeaturePlanRequest('single',(ModelContext(d['model_id']),),features=(feature,))
            if type(ids) is not list or not ids or len(set(ids))!=len(ids):
                raise ModelComparisonError('invalid_source_record_selection')
            for identity in ids:
                record = available.get(identity)
                if record is None or record['feature_id']!=feature or record['scope']['model_id'] not in (None,d['model_id']):
                    raise ModelComparisonError('record_feature_or_model_identity_mismatch')
        object.__setattr__(base,'_payload',canonical(d))
        return base


def _chosen(source, feature):
    data = source.to_dict(); ids = data['records'].get(feature,[])
    if len(ids)!=1:
        return None,'missing_selected_feature_evidence' if not ids else 'ambiguous_records;select_exact_record_id'
    return ids[0],None


def _selected_report(source, features):
    """Only selected source observations enter the per-model Advisor view."""
    data = source.to_dict()
    ids = {identity for f,values in data['records'].items() if f in features for identity in values}
    records = [EvidenceRecord(r) for r in data['report']['index']['records'] if r['record_id'] in ids]
    return EvidenceIndex(records).analyze(context=AnalysisContext(workload_id='comparison-observations',
        task=data['report']['context']['task'],model={'id':source.model_id},
        dataset=data['report']['context']['dataset']))


def _benchmark_diagnostic(a, b, record_a, record_b):
    raw = (a.to_dict()['benchmark'],b.to_dict()['benchmark'])
    for benchmark, record in zip(raw,(record_a,record_b)):
        if benchmark is None:continue
        native = None if record is None else record['native_payload']
        if (not native or native.get('identity',{}).get('source_result_id')!=benchmark['identity']['run_id']
                or native.get('provenance')!={k:benchmark[k] for k in ('model','input','environment','provenance','configuration')}
                or native.get('raw_evidence',{}).get('latency_samples_ms')!=benchmark['timing']['latency_samples_ms']
                or native.get('raw_evidence',{}).get('source_sha256')!=hashlib.sha256(
                    json.dumps(benchmark,sort_keys=True,allow_nan=False).encode()).hexdigest()):
            return dict(status='incompatible',reason='auxiliary_benchmark_source_mismatch')
    if any(x is None for x in raw):
        return dict(status='not_available',reason='original_benchmark_not_supplied;no_remeasurement')
    return compare_results(BenchmarkResult(raw[0]),BenchmarkResult(raw[1]))


def _derive(a, b, features, include_structure):
    # Local imports retain the old Advisor <-> Analysis initialization boundary.
    from ..advisor import (NativeMetric, NativeMetricResolver, ComparisonContract,
        P26QualityComparisonContract, P26BoundQualityComparisonContract,
        P26ModelProfileComparisonContract, compare_performance, compare_p26_model_profiles)
    from ..advisor.performance import _SEMANTICS
    from ..structure_comparison import compare_model_structures
    request = FeaturePlanRequest('selected',(ModelContext(a.model_id),ModelContext(b.model_id)),features=features)
    if type(include_structure) is not bool:
        raise ModelComparisonError('include_structure_must_be_boolean')
    ra = NativeMetricResolver(UnifiedAnalysisReport(a.to_dict()['report']))
    rb = NativeMetricResolver(UnifiedAnalysisReport(b.to_dict()['report']))
    rows = []; benchmark = None
    for feature in request.features:
        ia, why_a = _chosen(a,feature); ib, why_b = _chosen(b,feature)
        if feature not in _METRICS:
            rows.append(dict(feature_id=feature,metric=None,status='NOT_SUPPORTED_IN_COMPARISON',difference=None,
                reasons=['comparison_adapter_not_connected;no_measurement_started'],comparison=None))
            continue
        if feature=='P01':
            benchmark = _benchmark_diagnostic(a,b,ra.record(ia),rb.record(ib))
        for metric_name in _METRICS[feature]:
            metric = NativeMetric(metric_name)
            row = dict(feature_id=feature,metric=metric_name,preferred_direction=_SEMANTICS[metric][2],
                domain=_SEMANTICS[metric][1],unit=_SEMANTICS[metric][0],difference=None,comparison=None,
                scope='process_wide_not_model_exclusive' if feature=='P10' else 'native_metric_and_contract_scope')
            if why_a or why_b:
                row.update(status='INSUFFICIENT_EVIDENCE',reasons=[why for why in (why_a,why_b) if why])
            else:
                if feature=='P26':
                    versions = [r.record(i)['native_payload'].get('schema_version') if r.record(i)['native_payload'] else None
                                for r,i in ((ra,ia),(rb,ib))]
                    contract_type = P26BoundQualityComparisonContract if versions==[3,3] else P26QualityComparisonContract
                    contract = contract_type(metric=metric,allowed_changes=('model',))
                else:
                    # Retain the existing default general comparison thresholds.
                    contract = ComparisonContract(metric=metric,allowed_changes=('model',))
                native = compare_performance(ra,ia,rb,ib,contract=contract)
                row.update(status=native.eligibility.status.value,comparison=native.to_dict(),
                    reasons=[reason.code+':'+reason.field for reason in native.eligibility.reasons],
                    difference=None if native.difference is None else native.difference.to_dict())
                if native.eligibility.status.value=='COMPARABLE' and native.difference is None:
                    row.update(status='INSUFFICIENT_EVIDENCE',reasons=row['reasons']+['native_numeric_difference_unavailable'])
                if feature=='P01' and benchmark and benchmark['status'] not in ('not_available','compatible'):
                    row.update(status='INCOMPATIBLE' if benchmark['status']=='incompatible' else 'INSUFFICIENT_EVIDENCE',
                               difference=None,reasons=row['reasons']+['original_benchmark_conditions_not_compatible'])
            rows.append(row)
    structure = dict(status='NOT_SELECTED',comparison=None,reasons=['structure_comparison_not_requested'])
    structure_object = None
    if include_structure:
        sa,sb = a.to_dict()['structure'],b.to_dict()['structure']
        if (sa is None)!=(sb is None):
            structure = dict(status='INSUFFICIENT_EVIDENCE',comparison=None,reasons=['both_explicit_structure_snapshots_required'])
        elif sa is not None and sb is not None:
            structure_object = compare_model_structures(ModelStructureSnapshot(sa),ModelStructureSnapshot(sb))
            structure = dict(status='COMPARABLE',scope='explicit_retained_registered_metadata;model_association_declared',
                             reasons=['not_a_P11_or_runtime_measurement'],comparison=structure_object.to_dict())
        elif 'P26' in request.features:
            ia,wa = _chosen(a,'P26');ib,wb = _chosen(b,'P26')
            if not wa and not wb:
                try:
                    profile = compare_p26_model_profiles(ra,ia,rb,ib,
                        contract=P26ModelProfileComparisonContract(allowed_changes=('model',)))
                    structure_object = profile.structure_comparison
                    structure = dict(status='COMPARABLE',scope='native_P26_bound_registered_metadata',
                        reasons=['not_a_P11_or_runtime_measurement'],comparison=structure_object.to_dict(),
                        original_profile_sha256=profile.content_sha256)
                except ValueError as error:
                    structure = dict(status='INSUFFICIENT_EVIDENCE',comparison=None,reasons=[str(error)])
            else:
                structure = dict(status='INSUFFICIENT_EVIDENCE',comparison=None,reasons=['missing_selected_P26_binding'])
        else:
            structure = dict(status='INSUFFICIENT_EVIDENCE',comparison=None,reasons=['two_retained_snapshots_or_selected_bound_P26_required'])
        if structure['comparison'] is not None:
            # Sources already retain snapshots; the projection keeps its original
            # native hash. It is explicitly not a standalone structure artifact.
            del structure['comparison']['source_snapshots']
    sources = dict(baseline=a.to_dict(),candidate=b.to_dict())
    data = dict(format='hyso.two_model_comparison',schema_version=1,sources=sources,
        selection=dict(features=list(request.features),include_structure=include_structure),
        policy='existing_native_comparison_gates;model_change_only;fixed_batch_precision_and_protocol',
        metrics=rows,structure=structure,benchmark_compatibility=benchmark,
        claims=dict(global_winner=None,statistical_superiority='not_established',acceptance='NOT_EVALUATED'),
        limitations=list(_LIMITS))
    data['content_sha256'] = digest(data)
    return data,structure_object


@dataclass(frozen=True, init=False)
class ModelComparisonResult:
    """Immutable, rederived JSON result. Differences are candidate minus baseline."""
    _payload: str
    _structure: Any

    def __init__(self, data: dict[str,Any]) -> None:
        try:
            d = plain(data)
            if len(canonical(d).encode('utf-8'))>_MAX_BYTES:
                raise ModelComparisonError('comparison_size_limit')
            if d.get('format')!='hyso.two_model_comparison' or type(d.get('schema_version')) is not int or d['schema_version']!=1:
                raise ModelComparisonError('unsupported_comparison_version')
            a,b = (ModelComparisonSource._restore(d['sources'][side]) for side in ('baseline','candidate'))
            expected,structure = _derive(a,b,d['selection']['features'],d['selection']['include_structure'])
            if canonical(d)!=canonical(expected):
                raise ModelComparisonError('comparison_derivation_or_integrity_mismatch')
        except ModelComparisonError:
            raise
        except (ValueError,TypeError,KeyError,AttributeError,IndexError,RecursionError,OverflowError) as error:
            raise ModelComparisonError('invalid_comparison_evidence') from error
        object.__setattr__(self,'_payload',canonical(expected));object.__setattr__(self,'_structure',structure)

    def to_dict(self) -> dict[str,Any]:
        return json.loads(self._payload)

    @property
    def structure_comparison(self):
        """Original typed structure result, or None; JSON retains a labeled projection."""
        return self._structure

    def to_json(self, path=None, *, overwrite: bool = False) -> str:
        raw = self._payload+'\n'
        if path is not None: export(path,raw,overwrite)
        return raw

    @classmethod
    def from_json(cls, path) -> ModelComparisonResult:
        data,_,_ = read_json(path,max_bytes=_MAX_BYTES)
        return cls(data)

    def to_display_dict(self, *, language: str = 'en') -> dict[str,Any]:
        from ..advisor.model_comparison_presentation import to_display_dict
        return to_display_dict(self,language=language)

    def to_markdown(self, path=None, *, language: str = 'en', overwrite: bool = False) -> str:
        from ..advisor.model_comparison_presentation import to_markdown
        return to_markdown(self,path,language=language,overwrite=overwrite)

    def tracking_config(self) -> dict[str,Any]:
        """Existing Manager benchmark-run config; no run or store is changed."""
        data = self.to_dict()
        return dict(comparison_sha256=data['content_sha256'],selection=data['selection'],
                    models={side:source['model_id'] for side,source in data['sources'].items()},policy=data['policy'])


def compare_models(baseline: ModelComparisonSource, candidate: ModelComparisonSource, *,
                   features: str | Sequence[str], include_structure: bool = False) -> ModelComparisonResult:
    """Compare exactly two existing sources, only explicit feature IDs.

    Uses compare_performance/P26 policies and optional original benchmark/structure
    compatibility. No missing test is run. No caller metadata is promoted into a
    producer capture. P29 compare_performance_quality is deliberately not called:
    that API performs new measurements. No global winner is computed.
    """
    if type(baseline) is not ModelComparisonSource or type(candidate) is not ModelComparisonSource:
        raise ModelComparisonError('exactly_two_retained_model_sources_required')
    selected = (features,) if type(features) is str else features
    data,_ = _derive(baseline,candidate,selected,include_structure)
    return ModelComparisonResult(data)
