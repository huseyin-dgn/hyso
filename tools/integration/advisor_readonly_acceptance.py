"""Read retained MNIST evidence and explicitly export Advisor artifacts.

No model, workload, benchmark, device query, training or factory is invoked.
The output directory must be new; existing evidence is never overwritten.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    import torch
    assert not torch.cuda.is_initialized()
    calls = []
    def deny(label):
        def blocked(*a, **k):
            calls.append(label)
            raise AssertionError('Forbidden runtime call: ' + label)
        return blocked
    # Separate function identities preserve Dynamo's import-time rule registry.
    for name in ('init', 'is_available', 'device_count', 'current_device'):
        setattr(torch.cuda, name, deny('cuda.' + name))
    torch.nn.Module._call_impl = deny('Module.forward')
    def audit(event, values):
        if event == 'import' and values[0] in ('cnn_factories', 'hyso.models.modular'):
            calls.append('source_factory_import:' + values[0])
            raise AssertionError('Source model/factory import forbidden')
    sys.addaudithook(audit)
    from hyso.benchmark import BenchmarkRunner
    from hyso.training import Trainer
    BenchmarkRunner.run = deny('BenchmarkRunner.run')
    Trainer.fit = deny('Trainer.fit')
    from hyso.analysis import UnifiedAnalysisReport
    from hyso.advisor import (AdvisorReport, AdvisorRequest, FindingKind, NativeMetric as M,
                              NativeMetricResolver, NativeObservation, advise, validate_advisor_report)
    source_hash = hashlib.sha256(args.source.read_bytes()).hexdigest()
    source = UnifiedAnalysisReport.from_json(args.source)
    resolver = NativeMetricResolver(source)
    result = advise(source)
    observed = [f for f in result.findings if type(f) is NativeObservation]
    records = {r['feature_id']: r for r in source.to_dict()['index']['records']}
    checks = []
    def check(name, condition):
        assert condition, name
        checks.append(dict(check=name, status='PASS'))
    def metric(feature, identity, **kwargs):
        found = resolver.resolve(records[feature]['record_id'], identity, **kwargs)
        assert found.status == 'supported', found.reasons
        return found.resolved
    p95 = metric('P01', M.FORWARD_P95)
    check('P01 P95 identity and value', p95.value == .86681 and p95.native_context['quantile'] == .95
          and p95.native_context['measurement_scope'] == 'synchronous_host_forward')
    throughput = metric('P02', M.COMPLETED_IMAGES_PER_SECOND)
    check('P02 batch32 completed images/s', throughput.native_context['batch'] == 32
          and abs(throughput.value - 12442.170637760393) < 1e-9 and throughput.reference.unit == 'images/s')
    accuracy = metric('P26', M.TEST_ACCURACY)
    check('P26 fraction and macro F1', accuracy.value == .989 and accuracy.reference.unit == 'fraction'
          and abs(metric('P26', M.TEST_MACRO_F1).value - .9889074263059602) < 1e-12)
    e2e = metric('P03', M.E2E_P95)
    check('P03 separate E2E', e2e.reference.domain == 'e2e' and e2e.value == 4.642905
          and resolver.compare(p95.reference, p95.metric, e2e.reference, e2e.metric).status == 'insufficient_evidence')
    point = next(p for p in records['P08']['native_payload']['raw_evidence']['points'] if p['precision'] == 'bf16')
    bf16 = metric('P08', M.PRECISION_MEAN, point_id=point['point_id'])
    check('P08 BF16 observation only', bf16.value == 1.618805 and bf16.native_context['precision'] == 'bf16'
          and all(f.kind in (FindingKind.OBSERVATION, FindingKind.MEASURE_NEXT) for f in result.findings))
    sweep_rate = metric('P08', M.PRECISION_THROUGHPUT, point_id=point['point_id'])
    check('P08 native throughput domain', sweep_rate.reference.domain == 'throughput'
          and any('Projection domain disagrees' in s for s in sweep_rate.limitations))
    check('P05 synthetic input preserved', records['P05']['scope']['input_kind'] == 'synthetic_recipe'
          and any('P05' in s and 'synthetic_recipe' in s for s in result.evidence_limitations))
    risks = '\n'.join(result.risks)
    check('P18 strong roof acceptance not established', 'qualified_stable_roof=not_established_by_this_report' in risks
          and any('calibration FAIL' in s for s in result.risks))
    check('P29 no independently verified improvement', 'independently_verified_improvement=not_certified' in risks)
    check('P31 short run not 7200 acceptance', records['P31']['native_payload']['analysis']['duration_seconds'] < 7200
          and 'long_run_7200_acceptance=not_certified' in risks)
    check('B4 remains OPEN', any('B4' in s and 'OPEN' in s for s in result.risks))
    check('All observation references and structured metrics eligible', all(
        resolver.check_metric(f.resolved_metric).status == 'supported' for f in observed))
    base = {f.finding_id: f.to_dict() for f in result.findings}
    priority_details = {}
    for priority in ('speed', 'memory', 'preserve_accuracy', 'balanced'):
        variant = advise(source, request=AdvisorRequest(priority=priority))
        check('Priority ' + priority + ' affects order only',
              {f.finding_id: f.to_dict() for f in variant.findings} == base and variant.risks == result.risks
              and variant.evidence_status == result.evidence_status)
        priority_details[priority] = variant.findings[0].resolved_metric.metric.value
    args.output.mkdir(parents=True, exist_ok=False)
    result.to_json(args.output / 'mnist-advisor.json')
    result.to_markdown(args.output / 'mnist-advisor.md')
    reopened = AdvisorReport.from_json(args.output / 'mnist-advisor.json')
    check('Advisor JSON/Markdown roundtrip', reopened == result and reopened.to_markdown() == result.to_markdown())
    check('Advisor derivation against source', validate_advisor_report(reopened, source))
    check('Source bytes preserved', hashlib.sha256(args.source.read_bytes()).hexdigest() == source_hash)
    check('No runtime/factory/device calls', not calls and not torch.cuda.is_initialized())
    summary = dict(status='PASS', passed=len(checks), failed=0, checks=checks,
                   source_report_id=resolver.source_report_id, source_file_sha256=source_hash,
                   source_canonical_sha256=resolver.source_report_sha256,
                   advisor_report_id=result.report_id, observations=len(observed),
                   measure_next=sum(f.kind == FindingKind.MEASURE_NEXT for f in result.findings),
                   priority_first_metric=priority_details, forbidden_calls=calls, cuda_initialized=False,
                   output_files=['mnist-advisor.json', 'mnist-advisor.md'])
    (args.output / 'acceptance.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
