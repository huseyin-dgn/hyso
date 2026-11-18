"""Explicit tiny CPU captures, bounded separately from offline regression tests."""
from pathlib import Path

import pytest
import torch

from hyso import analysis as api, performance as perf


@pytest.fixture(autouse=True)
def no_GPU_training_or_benchmark(monkeypatch):
    from hyso.training import Trainer
    from hyso.benchmark import BenchmarkRunner
    def forbidden(*args, **kwargs):
        pytest.fail('bounded CPU smoke reached GPU, training or benchmark')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(Trainer, 'fit', forbidden)
    monkeypatch.setattr(BenchmarkRunner, 'run', forbidden)


def test_one_owned_CPU_profile_for_P12_P13_P14():
    recipe = perf.ColdStartRecipe('profiling_factories:spatial', input_shape=(1, 2, 3, 3),
        factory_search_path=str(Path(__file__).parent.resolve()))
    protocol = perf.ProfilingProtocol(warmup_iterations=0, measure_iterations=1,
        timeout_seconds=30., input_limit_bytes=65536, max_events=1000, max_trace_bytes=1024**2)
    binding = api.NativeBinding('tiny-owned-profile', ('synthetic',), ('P12', 'P13', 'P14'),
        api.RecipeSource(recipe, options=dict(protocol=protocol)), policy=api.OperationPolicy(allow_capture=True))
    plan = api.plan_analysis(api.ModelContext('synthetic'), features=binding.features, execution_mode='native')
    result = api.execute_analysis(api.preflight_analysis(plan, (binding,)),
        authorized_features=binding.features, explicit_authorization=True)
    rows = result.to_dict()['features']
    assert {row['status'] for row in rows} == {'COMPLETED'}, [(row['feature_id'], row['reasons']) for row in rows]
    assert len({row['execution_receipt']['source_sha256'] for row in rows}) == 1
    source = rows[0]['auxiliary_evidence']['native_sources']['operator_profile']
    assert source['actual']['measured_forwards'] == 1
    assert source['provenance']['environment']['device'] == 'cpu'
    assert api.AnalysisExecutionResult.from_dict(result.to_dict()).to_dict() == result.to_dict()


def test_two_sample_CPU_pipeline_and_stage_projection():
    from examples.analysis.native_workflow import cpu_bindings
    bindings = cpu_bindings(['P03', 'P04'])
    plan = api.plan_analysis(api.ModelContext('example'), features=('P03', 'P04'), execution_mode='native')
    result = api.execute_analysis(api.preflight_analysis(plan, bindings), authorized_features=('P03', 'P04'), explicit_authorization=True)
    rows = result.to_dict()['features']
    assert {row['status'] for row in rows} == {'COMPLETED'}, rows
    assert rows[0]['execution_receipt'] == rows[1]['execution_receipt']
