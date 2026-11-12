"""Review typed native operations, analyze retained artifacts and export reports.

Run --help or a subcommand's --help for inputs. Capture commands first print a
dry-run. --execute grants feature consent. Training also needs --allow-training.
The built-in CPU data is a small synthetic technical example.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hyso import analysis as api, performance as perf
from hyso.analysis._codec import read_json
from hyso.advisor import recommend
from hyso.reporting import create_report


def write_result(result: api.AnalysisExecutionResult, output: Path) -> None:
    """Export execution, evidence, Advisor and JSON/Markdown/HTML presentation."""
    output.mkdir(parents=True, exist_ok=True)
    context = api.AnalysisContext(workload_id='explicit-native-example', task='classification',
        model={'id': 'example'}, dataset={'association': 'caller-declared; see native evidence'})
    result.to_json(output / 'execution.json')
    result.to_evidence_index().to_json(output / 'evidence.json')
    advice = recommend(result.to_unified_report(context=context))
    advice.to_json(output / 'recommendations.json')
    report = create_report(result, advice, context=context, title='HYSo native operation')
    report.to_json(output / 'report.json')
    report.to_markdown(output / 'report.md')
    report.to_html(output / 'report.html')


def cpu_bindings(features: list[str]) -> list[api.NativeBinding]:
    """Use explicit tiny CPU resources, including one shared P03/P04 pipeline."""
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
    inputs = torch.ones(2, 1, 2, 2)
    policy = api.OperationPolicy(allow_capture=True)
    bindings = []
    pipeline = tuple(feature for feature in features if feature in ('P03', 'P04'))
    if pipeline:
        sequence = iter(range(100))
        def sample() -> perf.PipelineSample:
            return perf.PipelineSample(str(next(sequence)), inputs)
        source = api.PipelineSource(model, sample, options=dict(protocol=perf.PipelineProtocol(
            samples=2, warmup_samples=0, repeats=1)))
        bindings.append(api.NativeBinding('pipeline', ('example',), pipeline, source, policy=policy))
    for feature in features:
        if feature in pipeline:
            continue
        if feature == 'P26':
            source = api.EvaluationSource(targets=[0, 0, 1, 1], predictions=[0, 1, 1, 1],
                options=dict(class_names=['a', 'b'], split='synthetic-demo'))
        else:
            options = dict(protocol=perf.ThroughputProtocol(batch_count=2, warmup_iterations=0, repeats=1)) if feature == 'P02' else {}
            if feature == 'P01':
                options = dict(warmup_iterations=0, measure_iterations=1000, repeats=1)
            source = api.InferenceSource(model, inputs, options=options)
        bindings.append(api.NativeBinding(feature, ('example',), (feature,), source, policy=policy))
    return bindings


def main() -> None:
    """Select an example, print preflight, then perform explicitly requested work."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    cpu = commands.add_parser('cpu', help='Synthetic CPU inference, pipeline, memory and classification')
    cpu.add_argument('--features', nargs='+', choices=['P01', 'P02', 'P03', 'P04', 'P10', 'P11', 'P26'], default=['P10', 'P11'])
    retained = commands.add_parser('artifact', help='Validate or derive from a saved native JSON artifact')
    retained.add_argument('--input', type=Path, required=True)
    retained.add_argument('--sha256', help='Expected source file hash, if known')
    retained.add_argument('--features', nargs='+', required=True)
    retained.add_argument('--warmup-seconds', type=float, default=0.)
    training = commands.add_parser('training', help='One synthetic CPU episode with shared P19/P20/P22')
    training.add_argument('--allow-training', action='store_true')
    quality = commands.add_parser('quality', help='CPU classification of four synthetic labeled samples')
    cuda = commands.add_parser('cuda-preflight', help='Review a native CUDA recipe using metadata')
    cuda.add_argument('--factory', required=True, help='Trusted importable module:factory')
    cuda.add_argument('--device', default='cuda:0')
    providers = commands.add_parser('providers', help='Explicit ONNX bytes/feed factory and CPU ORT profile')
    providers.add_argument('--factory', required=True, help='Factory returns existing self-contained ONNX bytes and numpy feed')
    providers.add_argument('--directory', required=True, help='Fresh owned native runtime directory')
    compare = commands.add_parser('compare', help='Compare two saved execution snapshots')
    compare.add_argument('--baseline', type=Path, required=True)
    compare.add_argument('--candidate', type=Path, required=True)
    compare.add_argument('--baseline-model', required=True)
    compare.add_argument('--candidate-model', required=True)
    compare.add_argument('--features', nargs='+', default=['P01', 'P02', 'P10', 'P11', 'P26'])
    compare.add_argument('--output', type=Path, required=True)
    for command in (cpu, retained, training, quality, cuda, providers):
        command.add_argument('--execute', action='store_true')
        command.add_argument('--output', type=Path, help='Fresh report directory when executing')
    args = parser.parse_args()
    if args.command == 'compare':
        baseline = api.ModelComparisonSource.from_execution(api.AnalysisExecutionResult.from_json(args.baseline), args.baseline_model)
        candidate = api.ModelComparisonSource.from_execution(api.AnalysisExecutionResult.from_json(args.candidate), args.candidate_model)
        result = api.compare_models(baseline, candidate, features=args.features)
        args.output.mkdir(parents=True, exist_ok=True)
        result.to_json(args.output / 'comparison.json')
        report = create_report(result, title='HYSo retained comparison')
        report.to_json(args.output / 'report.json')
        report.to_markdown(args.output / 'report.md')
        report.to_html(args.output / 'report.html')
        return
    if args.execute and args.output is None:
        parser.error('--execute requires a fresh --output directory')
    device = 'cpu'
    if args.command == 'cpu':
        features = args.features
        bindings = cpu_bindings(features)
    elif args.command == 'artifact':
        payload, _, _ = read_json(args.input, args.sha256)
        features = args.features
        options = {'P32': dict(warmup_seconds=args.warmup_seconds)} if 'P32' in features else {}
        source = api.RetainedSource(payload, options=options)
        bindings = [api.NativeBinding('retained', ('example',), tuple(features), source,
                                     artifacts=(api.ArtifactSource.from_path(args.input),))]
    elif args.command == 'training':
        from torch.utils.data import DataLoader, TensorDataset
        from hyso.training import TrainingConfig
        loader = DataLoader(TensorDataset(torch.ones(2, 2), torch.tensor([0, 1])), batch_size=2)
        session = TrainingConfig(epochs=1, batch_size=2, shuffle=False).prepare(torch.nn.Linear(2, 2), loader)
        features = ['P19', 'P20', 'P22']
        bindings = [api.NativeBinding('one-episode', ('example',), tuple(features),
            api.TrainingSource.from_session(session),
            policy=api.OperationPolicy(allow_capture=True, allow_training=args.allow_training))]
    elif args.command == 'quality':
        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(TensorDataset(torch.ones(4, 1, 2, 2), torch.tensor([0, 0, 1, 1])), batch_size=2)
        model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(4, 2))
        features = ['P26']
        source = api.EvaluationSource(model, loader, options=dict(class_names=['a', 'b'],
            split='synthetic-untrained-demo', device='cpu', capture_evidence=True))
        bindings = [api.NativeBinding('CPU-classification', ('example',), tuple(features), source,
                                     policy=api.OperationPolicy(allow_capture=True))]
    elif args.command == 'cuda-preflight':
        features, device = ['P14'], args.device
        bindings = [api.NativeBinding('CUDA-profile', ('example',), tuple(features),
            api.RecipeSource(perf.ColdStartRecipe(args.factory, device=device)), device=device)]
    else:
        features = ['P35']
        bindings = [api.NativeBinding('ORT', ('example',), tuple(features),
            api.RuntimeSource(perf.WorkloadRecipe(args.factory), args.directory,
                options=dict(providers=['CPUExecutionProvider'], protocol=perf.ProviderProtocol())),
            policy=api.OperationPolicy(allow_capture=args.execute, allow_file_writes=args.execute))]
    plan = api.plan_analysis(api.ModelContext('example'), features=features, execution_mode='native', device=device)
    review = api.preflight_analysis(plan, bindings)
    print(json.dumps(review.to_dict(), indent=2, ensure_ascii=False))
    if args.execute:
        result = api.execute_analysis(review, authorized_features=tuple(features), explicit_authorization=True)
        write_result(result, args.output)
        print(json.dumps(result.to_dict()['status_counts'], indent=2))


if __name__ == '__main__':
    main()
