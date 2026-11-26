"""Source/evidence inventory. Presence, observed support and future work are separate."""
import ast
import json
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
USER=REPO/'outputs/final-user-acceptance-20261126'
CUDA=REPO/'outputs/cuda-functional-acceptance-20261126'
COMPONENTS=[
 ('ModelBuilder','hyso.ModelBuilder','hyso/models/builder.py','models.model-builder',['U03'],['test_pythonic_builders.py']),
 ('GraphBuilder','hyso.GraphModelBuilder','hyso/models/graph.py','models.graph-builder',['U04','U05'],['test_graph_builder.py']),
 ('External models','hyso.capture_model_structure; hyso.BenchmarkRunner','hyso/structure.py','models.support',['U01','U02'],['test_benchmark_state_safety.py']),
 ('Training','hyso.Trainer; hyso.TrainingConfig','hyso/training/trainer.py','training.trainer',['U05'],['test_trainer_updates.py']),
 ('Checkpoint/resume','hyso.checkpoint.save_training_checkpoint/load_training_checkpoint','hyso/checkpoint.py','training.checkpoints',['U06'],['test_training_checkpoint.py']),
 ('Evaluation','hyso.evaluate_classification','hyso/evaluation.py','analysis.p26',['U07'],['test_evaluation_structure_binding.py']),
 ('Quality/precision','hyso.evaluate_precision_quality','hyso/quality.py','analysis.p28',['U07'],['test_precision_mixed_output.py']),
 ('Benchmark','hyso.BenchmarkRunner','hyso/benchmark/runner.py','analysis.p01',['U01','U02','U07'],['test_benchmark_bounded_validation.py']),
 ('Performance P01-P35','hyso.performance','hyso/performance/__init__.py','analysis.overview',[],['test_cuda_functional_completion.py']),
 ('Analyzer','hyso.analysis.plan_analysis/preflight_analysis/execute_analysis','hyso/analysis/execution.py','analyzer.workflow',['U09'],['test_native_execution.py']),
 ('Advisor','hyso.advisor.advise/recommend','hyso/advisor/recommendations.py','advisor.recommendations',['U09'],['test_advisor_recommendations.py']),
 ('Model comparison','hyso.analysis.compare_models; hyso.compare_performance_quality','hyso/analysis/model_comparison.py','comparison.models',['U08'],['test_model_comparison.py']),
 ('Reports','hyso.reporting.create_report; PresentationReport','hyso/reporting/report.py','reporting.outputs',['U10'],['test_reporting.py']),
 ('CLI','python -m hyso','hyso/cli/main.py','api.python-cli',['U11'],['test_cli_presentation_languages.py']),
 ('Help/offline guide','help(); hyso docs; hyso_guide','hyso_guide.py','api.python-cli',['U11'],['test_contextual_help.py']),
 ('Autocomplete/typing','hyso.__all__; TYPE_CHECKING aliases; py.typed','hyso/__init__.py','api.python-cli',[],['test_product_experience.py']),
 ('Four-language presentation','hyso.reporting; hyso.advisor; hyso_guide','hyso/reporting/catalog.py','reporting.outputs',['U10'],['test_cli_presentation_languages.py']),
 ('Read-only model/data integrity','hyso.analysis; hyso.advisor; hyso.evaluation','hyso/evaluation.py','models.checks',['U12'],['test_evaluation_structure_binding.py']),
 ('Resource safety','owned controller; public measurement protocols','tools/final_acceptance/campaign_controller.py','support.cpu-gpu',[],['test_user_acceptance_runner.py']),
 ('Package exports','hyso.__all__; setup.cfg; commercial-profile.json','setup.cfg','getting-started.installation',[],['test_public_api.py']),
 ('Deployment analysis','hyso.performance.profile_execution_providers','hyso/performance/providers.py','analysis.p35',['U13'],['test_deployment_analysis.py']),
]


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def observation(cp,scenario,cpu=False):
    key='tests/test_user_acceptance_scenarios.py::test_'+scenario if cpu else scenario
    row=cp.get('cpu_results' if cpu else 'results',{}).get(key)
    if not row:return dict(status='NOT_OBSERVED',reason='No corresponding captured scenario; source presence is not verification')
    return dict(status=row['status'] if row.get('source_valid',True) else 'HISTORICAL_SOURCE_HOLD',
        case=key,evidence=row.get('evidence'),sha256=row.get('sha256'),scope='exact recorded model/device/protocol/runtime only')


def source_api_inventory():
    result={}
    for relative in ('hyso/__init__.py','hyso/models/builder.py','hyso/models/graph.py','hyso/training/trainer.py',
        'hyso/evaluation.py','hyso/quality.py','hyso/analysis/__init__.py','hyso/performance/__init__.py',
        'hyso/advisor/__init__.py','hyso/reporting/__init__.py'):
        tree=ast.parse((REPO/relative).read_text(encoding='utf-8'));rows=[]
        def public_nodes(body,prefix=''):
            for node in body:
                if isinstance(node,ast.ClassDef) and not node.name.startswith('_'):
                    yield from public_nodes(node.body,prefix+node.name+'.')
                elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and (not node.name.startswith('_') or prefix and node.name=='__init__'):
                    yield prefix+node.name,node
        for qualified,node in public_nodes(tree.body):
                rows.append(dict(name=qualified,line=node.lineno,return_annotation=ast.unparse(node.returns) if node.returns else None,
                    missing_parameter_annotations=[a.arg for a in [*node.args.posonlyargs,*node.args.args,*node.args.kwonlyargs] if a.arg not in ('self','cls') and a.annotation is None],
                    docstring=bool(ast.get_docstring(node))))
        result[relative]=rows
    return result


def export_inventory():
    """No export engines are built. External capability is not HYSo implementation."""
    rows=[]
    for technology in ('PyTorch -> ONNX','ONNX -> TensorRT','PyTorch -> TensorRT','TorchScript / LibTorch C++',
            'torch.export','AOTInductor C++','FP32 -> FP16','FP32 -> INT8','OpenVINO','TFLite'):
        onnx=technology=='PyTorch -> ONNX';fp16=technology=='FP32 -> FP16';int8=technology=='FP32 -> INT8'
        rows.append(dict(technology=technology,hyso_conversion_api='MISSING',
            external_conversion='torch.onnx.export used in U13 factory' if onnx else 'native PyTorch autocast recipe; not an FP16 export artifact' if fp16 else
                'torch.ao.quantization.quantize_dynamic Linear CPU recipe; not a deployment converter' if int8 else 'External technology; no HYSo integration or execution evidence',
            model_loading='ONNX bytes/feed factory into ORT' if onnx else 'no technology-specific loader',
            CPU_inference='U13 actual CPUExecutionProvider node events' if onnx else 'P28 CPU dynamic Linear' if int8 else 'NOT_OBSERVED for converted deployment artifact',
            CUDA_inference='U13 actual CUDAExecutionProvider node events' if onnx else 'AMP operator observations in U07; not converted artifact inference' if fp16 else 'NOT_OBSERVED',
            converted_model_analysis='P35 node/provider placement; not arbitrary converted-model Analyzer support' if onnx else 'MISSING',
            output_equivalence='MISSING paired PyTorch/converted equivalence; provider coverage is not equivalence',
            performance_comparison='provider host event timings only; no export counterfactual' if onnx else 'P28 quality, not conversion speed benchmark' if fp16 or int8 else 'MISSING',
            advisor_integration='retained P35 explanations; no conversion optimizer' if onnx else 'P28 retained evidence explanations only' if fp16 or int8 else 'MISSING',
            dependencies=['torch','onnx','onnxruntime (separate admitted GPU runtime where required)'] if onnx else ['torch'] if fp16 or int8 else [],
            source=['tools/user_acceptance/scenarios.py::onnx_model','hyso/performance/providers.py'] if onnx else ['hyso/performance/sweeps.py','hyso/quality.py'] if fp16 or int8 else [],
            tests=['U13','tests/test_deployment_analysis.py'] if onnx else ['U07','tests/test_precision_quality_completion.py'] if fp16 else ['tests/test_quality_experiments.py::test_actual_precision_recipes_quality'] if int8 else [],
            missing_work=['Independent explicit conversion API, artifact provenance, target runtime admission, equivalence and performance tests',
                'Actual C++ runtime compilation/execution required before any LibTorch/AOTInductor C++ support claim'] if 'C++' in technology else
                ['Independent explicit conversion API, target loader/runtime, output equivalence and safe performance evidence']))
    return rows


def build_inventory():
    cp=read(USER/'checkpoint.json');cuda=read(CUDA/'checkpoint.json')
    registry=read(REPO/'hyso/data/user-guide-topics.json');guide={r['id'] for r in registry['topics']}
    components=[]
    for name,api,source,topic,cases,tests in COMPONENTS:
        components.append(dict(feature=name,public_api=api,implemented=(REPO/source).is_file(),source=source,
            CPU=[observation(cp,c,True) for c in cases] or [dict(status='NOT_OBSERVED',reason='Use linked feature/campaign evidence, not source presence')],
            CUDA=[observation(cp,c) for c in cases] or [dict(status='NOT_OBSERVED',reason='No new GPU execution in U15-U18')],
            guide=dict(topic=topic,present=topic in guide,languages=registry['languages']),tests=tests,
            limitations='Recorded profiles only; arbitrary user Python is not sandboxed; no general hardware/architecture certification',
            release_work='U15-U18 acceptance pending; see typing/help/localization/export limitations'))
    features=[]
    apis=('analyze_latency','measure_throughput','measure_pipeline','measure_stages','measure_cold_start',
        'run_sweep','run_sweep','run_sweep','BenchmarkRunner','measure_memory','measure_footprint',
        'analyze_flops','profile_layers','analyze_nsys_operators','measure_transfers','measure_utilization',
        'find_max_batch_size','analyze_roofline','measure_training','measure_training_detailed',
        'measure_training_with_loader','analyze_training_throughput','measure_training_memory',
        'measure_time_to_quality','measure_gradient_diagnostics','evaluate_classification','evaluate_robustness',
        'evaluate_precision_quality','compare_performance_quality','measure_streaming','measure_stability',
        'analyze_memory_trend','analyze_energy','compare_devices','profile_execution_providers')
    topics={r['id']:r for r in read(REPO/'docs/user-guide/content.json')['topics']}
    for row,api in zip(registry['release_support']['features'],apis):
        record=cuda['results'].get(row['feature_id'])
        namespace='hyso' if row['feature_id'] in ('P09','P24','P26','P27','P28','P29') else 'hyso.performance'
        features.append(dict(feature=row['feature_id'],public_api=namespace+'.'+api,
            implementation='native/derived reader per feature catalog; source existence alone is not acceptance',
            release_support=row,CPU_evidence='outputs/release-acceptance-20261125/final-campaign/P01-P35-acceptance.json',
            CUDA_evidence=record,guide='analysis.'+row['feature_id'].lower(),
            method_scope_interpretation=topics['analysis.'+row['feature_id'].lower()]['translations'],
            tests=['tests/test_cuda_functional_completion.py'],
            missing='Deferred advanced scope remains closed' if row['feature_id'] in ('P18','P23','P31','P33') else 'Other hardware/runtime/workload profiles remain unverified'))
    return dict(schema_version=1,scope='U15-U18; metadata/offline only; no conversion development',
        components=components,performance_features=features,exports=export_inventory(),source_api_annotations=source_api_inventory(),
        language_surfaces=dict(offline_guide=['tr','en','de','es'],report_advisor_comparison=['tr','en','de','es'],
            Python_docstrings='primarily English; not four-language Python help',CLI_help='primarily English; localized presentation and docs subcommands are separate'),
        typing=dict(py_typed=True,real_IDE_tested=False,external_type_checker='not installed/not run',
            limits='Inline annotations and literal choices checked; dynamic metadata/factories retain Any. No strict whole-package typing certification.'),
        advisor_limits='Resolver supports a subset P01/P02/P03/P08/P10/P11/P26; other metrics have explanations/measurement requests, not invented optimization evidence.',
        full_cuda_acceptance=False,deferred=['P18 advanced counters','P23 saved activation/B4','P31 CUDA 7200s','P33 advanced energy/thermal'],
        extra_user_scenarios='U15 static discovery, U16 offline help, U17 retained multi-language presentation, U18 inventory/package metadata. No new P steps or repeated workloads.')
