"""Native executor CLI: pure planning, pinned retained sources, deferred factories."""
from dataclasses import replace
import os
from pathlib import Path
from .common import (Outcome, command, output_check, read_json, export_text, json_text,
                     require, factory_options, factory_kwargs, load_factory)


def register(sub):
    p = command(sub, 'analysis', 'Query P01–P35 and explicitly review/execute Stage 10 native operations.', run, export=False)
    s = p.add_subparsers(dest='action', required=True)
    command(s, 'list', 'List all 35 native executor feature descriptions, dependencies, device limits and cost classes.', run)
    q = command(s, 'show', 'Describe one exact feature ID, its requirements and native executor coverage.', run)
    q.add_argument('feature', help='Exact ID P01 through P35.')
    for action, desc in [('plan', 'Build a pure feature selection plan; never load factories or start native work.'),
                         ('run', 'Default: metadata preflight. --execute requires explicit feature authorization; retained/offline derivation also uses the native executor.')]:
        q = command(s, action, desc, run)
        choice = q.add_mutually_exclusive_group(required=True)
        choice.add_argument('--features', nargs='+', help='Exact explicitly selected P01–P35 IDs; dependencies are never auto-executed.')
        choice.add_argument('--profile', choices=('basic', 'inference', 'training', 'memory', 'comparison', 'full'))
        q.add_argument('--model-id', default='model', help='Caller model label (default: model), not verified checkpoint identity.')
        q.add_argument('--models', type=Path, help='Optional JSON array of ModelContext keyword objects, including explicit model_id labels.')
        q.add_argument('--device', default='cpu', help='Execution device cpu or cuda:INDEX; offline CUDA evidence is analyzed on cpu.')
        q.add_argument('--capabilities', nargs='*', default=[], help='Planner declarations only: gpu ncu power_sensor multiple_devices ort_cpu ort_cuda.')
        q.add_argument('--policy', type=Path, help='Optional CPUAnalysisPolicy JSON keyword object; budgets appear in the plan, not a hard resource sandbox.')
        if action == 'run':
            source = q.add_mutually_exclusive_group()
            source.add_argument('--source', type=Path, help='Pinned retained native JSON, including P19 -> P22 or P31 -> P32/P33 offline derive.')
            source.add_argument('--factory', help='Trusted module:callable returning a list/tuple of NativeBinding; deferred until explicit execution.')
            q.add_argument('--sha256', help='Required expected lowercase SHA256 when --source is supplied. Never inferred as acceptance.')
            q.add_argument('--source-options', type=Path, help='JSON per-feature derive options, e.g. {"P32":{"warmup_seconds":0}}.')
            q.add_argument('--companions', type=Path, help='JSON array of {path,sha256} pinned P21 companion files; paths relative to this manifest.')
            q.add_argument('--factory-args', type=Path, help='Plain JSON factory keyword object. No executable JSON source descriptors.')
            q.add_argument('--trust-factory', action='store_true', help='Authorize trusted Python import/call during execution only; arbitrary code is not sandboxed.')
            q.add_argument('--execute', action='store_true', help='Authorize execution after native metadata preflight (default: dry-run).')
            q.add_argument('--authorize', nargs='+', help='Exact selected feature subset to execute. Required with --execute; never inferred from --features.')
            for flag, desc in [('capture', 'new measurement/capture'), ('training', 'training episodes'),
                               ('cuda', 'CUDA access'), ('camera', 'camera access'), ('long-run', 'long-running work'),
                               ('power-sampling', 'physical power sensors'), ('file-writes', 'native artifact/checkpoint writes')]:
                q.add_argument('--allow-' + flag, action='store_true', help='Separately authorize ' + desc + '.')
            peak = q.add_mutually_exclusive_group()
            peak.add_argument('--allow-peak-reset', action='store_true', help='Explicitly authorize allocator peak reset in this CLI process only; uses its PID.')
            peak.add_argument('--peak-reset-pid', type=int, help='Explicit allocator owner PID; native admission accepts this CLI PID only. Prefer --allow-peak-reset in a shell.')


def _selection(args):
    from hyso import analysis as api
    contexts = [api.ModelContext(args.model_id)] if args.models is None else read_json(args.models)
    if args.models is not None:
        require(type(contexts) is list and bool(contexts), '--models must contain a nonempty JSON array of ModelContext objects.')
        contexts = [api.ModelContext(**row) for row in contexts]
    policy = {} if args.policy is None else read_json(args.policy)
    require(type(policy) is dict, '--policy must be a CPUAnalysisPolicy keyword object.')
    return api.plan_analysis(contexts, features=args.features, profile=args.profile, capabilities=args.capabilities,
                             execution_mode='native', device=args.device, policy=api.CPUAnalysisPolicy(**policy))


def _code(rows, *, execution=False):
    selected = [r for r in rows if r['selection'] != 'prerequisite' and r['status'] != 'SKIPPED']
    good = 'COMPLETED' if execution else 'SUPPORTED'
    successes = sum(r['status'] == good for r in selected)
    failures = [r['status'] for r in selected if r['status'] != good]
    if not failures:
        return (0, 'completed' if execution else 'dry_run') if successes else (3, 'blocked')
    if successes:
        return 6, 'partial'
    if 'FAILED' in failures:
        return 5, 'failed'
    if all(s in ('UNSUPPORTED', 'NOT_IMPLEMENTED_IN_EXECUTOR') for s in failures):
        return 4, 'unsupported'
    return 3, 'blocked'


def _policy(args):
    from hyso.analysis import OperationPolicy
    values = {name: getattr(args, name) for name in
        ('allow_capture', 'allow_training', 'allow_cuda', 'allow_camera', 'allow_long_run',
         'allow_power_sampling', 'allow_file_writes', 'peak_reset_pid')}
    if args.allow_peak_reset:
        values['peak_reset_pid'] = os.getpid()
    return OperationPolicy(**values)


def _retained(args, plan):
    from hyso import analysis as api
    from hyso.analysis._codec import read_json as pinned_read
    require(args.sha256 is not None, '--source requires --sha256 EXPECTED_HASH. Compute/review the artifact hash before invoking this command.')
    payload, digest, size = pinned_read(args.source, expected_sha256=args.sha256)
    artifacts = [api.ArtifactSource(str(args.source.resolve()), digest, size)]
    options = {} if args.source_options is None else read_json(args.source_options)
    require(type(options) is dict, '--source-options must contain a JSON object of derive options.')
    companions = []
    if args.companions is not None:
        manifest = read_json(args.companions)
        require(type(manifest) is list, '--companions must contain an array of {path,sha256} entries.')
        for entry in manifest:
            require(type(entry) is dict and set(entry) == {'path', 'sha256'}, 'Each companion needs exactly path and sha256.')
            path = Path(entry['path'])
            if not path.is_absolute():
                path = args.companions.parent / path
            output_check(args, inputs=(path,))
            value, digest, size = pinned_read(path, expected_sha256=entry['sha256'])
            companions.append(value)
            artifacts.append(api.ArtifactSource(str(path.resolve()), digest, size))
    models = tuple(m.model_id for m in plan.request.models)
    require(len(models) <= 2, 'A retained operation can bind at most two explicit model labels.')
    return [api.NativeBinding('cli-retained', models, plan.selected_features,
            api.RetainedSource(payload, options, tuple(companions)), device=args.device,
            policy=_policy(args), artifacts=tuple(artifacts))]


def _factory_guards(args, selected):
    # These checks happen BEFORE trusted import. Native admission rechecks typed
    # sources and narrowed authorization afterwards; a factory grants no consent.
    require(args.trust_factory, 'Execution with --factory requires --trust-factory. Dry-run never imports it.', code=3)
    require(args.allow_capture, 'A live binding factory requires --allow-capture before import.', code=3)
    if set(selected) & {f'P{i:02}' for i in range(19, 26)}:
        require(args.allow_training, 'Live training-feature factories require --allow-training before import. For offline P22 use --source.', code=3)
    if args.device.startswith('cuda:'):
        require(args.allow_cuda, 'CUDA binding factories require separate --allow-cuda before import.', code=3)
    if 'P30' in selected:
        require(args.allow_camera, 'P30 binding factories require --allow-camera before import (conservative CLI factory boundary).', code=3)
    if 'P31' in selected:
        require(args.allow_long_run, 'P31 binding factories require --allow-long-run before import.', code=3)


def run(args):
    from hyso import analysis as api
    output_check(args, inputs=tuple(getattr(args, n, None) for n in
                 ('models', 'policy', 'source', 'source_options', 'companions', 'factory_args')))
    if args.action in ('list', 'show'):
        data = [x.to_dict() for x in api.feature_catalog()] if args.action == 'list' else api.feature_catalog(args.feature).to_dict()
        export_text(args, json_text(data))
        text = None if args.action != 'list' else 'ID   FEATURE                                      COST / DEVICES\n' + ''.join(
            f"{r['feature_id']}  {r['name']:<44} {r['cost_class']} / {','.join(r['supported_devices'])}\n" for r in data)
        return Outcome(data, text=text)
    plan = _selection(args)
    if args.action == 'plan':
        data = plan.to_dict()
        export_text(args, json_text(data))
        return Outcome(data)
    kwargs = factory_kwargs(args)
    require(args.source is not None or (args.sha256 is None and args.source_options is None and args.companions is None),
            '--sha256, --source-options and --companions require --source retained.json.')
    authorized = tuple(args.authorize or ())
    if args.execute:
        require(authorized and len(set(authorized)) == len(authorized) and set(authorized).issubset(plan.selected_features),
                '--execute requires --authorize with a unique nonempty subset of explicitly selected features.', code=3)
    else:
        require(args.authorize is None, '--authorize requires --execute; dry-run never grants feature consent.')
    bindings = _retained(args, plan) if args.source is not None else []
    if args.factory and args.execute:
        _factory_guards(args, plan.selected_features)
        supplied = load_factory(args.factory)(**kwargs)
        require(type(supplied) in (tuple, list) and all(type(b) is api.NativeBinding for b in supplied),
                'Analysis factory must return a list/tuple of NativeBinding; do not call native producers in the factory.')
        bindings = [replace(b, policy=_policy(args)) for b in supplied]
        for binding in bindings:
            options = getattr(binding.source, 'options', {})
            native_targets = [options.get(name) for name in ('trace_path', 'serialize_to', 'save_last_to')]
            output_check(args, inputs=[*[a.path for a in binding.artifacts], *native_targets])
            directory = getattr(binding.source, 'directory', None)
            if args.output is not None and directory is not None:
                require(not args.output.resolve().is_relative_to(Path(directory).resolve()),
                        'CLI export must be outside the native runtime directory. Choose a separate --output file.')
    dry = api.preflight_analysis(plan, bindings)
    if not args.execute:
        data = dry.to_dict()
        data['cli_source'] = dict(factory=args.factory, factory_loaded=False,
                                  contract='factory(**kwargs) -> list[NativeBinding]; execution only; real caller-owned resources')
        code, status = _code(data['features'])
    else:
        result = api.execute_analysis(dry, authorized_features=authorized, explicit_authorization=True)
        data = result.to_dict()
        code, status = _code(data['features'], execution=True)
    export_text(args, json_text(data))
    return Outcome(data, code, status)
