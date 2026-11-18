"""TrainingConfig validation and explicit caller-owned TrainingSession execution."""
from pathlib import Path
from .common import (CLIError, Outcome, command, output_check, export_text, json_text,
                     require, factory_options, factory_kwargs, load_factory)


def register(sub):
    p = command(sub, 'train', 'Validate training settings or run a trusted real model/dataset TrainingSession.', run, export=False)
    s = p.add_subparsers(dest='action', required=True)
    command(s, 'options', 'Show supported optimizer, scheduler, precision and device contracts without probing hardware.', run)
    q = command(s, 'validate', 'Validate exact TrainingConfig schema1 JSON without loading models/data.', run)
    q.add_argument('config', type=Path)
    q = command(s, 'run', 'Default: validate and review only. Execution calls factory(config, **kwargs) -> prepared TrainingSession.', run)
    q.add_argument('config', type=Path)
    factory_options(q)
    for flag, help in [('execute', 'Authorize this run (also requires --allow-training and --trust-factory).'),
                       ('allow-training', 'Authorize one TrainingSession.fit; no implicit retry.'),
                       ('allow-cuda', 'Separately authorize the explicit TrainingConfig CUDA device.'),
                       ('allow-file-writes', 'Authorize TrainingConfig save_last_to checkpoint writes.')]:
        q.add_argument('--' + flag, action='store_true', help=help)


def run(args):
    from hyso.training import TrainingConfig, TrainingSession
    output_check(args, inputs=(getattr(args, 'config', None), getattr(args, 'factory_args', None)))
    if args.action == 'options':
        data = dict(optimizers=['sgd', 'adam', 'adamw'], schedulers=[None, 'step', 'cosine', 'onecycle', 'plateau'],
                    precision=['fp32', 'fp64', 'amp_fp16'], precision_constraints={'amp_fp16': 'requires cuda:INDEX'},
                    device=['cpu', 'cuda:INDEX'],
                    num_workers=[0], default_config=TrainingConfig().to_dict(),
                    limits='Classification only; caller must supply correctly placed/dtyped model and matching real loaders.')
    else:
        config = TrainingConfig.from_json(args.config)
        data = dict(valid=True, config=config.to_dict(), workload_started=False)
        if args.action == 'run':
            kwargs = factory_kwargs(args)
            settings = config.to_dict()
            requested = settings
            missing = []
            for condition, flag in [(args.factory is not None, '--factory module:callable'),
                                    (args.trust_factory, '--trust-factory'), (args.allow_training, '--allow-training'),
                                    (not settings['device'].startswith('cuda:') or args.allow_cuda, '--allow-cuda'),
                                    (settings['save_last_to'] is None or args.allow_file_writes, '--allow-file-writes')]:
                if not condition:
                    missing.append(flag)
            data.update(factory=args.factory, factory_loaded=False, execution_authorized=False,
                        blockers=missing, budget=dict(epochs=settings['epochs'], batch_size=settings['batch_size']),
                        source_contract='factory(config: TrainingConfig, **kwargs) -> prepared TrainingSession with actual model/loaders')
            if not args.execute:
                export_text(args, json_text(data))
                return Outcome(data, 3 if missing else 0, 'blocked' if missing else 'dry_run')
            require(not missing, 'Execution blocked. Add: ' + ', '.join(missing) + '.', code=3)
            if settings['save_last_to'] is not None:
                destination = Path(settings['save_last_to'])
                require(not destination.exists(), 'Checkpoint destination exists. Choose a new save_last_to in TrainingConfig; --overwrite applies only to CLI exports.', code=3)
                require(destination.parent.is_dir(), 'Checkpoint parent directory is missing. Create it before execution.')
                require(args.output is None or args.output.resolve() != destination.resolve(), 'Report output cannot equal the checkpoint destination.')
            if settings['resume_from'] is not None:
                require(Path(settings['resume_from']).is_file(), 'resume_from checkpoint does not exist. Correct TrainingConfig before execution.')
                require(args.output is None or args.output.resolve() != Path(settings['resume_from']).resolve(), 'Report output cannot replace the resume checkpoint.')
            session = load_factory(args.factory)(config, **kwargs)
            require(type(session) is TrainingSession and session.settings()['state'] == 'prepared',
                    'Training factory must return a fresh prepared TrainingSession, without calling fit().')
            require(session.settings()['requested'] == requested, 'Factory TrainingSession does not match the reviewed TrainingConfig.')
            try:
                result = session.fit()
            except Exception as exc:
                raise CLIError('Training failed; state may have advanced. Inspect native checkpoints before retrying: ' + str(exc), 5) from exc
            data = result.to_dict()
    export_text(args, json_text(data))
    return Outcome(data)
