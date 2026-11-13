"""Read existing Manager stores without migration, deletion or run mutation."""
from pathlib import Path
from .common import Outcome, command, output_check, export_text, json_text, existing_manager


def register(sub):
    p = command(sub, 'experiments', 'Read existing local ExperimentManager stores; no creation, migration or deletion.', run, export=False)
    s = p.add_subparsers(dest='action', required=True)
    for name, help in [('list', 'List a page of experiments with a sequence cursor.'),
                       ('history', 'Show experiment run history, authoritative states and schema1/schema2 support.'),
                       ('run', 'Show run details, metrics and artifact availability.'),
                       ('artifacts', 'List artifact metadata; schema2 required.'),
                       ('verify', 'Verify a registered artifact size/hash; Manager appends a verification observation, preserving expected identity.'),
                       ('open', 'Open and validate a registered workflow result; preserve unavailable/invalid statuses.')]:
        q = command(s, name, help, run)
        q.add_argument('--root', required=True, type=Path, help='Existing dedicated local store, outside synced/network folders.')
        if name in ('history', 'run', 'verify', 'open'):
            q.add_argument('id', help='Experiment ID, run ID or artifact ID, as appropriate.')
        if name in ('verify', 'open'):
            q.add_argument('--path', type=Path, help='Explicit replacement locator for a reference artifact, with original identity verified.')
        if name in ('list', 'history', 'artifacts'):
            q.add_argument('--after-seq', type=int, help='Sequence cursor from the previous page.')
            q.add_argument('--limit', type=int, default=100, help='Page size, subject to Manager bounds (default: 100).')
        if name == 'artifacts':
            q.add_argument('--run-id', help='Optional exact run filter.')
        if name == 'list':
            q.add_argument('--name', help='Optional exact experiment name filter.')


def run(args):
    # Avoid writing exports into a store that must remain dedicated/read-only.
    from .common import require
    output_check(args, inputs=(getattr(args, 'path', None),))
    if args.output is not None:
        require(not args.output.resolve().is_relative_to(args.root.resolve()), 'Export outside the ExperimentManager store directory.')
    manager = existing_manager(args.root)
    try:
        if args.action == 'list':
            rows = manager.list_experiments(name=args.name, after_seq=args.after_seq, limit=args.limit)
            data = dict(experiments=rows, next_after_seq=rows[-1]['seq'] if len(rows) == args.limit else None)
        elif args.action == 'history':
            data = manager.get_experiment_history(args.id, after_seq=args.after_seq, limit=args.limit)
        elif args.action == 'run':
            data = manager.get_run_details(args.id)
        elif args.action == 'artifacts':
            rows = manager.list_artifacts(args.run_id, after_seq=args.after_seq, limit=args.limit)
            data = dict(artifacts=rows, next_after_seq=rows[-1]['seq'] if len(rows) == args.limit else None)
        else:
            data = (manager.verify_artifact if args.action == 'verify' else manager.open_result)(args.id, path=args.path)
            if data.get('result') is not None:
                data = {**data, 'result': data['result'].to_dict()}
        export_text(args, json_text(data))
        status = data.get('status') if type(data) is dict else None
        if args.action in ('open', 'verify') and status != 'verified':
            return Outcome(data, 4 if status == 'unsupported' else 3, status or 'blocked')
        return Outcome(data)
    finally:
        manager.close()
