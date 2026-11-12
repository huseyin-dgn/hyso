"""Registry metadata and explicit CPU model inspection."""
from pathlib import Path
from .common import Outcome, command, output_check, require, export_text, json_text


def register(sub):
    p = command(sub, 'models', 'List components, validate architecture JSON or explicitly inspect a new CPU model.', run, export=False)
    s = p.add_subparsers(dest='action', required=True)
    q = command(s, 'list', 'List ModelRegistry metadata without constructing models.', run)
    q.add_argument('--category', help='Exact registry category filter.')
    q = command(s, 'show', 'Describe a registered component and all supported parameters.', run)
    q.add_argument('name', help='Registered component name; see models list.')
    for name, desc in [('validate', 'Validate bounded ModelBuilder JSON; no model construction.'),
                       ('summary', 'Summarize declared architecture and normalized defaults; no workload.'),
                       ('inspect', 'Allocate a new untrained CPU model and inspect parameter/buffer shapes; no forward or checkpoint loading.')]:
        q = command(s, name, desc, run)
        q.add_argument('input', type=Path, help='ModelBuilder architecture JSON, not weights or a trained checkpoint.')
        if name == 'inspect':
            q.add_argument('--execute', action='store_true', help='Explicitly authorize new CPU model allocation for inspection.')


def run(args):
    from hyso.models.registry import ModelRegistry
    from hyso.models.builder import ModelBuilder
    output_check(args, inputs=(getattr(args, 'input', None),))
    if args.action == 'list':
        rows = [spec.to_dict() for spec in ModelRegistry().list(args.category)]
        data = dict(components=rows)
    elif args.action == 'show':
        data = ModelRegistry().get(args.name).to_dict()
    else:
        builder = ModelBuilder.from_json(args.input)
        config = builder.to_config()
        data = dict(valid=True, architecture=config, weights_included=False, trained_checkpoint=False,
                    model_constructed=False)
        if args.action == 'summary':
            data['summary'] = dict(kind=config['kind'], input_shape=config.get('input_shape'),
                num_classes=config.get('num_classes', config.get('params', {}).get('num_classes')),
                block_count=len(config['blocks']) if 'blocks' in config else None,
                blocks=[b['name'] for b in config.get('blocks', [])],
                scope='declared architecture; no activation tracing or weight restoration')
        if args.action == 'inspect':
            require(args.execute, 'models inspect requires --execute to allocate a new CPU model. Use models summary for metadata only.', code=3)
            model = builder.build()
            data.update(model_constructed=True, device='cpu', forward_executed=False,
                        parameters=sum(x.numel() for x in model.parameters()),
                        tensors=[dict(name=n, kind=k, shape=list(t.shape), dtype=str(t.dtype), elements=t.numel())
                                 for k, items in [('parameter', model.named_parameters()), ('buffer', model.named_buffers())]
                                 for n, t in items])
    export_text(args, json_text(data))
    text = None
    if args.action == 'list':
        text = 'COMPONENT                 CATEGORY     DESCRIPTION\n' + ''.join(f"{r['name']:<25} {r['category']:<12} {r['description']}\n" for r in rows)
    return Outcome(data, text=text)
