"""CLI boundaries: bounded JSON, explicit trusted imports and atomic exports."""
from dataclasses import dataclass
import importlib
import json
from pathlib import Path
import re


class CLIError(ValueError):
    def __init__(self, message, code=2):
        super().__init__(message)
        self.code = code


@dataclass
class Outcome:
    data: object
    code: int = 0
    status: str = 'ok'
    text: str | None = None


def common_options(parser, *, export=True):
    # SUPPRESS allows these options before or after either command level.
    import argparse
    for flag in ('json', 'debug', 'verbose'):
        parser.add_argument('--' + flag, action='store_true', default=argparse.SUPPRESS,
                            help={'json': 'Emit one JSON envelope on stdout; diagnostics go to stderr.',
                                  'debug': 'Show a technical traceback on stderr on failure.',
                                  'verbose': 'Show technical failure details on stderr.'}[flag])
    if export:
        parser.add_argument('--output', type=Path, help='Export to this file; parent directory must already exist.')
        parser.add_argument('--overwrite', action='store_true', help='Explicitly replace an existing regular output file.')


def command(subparsers, name, help, handler, *, export=True, **kwargs):
    p = subparsers.add_parser(name, help=help, description=help, **kwargs)
    common_options(p, export=export)
    p.set_defaults(handler=handler)
    return p


def read_json(path, *, limit=2 * 1024**2, sha256=None):
    from hyso.analysis._codec import read_json as reader
    if not Path(path).is_file():
        raise CLIError(f'Input file does not exist or is not a regular file: {path}. Supply an existing JSON file.')
    return reader(path, expected_sha256=sha256, max_bytes=limit)[0]


def output_check(args, *, inputs=()):
    path = getattr(args, 'output', None)
    if path is None:
        return
    path = path.absolute()
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise CLIError(f'Output must be a regular file, not a directory or symlink: {path}.')
    if any(path.resolve() == Path(p).resolve() for p in inputs if p is not None):
        raise CLIError('Output cannot replace an input source. Choose a different --output file.')
    if not path.parent.is_dir():
        raise CLIError(f'Output directory does not exist: {path.parent}. Create it before running this command.')
    if path.exists() and not args.overwrite:
        raise CLIError(f'Output exists: {path}. Choose another filename or explicitly pass --overwrite.')


def export_text(args, raw):
    if args.output is not None:
        from hyso.analysis._codec import export
        export(args.output, raw, args.overwrite)


def json_text(value, *, ascii_only=False):
    return json.dumps(value, ensure_ascii=ascii_only, indent=2, allow_nan=False) + '\n'


def require(condition, message, *, code=2):
    if not condition:
        raise CLIError(message, code)


def factory_spec(value):
    require(isinstance(value, str) and re.fullmatch(r'[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*', value),
            'Use an importable trusted factory as module:callable (no file paths, expressions or nested attributes).')
    return value.split(':')


def load_factory(value):
    module, name = factory_spec(value)
    factory = getattr(importlib.import_module(module), name)
    require(callable(factory), 'The supplied module:callable is not callable. Export a factory function.')
    return factory


def factory_options(parser):
    parser.add_argument('--factory', help='Trusted installed module:callable; never imported during dry-run.')
    parser.add_argument('--factory-args', type=Path, help='Bounded plain JSON object passed as keyword arguments to the factory.')
    parser.add_argument('--trust-factory', action='store_true', help='Authorize trusted Python import/call during execution. This is not a sandbox.')


def factory_kwargs(args):
    if args.factory:
        factory_spec(args.factory)
    require(args.factory is not None or args.factory_args is None, '--factory-args requires --factory module:callable.')
    value = {} if args.factory_args is None else read_json(args.factory_args)
    require(type(value) is dict, '--factory-args must contain a JSON object.')
    return value


def presentation_options(parser, *, formats=('json', 'markdown', 'html'), levels=True):
    parser.add_argument('--format', choices=formats, default='json', help='Export/render format (default: json).')
    parser.add_argument('--language', choices=('tr', 'en'), default='tr', help='Presentation language (default: tr).')
    if levels:
        parser.add_argument('--level', choices=('beginner', 'technical', 'expert'), default='technical', help='Explanation detail.')


def analysis_context(path):
    from hyso.analysis import AnalysisContext
    if path is None:
        return None
    data = read_json(path)
    require(type(data) is dict, '--context must contain an AnalysisContext keyword object.')
    return AnalysisContext(**data)


def read_result(path):
    """Dispatch a closed set of existing validated readers; never pickle or code."""
    from hyso.analysis import AnalysisExecutionResult, UnifiedAnalysisReport, EvidenceIndex, ModelComparisonResult
    from hyso.advisor import RecommendationReport
    data = read_json(path, limit=128 * 1024**2)
    require(type(data) is dict, 'A saved HYSo result must be a JSON object.')
    fmt = data.get('format')
    if fmt == 'hyso.analysis_execution':
        return AnalysisExecutionResult.from_dict(data)
    if fmt == 'hyso.analysis_report':
        return UnifiedAnalysisReport(data)
    if fmt == 'hyso.evidence_index':
        require(set(data) == {'format', 'schema_version', 'records'} and type(data.get('schema_version')) is int
                and data['schema_version'] == 1, 'Unsupported EvidenceIndex schema. Use schema_version 1.')
        from hyso.analysis import EvidenceRecord
        return EvidenceIndex(EvidenceRecord(row) for row in data['records'])
    if fmt == 'hyso.two_model_comparison':
        return ModelComparisonResult(data)
    if fmt == 'hyso.advisor.AdvisorReport':
        from hyso.advisor import AdvisorReport
        return AdvisorReport.from_dict(data)
    if fmt == 'hyso.advisor.recommendations':
        return RecommendationReport(data)
    if fmt == 'hyso.training_session_result':
        # TrainingResult has no public JSON reader. Reuse Manager's validated
        # restore path, never construct a result from unchecked _payload text.
        from hyso.experiments.workflow import _restore_result
        return _restore_result('training_result', data)
    raise CLIError(f'Unsupported saved result format: {fmt!r}. Supply execution, analysis, evidence, comparison, training or recommendation JSON.')


def existing_manager(root):
    from hyso.experiments import ExperimentManager
    require((Path(root) / 'experiments.sqlite3').is_file(),
            f'Existing ExperimentManager store not found: {root}. Supply its dedicated local directory; CLI does not create or migrate stores.')
    return ExperimentManager(root)
