"""Argparse entry point with stable stdout, stderr and exit-code contracts."""
import argparse
from contextlib import redirect_stdout
import sys
import traceback

from .common import CLIError, common_options, json_text


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CLIError(message, suggestion=f'Run {self.prog} --help for valid arguments and examples.')


def _write_text(stream, text):
    """Keep Unicode reports readable through legacy Windows redirection."""
    try:
        stream.write(text)
    except UnicodeEncodeError:
        # TextIOWrapper encodes before writing, so retrying does not duplicate
        # output. Leave compatible encodings and in-memory streams unchanged.
        stream.reconfigure(encoding='utf-8')
        stream.write(text)


def build_parser():
    parser = Parser(prog='hyso', description='HYSo CPU-first models, training and analysis. Commands default to read-only or dry-run; CUDA execution requires separate authorization and profile validation.',
                    formatter_class=argparse.RawDescriptionHelpFormatter,
                    epilog='Start here: hyso models list | hyso train options | hyso analysis list\n'
                           'Read saved results: hyso compare --help | hyso advisor --help | hyso reports --help\n'
                           'Browse saved experiments: hyso experiments --help\n'
                           'Offline four-language guide: hyso-guide --open (tr/en/de/es).\n'
                           'Exit codes: 0 success/dry-run, 2 invalid input, 3 blocked/missing permission or source, 4 unsupported, 5 workload failure, 6 partial result, 1 unexpected error, 130 interrupted.')
    common_options(parser, export=False)
    parser.add_argument('--version', action='version', version='hyso 0.2.0')
    sub = parser.add_subparsers(dest='group', required=True)
    from . import models, train, analysis, experiments, presentation, docs
    for module in (models, train, analysis, experiments, presentation, docs):
        module.register(sub)
    examples = {
        'models': 'hyso models list\nhyso models show conv2d\nhyso models summary model.json\nhyso models inspect model.json --dry-run',
        'train': 'hyso train options\nhyso train validate training.json\nhyso train run training.json --dry-run\nExecution requires a trusted session factory, --execute, --trust-factory and --allow-training.',
        'analysis': 'hyso analysis list\nhyso analysis show P01\nhyso analysis plan --features P01 P02\nhyso analysis run --features P01 --dry-run\nA plan does not supply missing evidence or authorize execution.',
        'compare': 'hyso compare baseline.json candidate.json --baseline-model A --candidate-model B --features P01 P02\nUses saved analysis/execution files only. Model labels must match their evidence.',
        'advisor': 'hyso advisor recommend analysis.json --format markdown --language tr\nReads saved evidence and gives manual advice; does not train or edit model code.',
        'reports': 'hyso reports render result.json --format html --output report.html\nReads a supported saved HYSo result. Output parent must exist; files are not overwritten by default.',
        'experiments': 'hyso experiments list --root C:/local-hyso/store\nOpens an existing ExperimentManager store; does not create or migrate it.',
        'docs': 'hyso docs list\nhyso docs show models.graph-builder --lang tr\nhyso docs open analysis.p23 --lang de\nOnly docs open opens a browser; --dry-run prints its local URI.',
    }
    for name, example in examples.items():
        sub.choices[name].formatter_class = argparse.RawDescriptionHelpFormatter
        sub.choices[name].epilog = 'Examples:\n' + example
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    machine = '--json' in argv
    debug = '--debug' in argv or '--verbose' in argv
    diagnostics = None
    suggestion = None
    try:
        try:
            args = build_parser().parse_args(argv)
        except SystemExit as help_exit:
            return int(help_exit.code or 0)
        # Third-party factories/native progress use stderr; stdout stays a single
        # machine-readable document, including in failure and partial-result cases.
        with redirect_stdout(sys.stderr):
            outcome = args.handler(args)
        envelope = dict(format='hyso.cli', schema_version=1, status=outcome.status,
                        exit_code=outcome.code, result=outcome.data)
        if machine:
            # Escapes preserve arbitrary Unicode even through legacy Windows
            # redirected stream encodings; exported files remain UTF-8.
            sys.stdout.write(json_text(envelope, ascii_only=True))
        elif outcome.text is not None:
            _write_text(sys.stdout, outcome.text)
        else:
            _write_text(sys.stdout, f'{outcome.status}\n' + json_text(outcome.data))
        return outcome.code
    except SystemExit as exc:
        if debug:
            traceback.print_exc(file=sys.stderr)
        error, code = f'Command exited before returning a result (code {exc.code!r}).', 5
        suggestion = 'Inspect native state and retained artifacts before retrying.'
    except KeyboardInterrupt:
        if debug:
            traceback.print_exc(file=sys.stderr)
        error, code = 'Interrupted; workload state may have advanced.', 130
        suggestion = 'Inspect retained artifacts before retrying.'
    except Exception as exc:
        if debug:
            traceback.print_exc(file=sys.stderr)
        native_code = getattr(exc, 'code', None)
        if isinstance(exc, CLIError):
            code = exc.code
        elif isinstance(native_code, str) and native_code.startswith('unsupported'):
            code = 4
        elif isinstance(exc, (ValueError, TypeError, KeyError, AttributeError, OSError, ImportError)):
            code = 2
        else:
            code = 1
        if not isinstance(code, int):
            code = 2
        error = str(exc) or type(exc).__name__
        suggestion = getattr(exc, 'suggestion', None)
        if suggestion is None:
            if isinstance(exc, FileNotFoundError):
                suggestion = 'Check the input path and filename. Supply an existing file or store; no workload was retried.'
            elif isinstance(exc, PermissionError):
                suggestion = 'Check read/write permissions for the indicated path, or choose an accessible location.'
            elif code == 2:
                suggestion = 'Review the reported input and command --help; use --debug for technical details.'
            elif code in (3, 4):
                suggestion = 'Review the reported requirements and supported profiles in command --help.'
            elif code == 5:
                suggestion = 'Inspect retained results and native state before deciding whether to retry; use --debug for details.'
            else:
                suggestion = 'Use --debug for the original traceback; no automatic retry was performed.'
        from hyso.models.registry import ModelConfigError
        if isinstance(exc, ModelConfigError):
            diagnostics = [exc.to_dict()]
    if machine:
        status = {3: 'blocked', 4: 'unsupported', 5: 'failed', 130: 'interrupted'}.get(code, 'error')
        envelope = dict(format='hyso.cli', schema_version=1, status=status, exit_code=code, error=error)
        if diagnostics is not None:
            envelope['diagnostics'] = diagnostics
        sys.stdout.write(json_text(envelope, ascii_only=True))
    _write_text(sys.stderr, 'hyso: ' + error + '\n')
    if diagnostics is not None:
        _write_text(sys.stderr, 'Reason: ' + diagnostics[0]['message_tr'] + '\n')
    if suggestion is not None:
        _write_text(sys.stderr, 'Suggestion: ' + suggestion + '\n')
    return code
