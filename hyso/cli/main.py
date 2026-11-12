"""Argparse entry point with stable stdout, stderr and exit-code contracts."""
import argparse
from contextlib import redirect_stdout
import sys
import traceback

from .common import CLIError, common_options, json_text


class Parser(argparse.ArgumentParser):
    def error(self, message):
        raise CLIError(message + f'. Run {self.prog} --help for valid arguments.')


def build_parser():
    parser = Parser(prog='hyso', description='HYSo models, training and retained/native analysis workflows. Commands default to read-only or dry-run.',
                    epilog='Exit codes: 0 success/dry-run, 2 invalid input, 3 blocked/missing permission or source, 4 unsupported, 5 workload failure, 6 partial result, 1 unexpected error, 130 interrupted.')
    common_options(parser, export=False)
    parser.add_argument('--version', action='version', version='hyso 0.2.0')
    sub = parser.add_subparsers(dest='group', required=True)
    from . import models, train, analysis, experiments, presentation
    for module in (models, train, analysis, experiments, presentation):
        module.register(sub)
    return parser


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    machine = '--json' in argv
    debug = '--debug' in argv or '--verbose' in argv
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
            sys.stdout.write(outcome.text)
        else:
            sys.stdout.write(f'{outcome.status}\n' + json_text(outcome.data))
        return outcome.code
    except SystemExit as exc:
        if debug:
            traceback.print_exc(file=sys.stderr)
        error, code = f'Command exited before returning a result (code {exc.code!r}). Inspect native state before retrying.', 5
    except KeyboardInterrupt:
        error, code = 'Interrupted; workload state may have advanced. Inspect retained artifacts before retrying.', 130
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
        if not isinstance(exc, CLIError):
            error += '. Check the input format, paths and command --help; use --debug for details.'
    if machine:
        status = {3: 'blocked', 4: 'unsupported', 5: 'failed', 130: 'interrupted'}.get(code, 'error')
        sys.stdout.write(json_text(dict(format='hyso.cli', schema_version=1, status=status, exit_code=code, error=error), ascii_only=True))
    sys.stderr.write('hyso: ' + error + '\n')
    return code
