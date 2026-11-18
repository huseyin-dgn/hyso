"""Locate HYSo's installed four-language offline guide using only the stdlib."""
import argparse
import json
from pathlib import Path
import sys
from urllib.parse import urlencode


def _print(text):
    try:
        print(text)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(encoding='utf-8')
        print(text)


def main(argv=None):
    parser = argparse.ArgumentParser(prog='hyso-guide', description=__doc__)
    parser.add_argument('--language', choices=('tr','en','de','es'), default='tr')
    parser.add_argument('--topic', default='about.overview', help='Stable topic ID, for example analysis.p23.')
    parser.add_argument('--list-topics', action='store_true', help='List topic IDs and titles in the chosen language.')
    parser.add_argument('--path', action='store_true', help='Print the installed HTML path.')
    parser.add_argument('--open', action='store_true', help='Explicitly open the local HTML in your browser; no model code runs.')
    parser.add_argument('--version', action='store_true', help='Show the bundled guide version.')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent / 'hyso/data'
    try:
        registry = json.loads((root/'user-guide-topics.json').read_text(encoding='utf-8'))
        guide = root/'user-guide.html'
        if not guide.is_file():
            raise FileNotFoundError(str(guide))
        if args.version:
            _print('HYSo guide '+registry['hyso_version'])
            return 0
        if args.list_topics:
            for topic in registry['topics']:
                _print(topic['id']+'\t'+topic['titles'][args.language])
            return 0
        if args.topic not in {t['id'] for t in registry['topics']}:
            parser.error('Unknown topic ID. Use --list-topics to find a valid ID.')
        uri = guide.as_uri()+'?'+urlencode({'lang':args.language})+'#'+args.topic
        _print(str(guide) if args.path else uri)
        if args.open:
            import webbrowser
            if not webbrowser.open(uri):
                print('hyso-guide: Open the printed local file in a browser.',file=sys.stderr)
                return 2
        return 0
    except (OSError,ValueError,KeyError) as error:
        print('hyso-guide: Guide resource unavailable: '+str(error)+'. Reinstall the matching HYSo wheel.',file=sys.stderr)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
