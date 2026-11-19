"""Locate HYSo's installed four-language offline guide using only the stdlib."""
import typing as _typing

_TopicId = _typing.Literal['about.overview', 'getting-started.installation', 'getting-started.first-run', 'models.support', 'models.legacy', 'models.development-families', 'models.model-builder', 'models.graph-builder', 'models.checks', 'components.convolution', 'components.normalization', 'components.activation', 'components.pooling', 'components.residual', 'components.attention', 'components.regularization', 'components.sampling', 'components.heads', 'components.fusion', 'components.mobile', 'training.configuration', 'training.trainer', 'training.checkpoints', 'analysis.overview', 'metrics.interpretation', 'analyzer.workflow', 'comparison.models', 'comparison.architecture', 'advisor.recommendations', 'reporting.outputs', 'experiments.manager', 'api.python-cli', 'troubleshooting.common', 'support.cpu-gpu', 'compatibility.versions', 'analysis.p01', 'analysis.p02', 'analysis.p03', 'analysis.p04', 'analysis.p05', 'analysis.p06', 'analysis.p07', 'analysis.p08', 'analysis.p09', 'analysis.p10', 'analysis.p11', 'analysis.p12', 'analysis.p13', 'analysis.p14', 'analysis.p15', 'analysis.p16', 'analysis.p17', 'analysis.p18', 'analysis.p19', 'analysis.p20', 'analysis.p21', 'analysis.p22', 'analysis.p23', 'analysis.p24', 'analysis.p25', 'analysis.p26', 'analysis.p27', 'analysis.p28', 'analysis.p29', 'analysis.p30', 'analysis.p31', 'analysis.p32', 'analysis.p33', 'analysis.p34', 'analysis.p35']
import argparse
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
from urllib.parse import urlencode


def _resources() -> tuple[Path, dict]:
    root = Path(__file__).resolve().parent / 'hyso/data'
    registry = json.loads((root/'user-guide-topics.json').read_text(encoding='utf-8'))
    guide = root/'user-guide.html'
    if not guide.is_file():
        raise FileNotFoundError(str(guide))
    return guide, registry


def _topic(topic_id: _TopicId, lang: str) -> tuple[Path, dict, dict]:
    if lang not in ('tr','en','de','es'):
        raise ValueError('Unsupported guide language. Choose tr, en, de or es.')
    guide,registry = _resources()
    for topic in registry['topics']:
        if topic['id'] == topic_id:
            return guide,registry,topic
    raise ValueError('Unknown topic ID: '+str(topic_id)+'. Use hyso docs list.')


def list_topics(lang: _typing.Literal['tr', 'en', 'de', 'es'] = 'tr') -> list[dict[str,str]]:
    """List installed guide topic IDs and localized titles without opening a browser.

    Args:
        lang: Guide language: tr, en, de or es. Default: tr.
    Returns:
        Detached dictionaries with id and title, in guide order.
    Raises:
        ValueError: The language or bundled registry is invalid.
        OSError: Installed guide resources are unavailable.
    Example:
        topics = list_topics(lang='de')
    HYSo Guide: api.python-cli
    Offline: hyso docs show api.python-cli --lang en
    """
    _,registry,_ = _topic('about.overview',lang)
    return [dict(id=t['id'],title=t['titles'][lang]) for t in registry['topics']]


def guide_uri(topic_id: _TopicId, lang: _typing.Literal['tr', 'en', 'de', 'es'] = 'tr') -> str:
    """Resolve an installed topic to a local file URI; never open it implicitly.

    Args:
        topic_id: Exact stable ID, such as models.graph-builder or analysis.p23.
        lang: tr, en, de or es; default tr.
    Returns:
        A file URI with the language query and the exact topic fragment.
    Raises:
        ValueError: Topic/language is unknown; no fallback topic is fabricated.
        OSError: Packaged resources cannot be read.
    Example:
        uri = guide_uri('analysis.p23', lang='de')
    HYSo Guide: api.python-cli
    Offline: hyso docs show api.python-cli --lang en
    """
    guide,_,_ = _topic(topic_id,lang)
    return guide.as_uri()+'?'+urlencode({'lang':lang})+'#'+topic_id


class _TopicText(HTMLParser):
    """Read escaped guide prose as text; never execute HTML or example code."""
    def __init__(self, topic_id, lang):
        super().__init__(convert_charrefs=True)
        self.topic_id,self.lang=topic_id,lang
        self.active=False;self.depth=0;self.capture=None;self.parts=[]

    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs)
        if tag=='section' and attrs.get('id')==self.topic_id:
            self.active=True;self.depth=0
        if not self.active:return
        self.depth+=1
        if (tag=='div' and attrs.get('data-lang')==self.lang) or tag=='pre':
            self.capture=self.depth
            self.parts.append('\n')
        if self.capture is not None and tag in ('h2','h3','p','pre'):
            self.parts.append('\n')

    def handle_endtag(self,tag):
        if not self.active:return
        if self.capture is not None and tag in ('h2','h3','p','pre'):
            self.parts.append('\n')
        if self.depth==self.capture:self.capture=None
        self.depth-=1
        if tag=='section':self.active=False

    def handle_data(self,data):
        if self.active and self.capture is not None:self.parts.append(data)


def show_topic(topic_id: _TopicId, lang: _typing.Literal['tr', 'en', 'de', 'es'] = 'tr') -> dict:
    """Read one localized topic from the same installed HTML, without executing it.

    Args:
        topic_id: Exact ID returned by list_topics(); no inferred aliases.
        lang: tr, en, de or es; default tr.
    Returns:
        Topic ID, title, text, related IDs, language and an installed file URI.
        Shared code examples are included as inert text.
    Raises:
        ValueError: Topic/language or the topic body is invalid.
        OSError: Guide resources cannot be read.
    Example:
        help_text = show_topic('models.graph-builder', lang='tr')['text']
    HYSo Guide: api.python-cli
    Offline: hyso docs show api.python-cli --lang en
    """
    guide,_,topic = _topic(topic_id,lang)
    parser=_TopicText(topic_id,lang);parser.feed(guide.read_text(encoding='utf-8'))
    text='\n'.join(line.rstrip() for line in ''.join(parser.parts).splitlines())
    while '\n\n\n' in text:text=text.replace('\n\n\n','\n\n')
    if not text.strip():raise ValueError('Bundled topic text is missing: '+topic_id)
    return dict(id=topic_id,title=topic['titles'][lang],lang=lang,text=text.strip(),
                related=list(topic['related']),uri=guide_uri(topic_id,lang))


def topic_for_api(symbol: str) -> str:
    """Return the verified guide topic for an indexed public API or registry entry.

    Args:
        symbol: Exact indexed name, for example hyso.ModelBuilder or registry:conv2d.
    Returns:
        A real topic ID, validated against the installed topic registry.
    Raises:
        ValueError: No indexed mapping exists; unsupported names have no fake link.
        OSError: Installed documentation resources cannot be read.
    Example:
        uri = guide_uri(topic_for_api('hyso.GraphModelBuilder'))
    HYSo Guide: api.python-cli
    Offline: hyso docs show api.python-cli --lang en
    """
    guide,_ = _resources()
    index=json.loads((guide.parent/'api-topics.json').read_text(encoding='utf-8'))
    if symbol not in index['entries']:
        raise ValueError('No guide mapping for API: '+str(symbol))
    topic_id=index['entries'][symbol]['topic']
    _topic(topic_id,'tr')
    return topic_id


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
