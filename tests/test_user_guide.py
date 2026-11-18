"""Static guide, safe reader and clean-wheel checks; no model workloads."""
import ast
import base64
import copy
import hashlib
from html.parser import HTMLParser
import io
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
import tarfile
import venv
import zipfile

import pytest
from test_packaging import distributions

ROOT=Path(__file__).resolve().parents[1]
CONTENT=json.loads((ROOT/'docs/user-guide/content.json').read_text(encoding='utf-8'))
REGISTRY=json.loads((ROOT/'hyso/data/user-guide-topics.json').read_text(encoding='utf-8'))
HTML=(ROOT/'hyso/data/user-guide.html').read_text(encoding='utf-8')
LANGUAGES=('tr','en','de','es')


class Document(HTMLParser):
    def __init__(self,text):
        super().__init__(convert_charrefs=True)
        self.ids=[];self.links=[];self.panels=[];self.tags=[];self.attrs=[];self.scripts=[];self.styles=[];self.csp=None;self.active=None
        self.feed(text)
    def handle_starttag(self,tag,attrs):
        attrs=dict(attrs);self.tags.append(tag);self.attrs.append((tag,attrs))
        if 'id' in attrs:self.ids.append(attrs['id'])
        if 'href' in attrs:self.links.append(attrs['href'])
        if 'data-lang' in attrs:self.panels.append(attrs['data-lang'])
        if tag=='meta' and attrs.get('http-equiv')=='Content-Security-Policy':self.csp=attrs['content']
        if tag in ('script','style'):self.active=tag;getattr(self,tag+'s').append('')
    def handle_endtag(self,tag):
        if tag==self.active:self.active=None
    def handle_data(self,data):
        if self.active:getattr(self,self.active+'s')[-1]+=data


def test_generated_guide_is_current_and_version_matches():
    import configparser
    cfg=configparser.ConfigParser();cfg.read(ROOT/'setup.cfg',encoding='utf-8')
    builder=runpy.run_path(str(ROOT/'tools/docs/build_user_guide.py'))
    html,registry=builder['render'](CONTENT,cfg['metadata']['version'])
    assert html==HTML
    assert json.loads(registry)==REGISTRY
    assert REGISTRY['hyso_version']==cfg['metadata']['version']=='0.2.0'
    assert 'content="0.2.0"' in HTML


@pytest.mark.parametrize('language',LANGUAGES)
def test_complete_physical_language_and_parallel_meaning_fields(language):
    assert CONTENT['languages']==REGISTRY['languages']==list(LANGUAGES)
    assert len(CONTENT['topics'])==70
    for topic in CONTENT['topics']:
        assert set(topic['translations'])==set(LANGUAGES)
        assert set(topic['translations'][language])==set(topic['translations']['tr'])
        for text in topic['translations'][language].values():
            assert text.strip() and '\ufffd' not in text
            assert not re.search(r'\w\?\w',text)  # Catch lossy Windows pipe encoding.
            from html import escape
            assert escape(text).split('\n\n')[0] in HTML
    for translations in CONTENT['ui'].values():assert set(translations)==set(LANGUAGES)
    assert Document(HTML).panels.count(language)>70


def test_utf8_roundtrip_and_language_characters():
    raw=(ROOT/'hyso/data/user-guide.html').read_bytes()
    assert raw.decode('utf-8').encode('utf-8')==raw
    for sample in ['Türkçe','öğren','Größe','für','Español','clasificación','¿Qué']:
        assert sample in HTML
    assert '\ufffd' not in HTML


def test_unique_stable_topics_complete_features_and_no_broken_anchors():
    doc=Document(HTML);ids={t['id'] for t in CONTENT['topics']}
    assert len(doc.ids)==len(set(doc.ids))
    assert ids.issubset(doc.ids)
    assert all(re.fullmatch(r'[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*',id) for id in ids)
    assert [t['id'] for t in CONTENT['topics'] if t['id'].startswith('analysis.p')]==['analysis.p%02d'%i for i in range(1,36)]
    for href in doc.links:assert href.startswith('#') and href[1:] in doc.ids
    for topic in CONTENT['topics']:
        assert set(topic['related']).issubset(ids)
        assert topic['sources'] and all((ROOT/p).is_file() for p in topic['sources'])
    assert {t['id'] for t in REGISTRY['topics']}==ids


def test_all_source_registry_components_and_legacy_entries_have_topics(monkeypatch):
    import torch
    def forbidden(*args,**kwargs):raise AssertionError('guide coverage started a workload')
    monkeypatch.setattr(torch.nn.Module,'__init__',forbidden)
    for name in ('init','_lazy_init','is_available','device_count'):monkeypatch.setattr(torch.cuda,name,forbidden)
    from hyso.models.registry import ModelRegistry
    names={row.name for row in ModelRegistry().list()}
    covered={name for topic in CONTENT['topics'] for name in topic['inventory']}
    assert names.issubset(covered),sorted(names-covered)
    assert {'EBO','ELO','CEMO','EMA','CNNSENormalWithFit'}.issubset(covered)
    assert {'models.checks','comparison.architecture','analyzer.workflow','training.checkpoints'}.issubset({t['id'] for t in CONTENT['topics']})


def test_analysis_topics_have_all_required_fields_and_critical_constraints():
    for topic in CONTENT['topics']:
        if topic['id'].startswith('analysis.p'):
            for text in topic['translations'].values():
                assert set(text)=={'title','what','measures','meaning','when','limits'}
    by_id={t['id']:t for t in CONTENT['topics']}
    for language in LANGUAGES:
        assert '7200' in by_id['analysis.p31']['translations'][language]['limits']
        assert 'TDP' in by_id['analysis.p33']['translations'][language]['limits']
        assert 'B4' in by_id['analysis.p23']['translations'][language]['limits']
        assert 'tr/en' in ' '.join(by_id['reporting.outputs']['translations'][language].values())
        assert 'de/es' in by_id['reporting.outputs']['translations'][language]['limits']
        assert "device='cpu'" in by_id['models.legacy']['translations'][language]['limits']


def test_html_has_no_external_dependencies_or_executable_examples():
    doc=Document(HTML)
    assert len(doc.scripts)==len(doc.styles)==1
    assert not {'iframe','object','embed','form','base','img'}.intersection(doc.tags)
    for tag,attrs in doc.attrs:
        assert not any(k.lower().startswith('on') for k in attrs)
        assert not any(k in attrs for k in ['src','srcdoc'])
    for tag,text in [('script',doc.scripts[0]),('style',doc.styles[0])]:
        value=base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()
        assert tag+"-src 'sha256-"+value+"'" in doc.csp
    assert "default-src 'none'" in doc.csp and "connect-src 'none'" in doc.csp
    assert "'unsafe-inline'" not in doc.csp and "'unsafe-eval'" not in doc.csp
    for forbidden in ['innerHTML','eval(','Function(','fetch(','XMLHttpRequest','WebSocket','document.write','localStorage']:
        assert forbidden not in doc.scripts[0]
    assert 'session.fit()' in HTML and '<pre><code>' in HTML


def test_renderer_escapes_html_and_code_payloads():
    builder=runpy.run_path(str(ROOT/'tools/docs/build_user_guide.py'))
    data=copy.deepcopy(CONTENT)
    data['topics'][0]['translations']['tr']['what']='<img src=x onerror="alert(1)">'
    data['topics'][0]['code']='</code></pre><script>alert(2)</script>'
    html,_=builder['render'](data,'0.2.0');doc=Document(html)
    assert 'img' not in doc.tags and len(doc.scripts)==1
    assert '&lt;script&gt;alert(2)&lt;/script&gt;' in html


def test_lightweight_entry_uses_stdlib_and_opens_only_on_request(capsys,monkeypatch):
    import hyso_guide
    import webbrowser
    opened=[];monkeypatch.setattr(webbrowser,'open',lambda uri:opened.append(uri) or True)
    assert hyso_guide.main(['--language','de','--topic','analysis.p23'])==0
    assert '?lang=de#analysis.p23' in capsys.readouterr().out and not opened
    assert hyso_guide.main(['--language','es','--topic','analysis.p31','--open'])==0
    assert len(opened)==1 and '?lang=es#analysis.p31' in opened[0]
    assert hyso_guide.main(['--version'])==0
    assert '0.2.0' in capsys.readouterr().out


def test_unknown_topic_rejected_before_browser_open(monkeypatch):
    import hyso_guide,webbrowser
    monkeypatch.setattr(webbrowser,'open',lambda *args:pytest.fail('invalid topic opened'))
    with pytest.raises(SystemExit) as exc:hyso_guide.main(['--topic','not.a.topic','--open'])
    assert exc.value.code==2


def test_reader_unicode_through_legacy_redirection(monkeypatch):
    import hyso_guide
    raw=io.BytesIO();stream=io.TextIOWrapper(raw,encoding='ascii')
    monkeypatch.setattr(sys,'stdout',stream)
    assert hyso_guide.main(['--list-topics','--language','tr'])==0
    stream.flush();assert 'öğren' in raw.getvalue().decode('utf-8')


def test_archives_contain_the_same_single_four_language_html(distributions):
    _,wheel,sdist=distributions
    with zipfile.ZipFile(wheel) as z:
        assert [n for n in z.namelist() if n.endswith('.html')]==['hyso/data/user-guide.html']
        assert z.read('hyso/data/user-guide.html')==(ROOT/'hyso/data/user-guide.html').read_bytes()
        assert z.read('hyso/data/user-guide-topics.json')==(ROOT/'hyso/data/user-guide-topics.json').read_bytes()
    with tarfile.open(sdist) as z:
        assert z.extractfile('hyso-0.2.0/hyso/data/user-guide.html').read()==(ROOT/'hyso/data/user-guide.html').read_bytes()
        assert z.extractfile('hyso-0.2.0/hyso/data/user-guide-topics.json').read()==(ROOT/'hyso/data/user-guide-topics.json').read_bytes()


def test_clean_wheel_guide_without_torch_or_checkout(distributions,tmp_path):
    work,wheel,_=distributions
    target=tmp_path/'clean-guide-env';venv.EnvBuilder(with_pip=False).create(target)
    exe=target/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1',PYTHONDONTWRITEBYTECODE='1',PYTHONIOENCODING='utf-8',CUDA_VISIBLE_DEVICES='-1')
    env.pop('PYTHONPATH',None)
    def run(argv):
        result=subprocess.run(argv,cwd=work,env=env,capture_output=True,text=True,encoding='utf-8',timeout=90)
        assert result.returncode==0,result.stdout+result.stderr
        return result.stdout.strip()
    run([sys.executable,'-m','pip','--python',str(exe),'install','--no-index','--no-deps',str(wheel)])
    script='''import importlib.util, importlib.metadata, pathlib, sys
assert importlib.util.find_spec("torch") is None
import hyso_guide
assert pathlib.Path(hyso_guide.__file__).is_relative_to(pathlib.Path(sys.prefix))
assert "hyso" not in sys.modules and "torch" not in sys.modules
assert hyso_guide.main(["--version"]) == 0
assert importlib.metadata.version("hyso") == "0.2.0"
'''
    assert '0.2.0' in run([str(exe),'-I','-B','-c',script])
    path=Path(run([str(exe),'-I','-B','-m','hyso_guide','--path']))
    assert path.is_relative_to(target) and path.read_bytes()==(ROOT/'hyso/data/user-guide.html').read_bytes()
    console=target/('Scripts/hyso-guide.exe' if os.name=='nt' else 'bin/hyso-guide')
    assert '0.2.0' in run([str(console),'--version'])


def test_reader_python310_syntax_and_no_hyso_import():
    source=(ROOT/'hyso_guide.py').read_text(encoding='utf-8')
    ast.parse(source,feature_version=(3,10))
    assert 'import torch' not in source and 'from hyso' not in source
