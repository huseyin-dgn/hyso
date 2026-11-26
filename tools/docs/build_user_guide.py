"""Build the single offline guide from reviewed four-language editorial data.

Stdlib only; never imports HYSo, Torch, model factories or user code.
"""
import argparse
import base64
import configparser
import hashlib
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTENT = ROOT / 'docs/user-guide/content.json'
DESTINATION = ROOT / 'hyso/data'
LANGUAGES = ('tr', 'en', 'de', 'es')

STYLE = """
:root{color-scheme:light;font-family:system-ui,-apple-system,Segoe UI,sans-serif;color:#20252b;background:#fff;line-height:1.7}
*{box-sizing:border-box}[hidden]{display:none!important}body{margin:0}a{color:#23577a;text-underline-offset:3px}a:hover{text-decoration-thickness:2px}
header{border-bottom:1px solid #dce0e4;padding:1.2rem 2rem}header strong{font-size:1.35rem}header p{margin:.25rem 0;color:#505963}
.controls{display:flex;align-items:end;flex-wrap:wrap;gap:1rem;padding:.6rem 2rem;background:#fff;border-bottom:1px solid #dce0e4;position:sticky;top:0;z-index:10}.controls label{display:grid;gap:.2rem;font-size:.9rem}input,select{font:inherit;border:1px solid #8a949e;border-radius:3px;padding:.4rem .6rem}input{width:min(28rem,80vw)}
.layout{display:grid;grid-template-columns:18rem minmax(0,1fr);max-width:1380px;margin:auto}aside{padding:1.5rem;border-right:1px solid #e1e4e8;position:sticky;top:6rem;height:calc(100vh - 6rem);overflow:auto}
nav a{display:block;padding:.3rem 0;font-size:.92rem}nav .group{font-weight:650;margin:1rem 0 .4rem}main{padding:1.5rem 3rem;max-width:970px;min-width:0}section{padding:1.3rem 0 2rem;border-bottom:1px solid #dce0e4;scroll-margin-top:7rem}
h1{font-size:1.8rem;line-height:1.3;margin:.5rem 0 1rem}h2{font-size:1.5rem;line-height:1.35;margin:.4rem 0 1rem}h3{font-size:1.05rem;margin:1.2rem 0 .3rem}p{margin:.5rem 0 1rem}pre{overflow:auto;background:#f4f5f6;border:1px solid #e0e3e6;padding:1rem;font-size:.87rem;line-height:1.6}code{font-family:Consolas,monospace;overflow-wrap:anywhere}
.topic-id,.meta{font-size:.8rem;color:#5c6570}.limits{border-left:3px solid #7f8993;padding-left:1rem}.related{margin-top:1rem}.skip{position:absolute;left:-9999px}.skip:focus{left:1rem;top:.2rem;background:white;padding:.3rem;z-index:5}:focus-visible{outline:3px solid #23577a;outline-offset:3px}footer{padding:1.5rem 2rem;border-top:1px solid #dce0e4;color:#505963}
@media(max-width:850px){.layout{display:block}aside{position:static;height:auto;border-right:0;border-bottom:1px solid #ddd;padding:1rem}nav{max-height:14rem;overflow:auto}main{padding:1.2rem}header{padding:1rem}h2{font-size:1.3rem}.controls{display:grid;grid-template-columns:minmax(0,1fr) auto;padding:.5rem 1rem;gap:.4rem .75rem}.controls label:nth-child(2){grid-column:1/-1;grid-row:2}.controls input{width:100%;min-width:0}section{scroll-margin-top:12rem}}
@media print{aside,.controls,.skip{display:none}.layout{display:block}main{max-width:none;padding:0}section{break-inside:auto}pre{white-space:pre-wrap}a{color:inherit}}
""".strip()

SCRIPT = r"""
(() => {
  'use strict';
  const langs = ['tr','en','de','es'];
  const selector = document.getElementById('language');
  const search = document.getElementById('search');
  const sections = [...document.querySelectorAll('section[data-topic]')];
  const entries = [...document.querySelectorAll('nav a[data-topic-link]')];
  const ids = new Set(sections.map(s => s.id));
  let language = langs.includes(new URL(location.href).searchParams.get('lang')) ? new URL(location.href).searchParams.get('lang') : 'tr';
  const fold = s => s.toLocaleLowerCase(language).normalize('NFKD').replace(/\p{M}/gu,'').replace(/ı/g,'i').replace(/ß/g,'ss');
  function topic() { try { const id = decodeURIComponent(location.hash.slice(1)); return ids.has(id) ? id : null; } catch { return null; } }
  function filter() {
    const words = fold(search.value.trim()).split(/\s+/).filter(Boolean);
    let shown = 0;
    for (const section of sections) {
      const panel = section.querySelector('[data-lang="'+language+'"]');
      const text = fold(section.id + ' ' + panel.textContent + ' ' + (section.querySelector('pre')?.textContent || '') + ' ' + (section.dataset.inventory || ''));
      section.hidden = !words.every(word => text.includes(word));
      if (!section.hidden) shown++;
    }
    for (const entry of entries) entry.hidden = document.getElementById(entry.dataset.topicLink).hidden;
    document.getElementById('results').textContent = shown + ' / ' + sections.length;
    document.getElementById('empty').hidden = shown !== 0;
  }
  function apply(next, anchor) {
    language = next; selector.value = language; document.documentElement.lang = language;
    document.querySelectorAll('[data-lang]').forEach(node => { node.hidden = node.dataset.lang !== language; });
    document.querySelectorAll('[data-ui-label]').forEach(node => { node.setAttribute('aria-label', document.getElementById(node.dataset.uiLabel).querySelector('[data-lang="'+language+'"]').textContent); });
    filter();
    const url = new URL(location.href); url.searchParams.set('lang', language); if (anchor) url.hash = anchor;
    try { history.replaceState(null,'',url); } catch { /* File-viewer policies may forbid URL state updates. */ }
    if (anchor) document.getElementById(anchor).scrollIntoView();
  }
  selector.addEventListener('change', () => {
    const boundary = Math.max(document.querySelector('.controls').getBoundingClientRect().bottom + 16, parseFloat(getComputedStyle(sections[0]).scrollMarginTop) || 0) + 2;
    const visible = sections.filter(s => !s.hidden);
    const anchor = visible.find(s => s.getBoundingClientRect().top <= boundary && s.getBoundingClientRect().bottom > boundary)?.id || visible.find(s => s.getBoundingClientRect().bottom > boundary)?.id || topic();
    search.value = ''; apply(selector.value, anchor);
  });
  search.addEventListener('input', filter);
  window.addEventListener('hashchange', () => { const anchor = topic(); if (anchor) { search.value=''; filter(); document.getElementById(anchor).scrollIntoView(); } });
  apply(language, topic());
})();
""".strip()


def digest(text):
    return base64.b64encode(hashlib.sha256(text.encode('utf-8')).digest()).decode('ascii')


def translated(values, tag='span'):
    return ''.join(f'<{tag} lang="{lang}" data-lang="{lang}"'+(' hidden' if lang != 'tr' else '')+'>'+escape(values[lang])+f'</{tag}>' for lang in LANGUAGES)


def render(data, version):
    topics = data['topics']; ids = [t['id'] for t in topics]
    if len(ids) != len(set(ids)):
        raise ValueError('duplicate topic ID')
    by_id = {t['id']: t for t in topics}
    for topic in topics:
        if set(topic['translations']) != set(LANGUAGES) or any(r not in by_id for r in topic['related']):
            raise ValueError('missing language or related topic')
        fields = set(topic['translations']['tr'])
        if any(set(t) != fields or any(not isinstance(v,str) or not v.strip() for v in t.values()) for t in topic['translations'].values()):
            raise ValueError('incomplete parallel topic fields')
    ui = data['ui']; e = escape
    csp = f"default-src 'none'; script-src 'sha256-{digest(SCRIPT)}'; style-src 'sha256-{digest(STYLE)}'; connect-src 'none'; base-uri 'none'; form-action 'none'; object-src 'none'"
    chunks = ['<!doctype html><html lang="tr"><head><meta charset="utf-8">',
              '<meta name="viewport" content="width=device-width,initial-scale=1">',
              f'<meta http-equiv="Content-Security-Policy" content="{e(csp,quote=True)}">',
              f'<meta name="hyso-version" content="{e(version)}"><title>HYSo {e(version)} — User guide</title>',
              '<style>'+STYLE+'</style></head><body>',
              '<a class="skip" href="#main">'+translated(ui['skip'])+'</a>',
              '<header><strong>HYSo '+e(version)+'</strong><p>'+translated(ui['subtitle'])+'</p></header>',
              '<div class="controls"><label id="language-label">'+translated(ui['language'])+
              '<select id="language" data-ui-label="language-label" aria-label="Dil">'+''.join(f'<option value="{code}">{name}</option>' for code,name in zip(LANGUAGES,('Türkçe','English','Deutsch','Español')))+'</select></label>',
              '<label id="search-label">'+translated(ui['search'])+'<input id="search" type="search" autocomplete="off" data-ui-label="search-label" aria-label="Konularda ara"></label>',
              '<span aria-live="polite"><span id="results"></span> '+translated(ui['topics'])+'</span></div>',
              '<div class="layout"><aside><details open><summary id="contents-label">'+translated(ui['contents'])+'</summary><nav data-ui-label="contents-label" aria-label="İçindekiler">']
    for topic in topics:
        chunks.append(f'<a href="#{e(topic["id"])}" data-topic-link="{e(topic["id"])}">'+translated({l:topic['translations'][l]['title'] for l in LANGUAGES})+'</a>')
    chunks.append('</nav></details></aside><main id="main"><h1>'+translated(ui['title'])+'</h1><p id="empty" hidden>'+translated(ui['empty'])+'</p>')
    for topic in topics:
        inventory = topic.get('inventory', [])
        chunks.append(f'<section id="{e(topic["id"])}" data-topic="{e(topic["id"])}" data-inventory="{e(" ".join(inventory))}"><div class="topic-id">{e(topic["id"])}</div>')
        for lang in LANGUAGES:
            text = topic['translations'][lang]
            chunks.append(f'<div lang="{lang}" data-lang="{lang}"'+(' hidden' if lang!='tr' else '')+'><h2>'+e(text['title'])+'</h2>')
            for field in ('what','measures','meaning','when','benefit','limits'):
                if field in text:
                    chunks.append(('<div class="limits">' if field=='limits' else '<div>')+'<h3>'+e(ui[field][lang])+'</h3><p>'+e(text[field]).replace('\n\n','</p><p>')+'</p></div>')
            chunks.append('</div>')
        if inventory:
            chunks.append('<p><code>'+e(' · '.join(inventory))+'</code></p>')
        if topic.get('code'):
            chunks.append('<pre><code>'+e(topic['code'])+'</code></pre>')
        chunks.append('<div class="related">'+translated(ui['related'])+' '+ ' · '.join('<a href="#'+e(r)+'">'+translated({l:by_id[r]['translations'][l]['title'] for l in LANGUAGES})+'</a>' for r in topic['related'])+'</div></section>')
    chunks.append('</main></div><footer>'+translated(ui['footer'])+'</footer><script>'+SCRIPT+'</script></body></html>\n')
    registry = dict(format='hyso.user_guide.topics', schema_version=1, hyso_version=version, languages=list(LANGUAGES),
                    topics=[dict(id=t['id'], titles={l:t['translations'][l]['title'] for l in LANGUAGES}, related=t['related'], sources=t['sources'], inventory=t.get('inventory',[])) for t in topics])
    registry['release_support']=data['release_support']
    return ''.join(chunks), json.dumps(registry,ensure_ascii=False,indent=2)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    cfg=configparser.ConfigParser();cfg.read(ROOT/'setup.cfg',encoding='utf-8')
    data=json.loads(CONTENT.read_text(encoding='utf-8'))
    html,registry=render(data,cfg['metadata']['version'])
    for name,text in [('user-guide.html',html),('user-guide-topics.json',registry)]:
        path=DESTINATION/name
        if args.check:
            if not path.is_file() or path.read_text(encoding='utf-8')!=text:raise SystemExit('stale guide: '+str(path))
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8',newline='\n')
    print('Guide current: '+str(len(data['topics']))+' topics, 4 embedded languages.')


if __name__=='__main__':main()
