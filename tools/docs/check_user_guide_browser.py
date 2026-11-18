"""Offline headless guide checks; requires existing Playwright and Chromium.

Developer validation only. Never downloads a browser or imports HYSo/Torch.
"""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--browser',type=Path,required=True,help='Existing Chrome/Edge/Chromium executable.')
    parser.add_argument('--output',type=Path,required=True,help='New directory for screenshots and the receipt; never reused.')
    parser.add_argument('--guide',type=Path,default=Path(__file__).resolve().parents[2]/'hyso/data/user-guide.html')
    args=parser.parse_args()
    if not args.browser.is_file() or not args.guide.is_file():parser.error('Browser and guide files must already exist.')
    from playwright.sync_api import sync_playwright
    args.output.mkdir(parents=True,exist_ok=False)
    checks=[];errors=[];requests=[]
    with sync_playwright() as p:
        browser=p.chromium.launch(executable_path=str(args.browser),headless=True,
            args=['--disable-gpu','--disable-background-networking','--disable-extensions','--no-first-run'])
        context=browser.new_context(offline=True,viewport={'width':1440,'height':1000})
        page=context.new_page()
        page.on('pageerror',lambda error:errors.append(str(error)))
        page.on('request',lambda request:requests.append(request.url))
        try:
            page.goto(args.guide.resolve().as_uri()+'?lang=tr#analysis.p23')
            for lang,label in [('tr','İçindekiler'),('en','Contents'),('de','Inhaltsverzeichnis'),('es','Índice')]:
                page.locator('#language').select_option(lang)
                assert page.locator('html').get_attribute('lang')==lang
                assert page.evaluate('decodeURIComponent(location.hash)')=='#analysis.p23'
                assert page.locator('section[id="analysis.p23"] > div[data-lang="'+lang+'"]').is_visible()
                assert page.locator('nav').get_attribute('aria-label')==label
                assert page.evaluate('document.getElementById("analysis.p23").getBoundingClientRect().top >= document.querySelector(".controls").getBoundingClientRect().bottom')
                checks.append('language_anchor_'+lang)
            checks.append('localized_navigation_labels')
            page.screenshot(path=str(args.output/'desktop-es.png'))
            for lang,query,topic in [('tr','egitim','training.trainer'),('de','gross','models.model-builder'),('es','clasificacion','analysis.p26'),('en','energy','analysis.p33')]:
                page.locator('#language').select_option(lang);page.locator('#search').fill(query)
                assert page.locator('section[id="'+topic+'"]').is_visible(),(lang,query)
                assert not page.locator('#empty').is_visible()
                checks.append('search_'+lang)
            page.locator('#search').fill('no-result-ff88292');assert page.locator('#empty').is_visible()
            page.locator('#search').fill('');checks.append('empty_search_state')
            page.evaluate('location.hash="training.trainer"')
            page.wait_for_function('document.getElementById("training.trainer").getBoundingClientRect().top < 150')
            page.evaluate('document.getElementById("analysis.p30").scrollIntoView()')
            page.locator('#language').select_option('de')
            assert page.evaluate('decodeURIComponent(location.hash)')=='#analysis.p30'
            checks.append('manual_scroll_language_anchor')
            page.locator('#search').fill('camera');page.locator('#language').select_option('es')
            assert page.locator('#search').input_value()=='';checks.append('language_clears_search')
            page.set_viewport_size({'width':390,'height':844})
            page.goto(args.guide.resolve().as_uri()+'?lang=de#models.model-builder')
            assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
            page.screenshot(path=str(args.output/'mobile-de.png'));checks.append('mobile_no_horizontal_overflow')
            assert not errors,errors
            assert all(not url.startswith(('http:','https:','ws:','wss:')) for url in requests),requests
            checks += ['no_javascript_errors','no_network_dependencies']
        finally:
            context.close();browser.close()
    receipt=dict(passed=len(checks),checks=checks,errors=errors,requests=requests,offline=True)
    (args.output/'browser-validation.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=len(checks),checks=checks)))


if __name__=='__main__':main()
