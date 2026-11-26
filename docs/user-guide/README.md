# Central user guide

The deliverable is one self-contained file:
[`hyso/data/user-guide.html`](../../hyso/data/user-guide.html).
All 70 topics and all four languages are physically present in that file.
It uses no remote assets, fonts, translation service, HTTP server or fetch call.
Code examples are escaped text, with no execution controls.

After wheel installation:

```text
hyso-guide --open
hyso-guide --language es --topic training.trainer
hyso-guide --path
hyso-guide --list-topics --language de
python -m hyso_guide --version
```

The stdlib-only reader also works in a wheel installation made with `--no-deps`
when Torch is absent. That makes the guide available for installation help;
it does not make the model/training APIs usable without their dependencies.
Opening a browser requires the explicit `--open` option. Otherwise a file URI
or path is printed. The existing seven `hyso` command groups are unchanged.

`content.json` is the editorial source, with parallel tr/en/de/es fields and
per-topic source references. Edit it as UTF-8, then run:

```text
python tools/docs/build_user_guide.py
python tools/docs/build_user_guide.py --check
```

The builder imports no HYSo/Torch code. It generates the HTML and
[`user-guide-topics.json`](../../hyso/data/user-guide-topics.json), the packaged
topic/anchor registry. Keep IDs independent of version and language. New topics
need four complete translations, valid related IDs and checked source references.
Do not rename published topic IDs casually. Rendering escapes text and code;
the CSP permits only the exact embedded navigation script and stylesheet hashes.

`setup.cfg`, `MANIFEST.in` and the commercial source allowlist include the guide
resources and the `hyso_guide.py` reader. The commercial model profile remains
unchanged; development-only architectures are described with explicit limits.
The authoring documents are not needed at runtime and are not copied into the
commercial distribution.

See [language-quality.md](language-quality.md) for semantic and terminology
review, [the guide tests](../../tests/test_user_guide.py) for static/install checks,
and the task QA receipt for actual browser screenshots and execution results.
See [contextual-help.md](contextual-help.md) for hover documentation, the API/topic
inventory and the `hyso docs list/show/open` commands, including editor limits.
The reproducible browser check is `tools/docs/check_user_guide_browser.py`;
pass an already-installed Chromium browser with `--browser` and a fresh
`--output` directory. It uses offline headless mode with GPU disabled. Playwright
is a developer test dependency, not a runtime dependency of HYSo or the guide.

The [release support matrix](../guides/release-support.md) is sourced from `content.json` and embedded in the packaged registry. All four languages describe the P23 CUDA saved-activation exclusion and P31 CUDA long-run deferral.
