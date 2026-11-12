# HYSo examples

Examples are grouped by user workflow. Each example accepts explicit input/output
paths and reports the scope of its data. Scripts run directly from a checkout
with HYSo installed in the selected Python environment.

| Directory | Available examples |
|---|---|
| `reporting/` | Retained result exports and existing experiment history |
| `advisor/` | Manual recommendations from retained analysis/comparison evidence |
| `analysis/` | Typed native CPU/preflight, retained artifacts, training and ORT analysis, comparisons |

Future model construction, training and analysis examples belong in separate
workflow directories when their executable scripts and tests are added. The
current API guides provide those workflows today.

The reporting examples consume existing results. Their default training fixture
contains synthetic stored history for demonstrating report layout.

```powershell
python examples/reporting/retained_results.py --output C:/Temp/hyso-report --language tr --level technical
python examples/reporting/retained_results.py --input saved-analysis.json --input saved-comparison.json --output C:/Temp/hyso-expert --language en --level expert
python examples/reporting/experiment_history.py --store C:/local-experiments/study --experiment EXPERIMENT_ID --output C:/Temp/hyso-history
```

Use a fresh output directory or fresh filenames. The exports retain their
no-clobber behavior. Open `report.html` locally for embedded charts. `report.json`
retains all source snapshots and chart data, and `report.md` contains metric and
scope explanations. HTML graphics use the existing Matplotlib dependency.

## General terminal interface

See [cli/README.md](cli/README.md) for installed `hyso` commands, safe offline
fixtures and deferred typed-binding factories.
