# HYSo examples

Examples are grouped by user workflow. Each example accepts explicit input/output
paths and reports the scope of its data. Scripts run directly from a checkout
with HYSo installed in the selected Python environment.

| Directory | Available examples |
|---|---|
| `reporting/` | Retained result exports and existing experiment history |
| `advisor/` | Manual recommendations from retained analysis/comparison evidence |
| `analysis/` | Typed native CPU/preflight, retained artifacts, training and ORT analysis, comparisons |
| `models/` | M3 residual/dense families, attention, custom Builder CNN, short CPU training and state_dict |
| `models/python_first.py` | End-to-end branching graph, selected output, real CPU training/inference/benchmark, comparison, ExperimentManager and Advisor |
| `models/image_classification.py` | Explicit PNG/JPEG train/val/test directories, Builder or branching Graph CNN, short CPU training, inference, checkpoint and reports |

Use `python -m examples.models.image_classification --dataset C:/my-images --output C:/local-hyso/image-run-01 --model graph --epochs 2`
from the checkout with a fresh output directory. The dataset must already exist;
this command does not download images. See the [file-backed classification guide](../docs/guides/file-image-classification.md)
for split/channel contracts, Python calls and checkpoint evidence limitations.

Run `python -m examples.models.python_first --output C:/local-hyso/python-demo-01`
with a fresh local output directory. It creates/migrates only its own empty example
experiment store; no previous user workspace is accessed. Synthetic fixture values
do not establish model quality. The preserved 60 component configurations in
`models/component-configs.json` use standard Core Builder JSON without UI metadata.

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
