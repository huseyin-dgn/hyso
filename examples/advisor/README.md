# Performance Advisor examples

Run from the repository root with HYSo installed in the chosen Python environment:

For a checkout without an installed HYSo package, first set the source directory
in the current PowerShell session (from the repository root):

```powershell
$env:PYTHONPATH = (Get-Location).Path
```

```powershell
python examples/advisor/retained_recommendations.py --output C:/Temp/hyso-advisor-empty --language tr --level beginner
python examples/advisor/retained_recommendations.py --input saved-analysis.json --output C:/Temp/hyso-advisor-analysis --language en --level expert
python examples/advisor/retained_recommendations.py --input saved-comparison.json --output C:/Temp/hyso-advisor-comparison --language tr --level technical
```

Use fresh output directories. The default demo contains **no measurements**: its
cards explain evidence gaps without claiming bottlenecks or improvements. Supply
your own saved `UnifiedAnalysisReport` or `ModelComparisonResult` for retained
observations. No workload runs, configuration edits or model loading occur.

Outputs are `recommendations.json` (neutral, rederived on import), `report.json`
(Stage 8 presentation), `report.md` and self-contained `report.html`. Every card
links to retained evidence and manual verification steps. Native units, missing
data and comparison statuses remain unchanged. A conditional candidate stays
blocked until its original plan prerequisites are resolved; it is not an execution
authorization. See the [guide](../../docs/guides/performance-advisor.md).
