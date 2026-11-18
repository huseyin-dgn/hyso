# Bilingual Advisor presentation — Stage 3B.3C

Human-facing reports support `language="tr"` and `language="en"` with
`audience="summary"` or `audience="technical"`. They consume a retained
AdvisorReport only: no analysis, GoalEvaluation, planning, native resolver,
model, device query or workload is invoked. No translation service or LLM is used.

```python
from hyso.advisor import AdvisorReport

result = AdvisorReport.from_json("existing-canonical-advisor.json")
result.to_markdown("advisor-tr.md", language="tr", audience="summary")
result.to_markdown("advisor-en.md", language="en", audience="summary")
result.to_markdown("advisor-tr-technical.md", language="tr", audience="technical")
result.to_markdown("advisor-en-technical.md", language="en", audience="technical")
result.to_display_json("advisor-tr.display.json", language="tr", audience="summary")
result.to_display_json("advisor-en.display.json", language="en", audience="summary")
```

All exports are explicit UTF-8, deterministic, bounded to 2 MiB and no-clobber.
Methods return text when no path is supplied. `to_display_dict()` returns a
detached dictionary for UI use. Unsupported languages/audiences raise AdvisorError
before creating a file; no OS-locale or fallback translation is guessed.

## Compatibility

`to_markdown(path=None, *, language=None, audience=None)` preserves the exact
historical renderer when both options are omitted/None. The default remains the
old English engineering report. Explicit localization defaults to English if
only audience is supplied, or summary if only language is supplied.

`to_json()` and every canonical field, schema/version, source hash, finding ID,
record/metric path, rule version and evidence status remain unchanged. Language
is not part of AdvisorRequest or AdvisorReport identity. No dataclass field,
engine, planner, goal rule or native evidence format was changed in this stage.

## One semantic model, two languages

`hyso.advisor.presentation.build_display_model(result)` makes one detached,
language-neutral structured snapshot. All languages/audiences receive exactly
the same `data`: typed metric values/units and native scopes/protocols, evidence
references/bindings, finding IDs/kinds, rule versions, goal inputs/outcomes,
plan authorization/effects and original risk/limitation records. Locale changes
only the separate `text` subtree, language/audience metadata and display hash.

The summary contains current performance, goal status, important findings,
next actions, why they matter, evidence gaps, risks and execution status.
It distinguishes resident model P95 from E2E P95, images/s from camera FPS,
process RSS from logical model bytes and native/declared/missing associations.
Accuracy is formatted as percent from its recorded fraction; canonical fractions
are untouched. Turkish uses decimal commas and localized rate/byte labels;
English uses decimal points. Display rounding never writes back to evidence.

Technical detail appends the **complete original canonical record**, marked as
original technical material with untranslated source text. It retains report IDs,
record IDs, exact paths, scope/protocol, eligibility, provenance, associations,
rule versions and acceptance boundaries. The summary does not pretend that
untranslated free-form source material has been interpreted or resolved.

## Deterministic dictionaries and unknown input

Version 1 dictionaries use typed NativeMetric identities and exact rule IDs with
supported rule version 1. Known planner explanations are full bilingual templates,
not replacements inside source sentences. Goal statuses have independent localized
labels; original status codes remain in data. Fixed acceptance-boundary messages
have exact dictionary entries. Other source risks remain explicitly original.

Unknown rule IDs/versions are marked as unavailable localization and preserved in
the original technical record. Untyped observations are not guessed from field
names or prose. Unknown typed metrics fail contract validation. No missing value
is converted to zero, no finding is promoted and no translated text is used as an
instruction or as new measurement/optimization evidence.

## Presentation-only JSON

Display JSON uses `format="hyso.advisor.display"`, `schema_version=1`,
`presentation_only=true`, `authoritative_evidence=false` and localization_version.
It carries the original AdvisorReport identity and original structured technical
record plus localized text. `display_content_sha256` identifies this view only;
different languages may have different display hashes while the source canonical
identity is identical. The canonical AdvisorReport loader rejects display JSON.
There is no display-to-evidence importer and no authority upgrade.

NOT IMPLEMENTED: other languages, translation of arbitrary free text, automatic
language inference, native evidence certification from display files, changes to
GoalEvaluation/planning, executor, ComparisonContract or independent improvement.
P18/P29/P31 boundaries and B4 OPEN remain the source report's unchanged claims.

Focused and retained-MNIST checks:

```text
python -B -m unittest discover -s tests -p "test_advisor*.py" -v
python -B tools/integration/advisor_presentation_acceptance.py --report <3B.3B-auto.json> --legacy-markdown <3B.3B-auto.md> --goal-report <3B.3B-advanced.json> --output <new-directory>
```

Acceptance compares semantic models, raw numeric/reference/status data and
canonical bytes, not merely successful file creation. It also compares the
default Markdown against the actual previous file, exercises both audiences and
blocks analysis/planner/native resolver/workload/device calls during rendering.
