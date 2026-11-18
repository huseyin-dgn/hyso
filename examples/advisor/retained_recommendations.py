"""Explain retained analysis/comparison evidence; default demo has no measurements."""
from __future__ import annotations

import argparse
from pathlib import Path

from hyso.analysis import AnalysisContext, EvidenceIndex, ModelComparisonResult, UnifiedAnalysisReport
from hyso.analysis._codec import read_json
from hyso.advisor import recommend
from hyso.reporting import create_report


def main() -> None:
    """Export recommendations and Stage 8 reports without measuring or training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Saved UnifiedAnalysisReport or ModelComparisonResult JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("tr", "en"), default="tr")
    parser.add_argument("--level", choices=("beginner", "technical", "expert"), default="technical")
    args = parser.parse_args()
    if args.input is None:
        source = EvidenceIndex().analyze(context=AnalysisContext(workload_id="advisor-empty-demo",
            task="classification", model={}, dataset={}))
    else:
        data, _, _ = read_json(args.input, max_bytes=128 * 1024**2)
        readers = {"hyso.analysis_report": UnifiedAnalysisReport, "hyso.two_model_comparison": ModelComparisonResult}
        if type(data) is not dict or data.get("format") not in readers:
            parser.error("Input must be saved analysis or two-model comparison JSON")
        source = readers[data["format"]](data)
    recommendations = recommend(source)
    report = create_report(recommendations, language=args.language, level=args.level, title="HYSo Performance Advisor")
    args.output.mkdir(parents=True, exist_ok=True)
    recommendations.to_json(args.output / "recommendations.json")
    report.to_json(args.output / "report.json")
    report.to_markdown(args.output / "report.md")
    report.to_html(args.output / "report.html")


if __name__ == "__main__":
    main()
