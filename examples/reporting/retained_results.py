"""Export retained results, using a synthetic stored training history by default."""
from __future__ import annotations

import argparse
from pathlib import Path

from hyso.analysis import UnifiedAnalysisReport, ModelComparisonResult
from hyso.analysis._codec import canonical, read_json
from hyso.training import TrainingResult
from hyso.reporting import create_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("tr", "en"), default="tr")
    parser.add_argument("--level", choices=("beginner", "technical", "expert"), default="technical")
    args = parser.parse_args(argv)
    paths = args.input or [Path(__file__).parent / "fixtures/training-result-v1.json"]
    readers = {"hyso.training_session_result": lambda d: TrainingResult(canonical(d)),
               "hyso.analysis_report": UnifiedAnalysisReport,
               "hyso.two_model_comparison": ModelComparisonResult}
    results = []
    for path in paths:
        data, _, _ = read_json(path, max_bytes=128 * 1024**2)
        if data.get("format") not in readers:
            parser.error("Supported inputs are TrainingResult, UnifiedAnalysisReport and ModelComparisonResult JSON")
        results.append(readers[data["format"]](data))
    report = create_report(*results, language=args.language, level=args.level, title="HYSo retained results")
    args.output.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output / "report.json")
    report.to_markdown(args.output / "report.md")
    report.to_html(args.output / "report.html")


if __name__ == "__main__":
    main()
