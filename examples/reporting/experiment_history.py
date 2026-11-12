"""Export the retained history of an existing local ExperimentManager store."""
from __future__ import annotations

import argparse
from pathlib import Path

from hyso import ExperimentManager
from hyso.reporting import report_experiment


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--language", choices=("tr", "en"), default="tr")
    parser.add_argument("--level", choices=("beginner", "technical", "expert"), default="technical")
    args = parser.parse_args(argv)
    if not (args.store / "experiments.sqlite3").is_file():
        parser.error("An existing ExperimentManager store is required")
    manager = ExperimentManager(args.store)
    try:
        report = report_experiment(manager, args.experiment, language=args.language, level=args.level)
    finally:
        manager.close()
    args.output.mkdir(parents=True, exist_ok=True)
    report.to_json(args.output / "report.json")
    report.to_markdown(args.output / "report.md")
    report.to_html(args.output / "report.html")


if __name__ == "__main__":
    main()
