"""One terminal entry point for prepared, source-pinned HYSo acceptance.

Preparation never authorizes native GPU, training, camera or long-run work.
Run --help or read tools/full_acceptance/README.md before execution.
"""
import sys

# The coordinator also imports helpers from the sealed snapshot. Make ordinary
# terminal invocation read-only before importing any acceptance package code.
sys.dont_write_bytecode = True

from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parent))
from full_acceptance.cli import main

if __name__=='__main__':raise SystemExit(main())
