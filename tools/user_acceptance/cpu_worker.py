"""Admitted CPU pytest payload; retain scenario artifacts outside disposable work."""
from pathlib import Path
import os
import sys
sys.path[:0] = [str(Path(__file__).resolve().parents[2]),
                str(Path(__file__).resolve().parents[1] / 'final_acceptance')]
import campaign_runner

if __name__ == '__main__':
    os.environ['HYSO_USER_ATTEMPT'] = str(Path(sys.argv[1]).resolve().parent)
    raise SystemExit(campaign_runner.main())
