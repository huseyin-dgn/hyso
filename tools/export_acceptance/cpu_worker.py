"""Admitted CPU-only worker, explicit prehashed existing ORT runtime."""
from pathlib import Path
import json
import os
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'tools/final_acceptance')]
import campaign_runner

if __name__=='__main__':
    request=json.loads(Path(sys.argv[1]).read_text())
    from tools.user_acceptance.cpu_onnx import runtime_options
    options,_=runtime_options(request)
    # CPU Torch was selected by the controller; this site supplies ORT only.
    sys.path.append(options['runtime_search_path'])
    os.environ['HYSO_EXPORT_ATTEMPT']=str(Path(sys.argv[1]).resolve().parent)
    raise SystemExit(campaign_runner.main())
