"""Unchanged CUDA ownership/runtime probe, with lazy export-only dispatch."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.cuda_acceptance import worker

if __name__=='__main__':
    original=worker.run
    def run(request,folder,journal):
        from tools.cuda_acceptance import feature_evidence
        from tools.export_acceptance.evidence import verify_feature
        feature_evidence.verify_feature=verify_feature
        emit=journal.emit
        def event(kind,**fields):
            value=emit(kind,**fields)
            if kind=='gpu_identity':
                from tools.cuda_acceptance import cases
                from tools.export_acceptance.scenarios import gpu_group
                cases.GROUPS={g:gpu_group for g in request['gpu_plan']['groups'] if g!='01-admission'}
            return value
        journal.emit=event
        return original(request,folder,journal)
    worker.run=run
    raise SystemExit(worker.main())
