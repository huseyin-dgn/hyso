"""Use the unchanged pre-import Job, runtime, UUID, allocator and journal gates."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.cuda_acceptance import worker

if __name__ == '__main__':
    # No Torch or scenario module import until worker.run has admitted the payload.
    original = worker.run
    def run(request, folder, journal):
        # Existing worker imports cases only after its real CUDA probe. Install
        # dispatch using a lazy callable, so this adapter never initializes CUDA.
        from tools.cuda_acceptance import feature_evidence
        from tools.user_acceptance.evidence import verify_feature
        feature_evidence.verify_feature = verify_feature
        # cases itself imports Torch; load ONLY inside the admitted run, after
        # admission has been validated by the original worker. A group callable
        # is installed by the post-identity journal hook below.
        emit = journal.emit
        def event(kind, **fields):
            result = emit(kind, **fields)
            if kind == 'gpu_identity':
                from tools.cuda_acceptance import cases
                from tools.user_acceptance.scenarios import gpu_group
                cases.GROUPS = {g: gpu_group for g in request['gpu_plan']['groups'] if g != '01-admission'}
            return result
        journal.emit = event
        return original(request, folder, journal)
    worker.run = run
    raise SystemExit(worker.main())
