"""Manual, explicitly authorized 21-task native CUDA diagnostic. Never legacy PASS."""
import argparse
import json
from pathlib import Path
from full_acceptance.cuda_remaining_probe import run_remaining


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--approved-sha256', required=True)
    p.add_argument('--confirm-remaining-21', action='store_true', required=True)
    a = p.parse_args()
    try:
        result = run_remaining(a.manifest, a.approved_sha256)
        print(json.dumps(dict(state=result['state'], tasks_attempted=len(result['tasks_attempted']),
                              stopped_reason=result['stopped_reason'],
                              legacy_04_cuda_accepted=False, suite_complete=False), ensure_ascii=False))
        return 0 if result['state'] == 'DIAGNOSTIC_PASS' else 1
    except (OSError, ValueError, RuntimeError) as exc:
        p.exit(1, type(exc).__name__ + ': ' + str(exc) + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
