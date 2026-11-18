"""Explicit-start outer observer: retain launcher exit and owned Job cleanup."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

import launch_p31 as launcher


def observe(command, directory, *, timeout=7440):
    # Called only after explicit authorization/preflight, or by bounded test doubles.
    from hyso.performance._training_process import _Job
    root=Path(directory);job=_Job(None,None,None)
    process=None;cleanup=None;error=None;reason=None;started=time.monotonic();cleanup_errors=[]
    try:
        with (root/'launcher-stdout.txt').open('xb') as out,(root/'launcher-stderr.txt').open('xb') as err:
            process=subprocess.Popen(command,stdout=out,stderr=err,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'),creationflags=subprocess.CREATE_NO_WINDOW)
            job.assign(process)
            try:process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                reason='outer_watchdog_timeout';raise
    except BaseException as exc:
        error=exc;reason=reason or 'outer_interrupted_or_failed'
    finally:
        try:
            job.terminate()
            if process is not None:
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
            end=time.monotonic()+5
            while True:
                cleanup=job.observation()
                if cleanup['active_processes']==0 or time.monotonic()>=end:break
                time.sleep(.01)
        except BaseException as exc:
            cleanup_errors.append(launcher._error_detail(exc))
            if error is None:error=exc;reason='outer_cleanup_failed'
        finally:
            try:job.close()
            except BaseException as exc:
                cleanup_errors.append(launcher._error_detail(exc))
                if error is None:error=exc;reason='outer_cleanup_failed'
        receipt=dict(status='PARENT_EXIT_OBSERVED' if error is None else 'INTERRUPTED_OR_FAILED',
            launcher_pid=process.pid if process else None,exit_code=process.returncode if process else None,
            reason=reason,cleanup=cleanup,kill_on_close=True,elapsed_seconds=time.monotonic()-started,
            watchdog_seconds=timeout,cleanup_errors=cleanup_errors,automatic_retry=False,P31='PARTIAL_PENDING_REVIEW')
        try:launcher.publish_new(root/'parent-exit.json',receipt)
        except BaseException:
            if error is None:raise
    if error is not None:raise error
    if cleanup['active_processes']!=0:raise RuntimeError('outer owned process tree cleanup failed')
    return process.returncode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence-root',type=Path,required=True)
    parser.add_argument('--start-two-hours',action='store_true')
    args=parser.parse_args()
    if not args.start_two_hours:parser.error('Explicit --start-two-hours required; no workload started')
    launcher.configure_output(args.evidence_root)
    launcher.preflight()  # Source/runtime checks only; still no workload.
    return observe([str(launcher.PYTHON),'-B',str(Path(launcher.__file__).resolve()),
        '--evidence-root',str(launcher.ROOT),'--start-two-hours'],launcher.ROOT)


if __name__=='__main__':raise SystemExit(main())
