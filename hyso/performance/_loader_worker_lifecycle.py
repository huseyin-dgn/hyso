"""Explicit opt-in shutdown for caller-owned traced DataLoader iterators.

PyTorch's private shutdown can block; isolate it in a daemon thread and enforce
one shared wall deadline. Forced termination never constitutes a clean receipt.
The ordinary P19/P20/P21 measurement APIs do not invoke this helper.
"""
import math
import threading
import time


def shutdown_traced_workers(iterator, *, timeout=8.0):
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("invalid shutdown deadline")
    workers = list(iterator._workers)
    deadline = time.monotonic() + timeout
    errors = []

    def shutdown():
        try:
            iterator._shutdown_workers()
        except BaseException as exc:
            errors.append(repr(exc))

    thread = threading.Thread(target=shutdown, daemon=True)
    thread.start()
    # Reserve time for terminate/kill joins within the same deadline.
    thread.join(max(0, timeout * .6))
    forced = thread.is_alive() or any(w.is_alive() for w in workers)
    if forced:
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
        for worker in workers:
            worker.join(max(0, (deadline - time.monotonic()) * .5))
        for worker in workers:
            if worker.is_alive():
                worker.kill()
        for worker in workers:
            worker.join(max(0, deadline - time.monotonic()))
    states = [dict(worker_id=i, pid=w.pid, alive=w.is_alive(), exitcode=w.exitcode)
              for i, w in enumerate(workers)]
    if forced or errors or any(s["alive"] or s["exitcode"] != 0 for s in states):
        raise RuntimeError(f"unclean worker shutdown: forced={forced}, states={states}, errors={errors}")
    return states
