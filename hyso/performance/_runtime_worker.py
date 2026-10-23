"""Stdlib-only admission bootstrap; optional native libraries load after Job assignment."""
import json
import os
from pathlib import Path
import sys
import time


def main():
    root=Path(sys.argv[1]);request=json.loads((root/'request.json').read_text(encoding='utf-8'))
    deadline=time.monotonic()+120
    while not (root/'admit.json').is_file():
        if time.monotonic()>deadline:raise TimeoutError('admission not received')
        time.sleep(.01)
    admit=json.loads((root/'admit.json').read_text(encoding='utf-8'))
    if admit!={'token':request['token'],'pid':os.getpid()}:raise ValueError('invalid process admission')
    os.environ['HYSO_RUNTIME_ADMITTED_PID']=str(os.getpid())
    sys.path[:0]=[request['package_root']]+request['runtime_paths']
    if request['mode']=='stability':
        from hyso.performance.stability import _worker_stability
        _worker_stability(root,request)
    elif request['mode']=='streaming':
        from hyso.performance.streaming import _worker_stream
        _worker_stream(root,request)
    elif request['mode'] in ('devices','providers'):
        from hyso.performance.devices import _worker_dispatch
        _worker_dispatch(root,request)
    else:raise ValueError('unsupported owned runtime mode')


if __name__=='__main__':main()
