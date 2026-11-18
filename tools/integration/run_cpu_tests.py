"""CPU regression runner with explicit minimum/development runtime provenance."""
import argparse
import faulthandler
import json
import os
from pathlib import Path
import subprocess
import sys

REPO=Path(__file__).resolve().parents[2]
BASE=Path('C:/Users/hdgn5/.conda/envs/torch_gpu')
TOOLING=[BASE/'Lib/site-packages',
    Path('C:/Users/hdgn5/AppData/Local/Temp/hyso-p01b-tests-40awqikl/Lib/site-packages'),
    Path('C:/Users/hdgn5/AppData/Roaming/Python/Python310/site-packages')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile',choices=['development','minimum'],required=True)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('pytest_args',nargs=argparse.REMAINDER)
    args=parser.parse_args();args.evidence.mkdir(parents=True,exist_ok=False)
    for path in TOOLING:
        if str(path).lower() not in [s.lower() for s in sys.path]:sys.path.append(str(path))
    sys.path.insert(0,str(REPO));os.chdir(REPO)
    os.environ.update(CUDA_VISIBLE_DEVICES='-1',MPLBACKEND='Agg',PYTEST_DISABLE_PLUGIN_AUTOLOAD='1',
        PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    import torch,torchvision
    expected=('2.2.2+cpu','0.17.2+cpu') if args.profile=='minimum' else ('2.5.1+cu121','0.20.1+cu121')
    assert (str(torch.__version__),str(torchvision.__version__))==expected
    if args.profile=='minimum':
        assert Path(torch.__file__).resolve().is_relative_to(Path(sys.executable).resolve().parents[1])
        assert Path(torchvision.__file__).resolve().is_relative_to(Path(sys.executable).resolve().parents[1])
    assert not torch.cuda.is_initialized()
    # Ordinary subprocesses and Windows spawn workers must inherit the same
    # primary scientific runtime; isolated owned workers carry explicit paths.
    os.environ['PYTHONPATH']=os.pathsep.join(map(str,[Path(torch.__file__).resolve().parent.parent,
        *TOOLING,REPO/'tests',REPO]))
    identity=dict(python=sys.version,executable=sys.executable,torch=str(torch.__version__),
        torchvision=str(torchvision.__version__),torch_source=torch.__file__,torchvision_source=torchvision.__file__,
        sys_path=sys.path,cuda_initialized=False)
    (args.evidence/'runtime.json').write_text(json.dumps(identity,indent=2))
    import pytest
    class PackagingMinimum:
        def pytest_runtest_setup(self,item):
            if args.profile!='minimum' or item.module.__name__!='test_packaging':return
            def command(argv,cwd):
                env=dict(os.environ,PIP_CONFIG_FILE=os.devnull,PIP_DISABLE_PIP_VERSION_CHECK='1')
                if argv[0]=='-m':
                    launch=[str(BASE/'python.exe'),'-I','-B',*argv]
                else:
                    assert argv[0]=='-c'
                    boot='import sys;sys.path.extend('+repr(list(map(str,TOOLING)))+');'
                    boot+="import torch;assert str(torch.__version__)=='2.2.2+cpu' and torch.version.cuda is None;"
                    boot+='sys.argv='+repr(['-c',*argv[2:]])+';exec(compile('+repr(argv[1])+",'<minimum-installed-smoke>','exec'))"
                    launch=[sys.executable,'-I','-B','-c',boot]
                result=subprocess.run(launch,cwd=cwd,env=env,capture_output=True,text=True,timeout=240)
                count=len(list(args.evidence.glob('packaging-command-*.json')))
                (args.evidence/f'packaging-command-{count}.json').write_text(json.dumps(dict(
                    executable=launch[0],arguments=argv,exit_code=result.returncode,stdout=result.stdout,stderr=result.stderr),indent=2))
                assert result.returncode==0,result.stdout+result.stderr
                return result.stdout
            item.module.command=command
    faulthandler.dump_traceback_later(120,repeat=True)
    extra=args.pytest_args
    if extra[:1]==['--']:extra=extra[1:]
    status=pytest.main(['-c',str(REPO/'tests/pytest.ini'),'-q','--tb=short','--maxfail=1',
        '--basetemp='+str(args.evidence/'work'),'--junitxml='+str(args.evidence/'tests.xml'),*extra],plugins=[PackagingMinimum()])
    faulthandler.cancel_dump_traceback_later()
    assert not torch.cuda.is_initialized()
    (args.evidence/'exit.json').write_text(json.dumps(dict(exit_code=int(status),cuda_initialized=False)))
    return int(status)


if __name__=='__main__':raise SystemExit(main())
