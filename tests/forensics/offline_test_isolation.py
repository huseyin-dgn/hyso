"""Run explicit no-Torch assertions independently of the parent pytest conftest.

The entire original unittest method runs in a fresh stdlib-only interpreter;
no assertion is removed and the parent's torch module is never deleted.
"""
from functools import wraps
import os
from pathlib import Path
import subprocess
import sys


def offline_test(method):
    @wraps(method)
    def isolated(self):
        module=method.__module__;case=module+'.'+type(self).__name__+'.'+method.__name__
        if os.environ.get('HYSO_OFFLINE_TEST_CASE')==case:
            return method(self)
        script='''import importlib,sys,unittest
sys.path.insert(0,sys.argv[1])
assert 'torch' not in sys.modules
module=importlib.import_module(sys.argv[2])
case=getattr(module,sys.argv[3])(sys.argv[4])
result=unittest.TextTestRunner().run(unittest.TestSuite([case]))
assert result.testsRun==1 and result.wasSuccessful()
assert 'torch' not in sys.modules
'''
        env=dict(os.environ,HYSO_OFFLINE_TEST_CASE=case,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='-1')
        result=subprocess.run([sys.executable,'-I','-S','-B','-c',script,str(Path(__file__).parent),
            module,type(self).__name__,method.__name__],env=env,capture_output=True,text=True,timeout=60,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
    return isolated
