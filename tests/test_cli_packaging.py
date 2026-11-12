"""Installed CLI smoke from wheel and rebuilt sdist, outside the checkout."""
import configparser
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest
from test_packaging import distributions, command


@pytest.mark.parametrize('kind', ['wheel', 'sdist'])
def test_installed_cli_entry_points(distributions, kind):
    work, wheel, sdist = distributions
    with zipfile.ZipFile(wheel) as archive:
        cfg = configparser.ConfigParser()
        cfg.read_string(archive.read('hyso-0.2.0.dist-info/entry_points.txt').decode())
        assert cfg['console_scripts']['hyso'] == 'hyso.cli:main'
    target = work / ('cli-installed-' + kind)
    command(['-m', 'pip', 'install', '--no-index', '--no-deps', '--no-build-isolation',
             '--target', str(target), str(wheel if kind == 'wheel' else sdist)], work)
    env = dict(os.environ, PYTHONPATH=str(target), PYTHONDONTWRITEBYTECODE='1',
               CUDA_VISIBLE_DEVICES='-1', MPLBACKEND='Agg', PYTHONIOENCODING='utf-8')
    # pip --target uses bin on some Windows/Conda installations too.
    candidates = [target / directory / ('hyso.exe' if os.name == 'nt' else 'hyso')
                  for directory in ('Scripts', 'bin')]
    console = next((p for p in candidates if p.is_file()), None)
    assert console is not None
    for prefix in ([str(console)], [sys.executable, '-B', '-m', 'hyso']):
        proc = subprocess.run([*prefix, '--help'], cwd=work, env=env, capture_output=True, text=True, encoding='utf-8', timeout=90)
        assert proc.returncode == 0, proc.stderr
        assert all(name in proc.stdout for name in ('models', 'train', 'analysis', 'experiments', 'compare', 'advisor', 'reports'))
        proc = subprocess.run([*prefix, 'analysis', 'show', 'P35', '--json'], cwd=work, env=env,
                              capture_output=True, text=True, encoding='utf-8', timeout=90)
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout)['result']['feature_id'] == 'P35'
    # A fresh interpreter fences construction, forward, fit and device queries
    # before importing the installed CLI; help must survive every fence.
    script = r'''
import pathlib, sys
sys.path.insert(0, sys.argv[1])
import torch
def fence(name):
    # Distinct function identities preserve Torch Dynamo's import-time rule map.
    def forbidden(*a, **kw): raise AssertionError('CLI import/help started ' + name)
    return forbidden
torch.nn.Module._call_impl = fence('forward')
for name in ('init', '_lazy_init', 'is_available', 'device_count', 'current_device'):
    setattr(torch.cuda, name, fence(name))
from hyso import ModelBuilder, TrainingSession, Trainer, BenchmarkRunner
ModelBuilder.build = fence('model build')
TrainingSession.fit = fence('session fit')
Trainer.fit = fence('trainer fit')
BenchmarkRunner.run = fence('benchmark')
from hyso.cli import main
import hyso.cli
assert pathlib.Path(hyso.cli.__file__).resolve().is_relative_to(pathlib.Path(sys.argv[1]).resolve())
assert main(['analysis', 'run', '--help']) == 0
'''
    command(['-c', script, str(target)], work)
