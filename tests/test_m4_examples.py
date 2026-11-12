"""Execute every documented M4 example and resolve public typing contracts."""
import inspect
from pathlib import Path
import runpy
import sys
import typing

import pytest
import torch



SCENARIOS = (
    'mobilenet_v2', 'mobilenet_v3_small', 'mobilenet_v3_large', 'efficientnet_v2',
    'ghostnet', 'shufflenet_v2', 'repvgg', 'mobileone',
    'mbconv', 'dropblock', 'builder', 'state_dict',
)


@pytest.mark.parametrize('scenario', SCENARIOS)
def test_documented_cpu_example(scenario, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail('CPU example attempted CUDA work')
    monkeypatch.setattr(torch.cuda, '_lazy_init', forbidden)
    monkeypatch.setattr(sys, 'argv', ['m4_workflows.py', scenario])
    path = Path(__file__).resolve().parents[1] / 'examples/models/m4_workflows.py'
    runpy.run_path(str(path), run_name='__main__')
    assert scenario in capsys.readouterr().out


def test_public_m4_typing_resolves():
    from hyso.models import m4_blocks, m4_reparam, mobile_architectures, reparam_architectures
    for module in (m4_blocks, m4_reparam, mobile_architectures, reparam_architectures):
        for name, value in vars(module).items():
            if name.startswith('_') or getattr(value, '__module__', None) != module.__name__:
                continue
            if inspect.isfunction(value):
                typing.get_type_hints(value)
            elif inspect.isclass(value):
                typing.get_type_hints(value.__init__)
                if 'forward' in value.__dict__:
                    typing.get_type_hints(value.forward)
