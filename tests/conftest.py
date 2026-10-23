"""CPU-only fixtures for the P0.1-B compatibility baseline."""
from fixture_support import load_fixture

import copy
import importlib
import json
import os
from pathlib import Path

# Set visibility before importing torch: CNNS chooses its device automatically.
# This selects a real CPU runtime; no production dependency is stubbed or patched.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["MPLBACKEND"] = "Agg"

import pytest
import torch


MANIFEST = load_fixture("model_state_manifest")


@pytest.fixture(scope="session", autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def seeded_cpu_rng():
    # fork_rng restores the CPU RNG after each independent test.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        yield


@pytest.fixture(params=MANIFEST["models"], ids=lambda case: case["id"])
def model_case(request):
    return copy.deepcopy(request.param)


@pytest.fixture
def manifest_cases():
    return copy.deepcopy(MANIFEST["models"])


@pytest.fixture
def make_model():
    def build(case):
        module_name, class_name = case["import_path"].rsplit(".", 1)
        model_class = getattr(importlib.import_module(module_name), class_name)
        model = model_class(**copy.deepcopy(case["constructor"]))
        assert all(p.device.type == "cpu" for p in model.parameters())
        assert all(b.device.type == "cpu" for b in model.buffers())
        return model

    return build


@pytest.fixture
def cpu_input():
    generator = torch.Generator(device="cpu").manual_seed(17)
    return torch.randn(2, 3, 16, 16, generator=generator, device="cpu")


@pytest.fixture
def quiet_training(monkeypatch):
    """Isolate numeric training tests from summary forwards and GUI output."""
    import matplotlib.pyplot as plt
    from hyso.models import bcam

    # torchsummary may run extra forwards and change BN/RNG state (R21).
    # It is presentation-only here; real forward/loss/backward remain intact.
    monkeypatch.setattr(bcam, "summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
    yield
    plt.close("all")
