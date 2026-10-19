"""Build/install the distribution outside the checkout using the existing runtime."""

import ast
import configparser
import email
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import zipfile

import pytest
from packaging.requirements import Requirement

ROOT = Path(__file__).resolve().parents[1]
MODULES = {
    "hyso/models/attention.py",
    "hyso/models/blocks.py",
    "hyso/checkpoint.py",
    "hyso/__init__.py", "hyso/models/__init__.py", "hyso/models/bcam.py",
    "hyso/models/WithSE.py", "hyso/models/fit.py",
    "hyso/augmentation/__init__.py", "hyso/augmentation/augmentation.py",
    "hyso/callbacks/__init__.py", "hyso/callbacks/callbacks.py",
}
DEPENDENCIES = {"torch", "torchvision", "numpy", "matplotlib", "seaborn",
                "scikit-learn", "tqdm", "torchsummary"}
CONFIG_FILES = ("pyproject.toml", "setup.cfg", "MANIFEST.in", "README.md", "LICENSE")


def command(args, cwd):
    env = dict(os.environ, PIP_CONFIG_FILE=os.devnull, PIP_DISABLE_PIP_VERSION_CHECK="1",
               PYTHONDONTWRITEBYTECODE="1", CUDA_VISIBLE_DEVICES="-1", MPLBACKEND="Agg")
    result = subprocess.run([sys.executable, "-I", "-B", *args], cwd=cwd,
                            env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.fixture(scope="module")
def distributions(tmp_path_factory):
    work = tmp_path_factory.mktemp("packaging")
    stage = work / "source"
    stage.mkdir()
    for name in (*CONFIG_FILES, *MODULES):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    # Realistic unwanted content must stay out of both archives.
    for name in ("docs/audit/sentinel.md", "tests/test_sentinel.py",
                 "hyso/dist/old.whl", "hyso/models/weights.pth",
                 "hyso/models/__pycache__/sentinel.pyc"):
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("excluded", encoding="utf-8")
    dist = work / "dist"
    log = command(["-m", "build", "--no-isolation", "--sdist", "--wheel",
                   "--outdir", str(dist), str(stage)], work)
    (work / "build.log").write_text(log, encoding="utf-8")
    return work, next(dist.glob("*.whl")), next(dist.glob("*.tar.gz"))


def test_canonical_packaging_configuration():
    assert (ROOT / "pyproject.toml").is_file()
    assert not (ROOT / "hyso/pyproject.toml").exists()
    assert not (ROOT / "hyso/setup.cfg").exists()
    cfg = configparser.ConfigParser()
    cfg.read(ROOT / "setup.cfg", encoding="utf-8")
    assert cfg["metadata"]["name"] == "hyso"
    assert cfg["metadata"]["version"] == "0.2.0"
    assert cfg["options"]["python_requires"] == ">=3.10"
    assert cfg["options.packages.find"]["where"] == "."
    for name in MODULES:
        ast.parse((ROOT / name).read_text(encoding="utf-8-sig"), feature_version=(3, 10))


def test_wheel_and_sdist_content_and_metadata(distributions):
    _, wheel, sdist = distributions
    assert wheel.name == "hyso-0.2.0-py3-none-any.whl"
    assert sdist.name == "hyso-0.2.0.tar.gz"
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        wheel_meta = email.message_from_bytes(archive.read("hyso-0.2.0.dist-info/METADATA"))
        assert {name for name in wheel_names if name.endswith(".py")} == MODULES
        assert archive.read("hyso-0.2.0.dist-info/top_level.txt").decode().split() == ["hyso"]
        license_names = [name for name in wheel_names if name.endswith("/LICENSE")]
        assert len(license_names) == 1
        assert archive.read(license_names[0]) == (ROOT / "LICENSE").read_bytes()
        for name in MODULES:
            assert archive.read(name) == (ROOT / name).read_bytes()
    with tarfile.open(sdist) as archive:
        names = {name.removeprefix("hyso-0.2.0/") for name in archive.getnames()}
        sdist_meta = email.message_from_bytes(archive.extractfile("hyso-0.2.0/PKG-INFO").read())
        sdist_requires = archive.extractfile("hyso-0.2.0/hyso.egg-info/requires.txt").read().decode().splitlines()
        assert {name for name in names if name.endswith(".py")} == MODULES
        assert set(CONFIG_FILES) <= names
        # setuptools adds egg_info settings to the sdist's setup.cfg.
        for name in ("README.md", "LICENSE", "pyproject.toml", "MANIFEST.in", *MODULES):
            assert archive.extractfile("hyso-0.2.0/" + name).read() == (ROOT / name).read_bytes()
    for name in wheel_names | names:
        assert "sentinel" not in name and not name.endswith((".pth", ".pt", ".ckpt", ".pyc"))
        assert not any(part in {"docs", "tests", "__pycache__", "dist", "build"} for part in Path(name).parts)
    for metadata in (wheel_meta, sdist_meta):
        assert metadata["Name"] == "hyso" and metadata["Version"] == "0.2.0"
        assert metadata["Requires-Python"] == ">=3.10"
        description = metadata.get_payload(decode=True).decode("utf-8")
        # Email metadata may use CRLF; compare every content line unchanged.
        assert description.strip().splitlines() == (ROOT / "README.md").read_text(encoding="utf-8").strip().splitlines()
    requirements = [Requirement(value) for value in wheel_meta.get_all("Requires-Dist")]
    assert {str(Requirement(value)) for value in sdist_requires if value.strip()} == {str(item) for item in requirements}
    assert {item.name for item in requirements} == DEPENDENCIES
    assert str(next(item.specifier for item in requirements if item.name == "torch")) == ">=2.2.2"
    assert str(next(item.specifier for item in requirements if item.name == "torchvision")) == ">=0.17.2"
    assert not next(item.specifier for item in requirements if item.name == "numpy")


SMOKE = """
import importlib, importlib.metadata, json, pathlib, sys
target = pathlib.Path(sys.argv[1]).resolve()
sys.path.insert(0, str(target))
import torch, hyso
torch.set_num_threads(1)
from hyso.models import EBO, ELO, CEMO, EMA, CNNSENormalWithFit
from hyso.augmentation import build_transforms, get_loaders
from hyso.callbacks.callbacks import EarlyStopping, ModelCheckpoint, LRSchedulerCallback
for module, names in {
    'hyso.checkpoint': ('save_training_checkpoint', 'load_training_checkpoint', 'TrainingCheckpointError'),
    'hyso.models.bcam': ('EBO', 'ELO'),
    'hyso.models.blocks': ('ConvNormAct', 'ResidualBlock'),
    'hyso.models.attention': ('ECAAttention', 'CoordinateAttention'),
    'hyso.models.WithSE': ('CEMO', 'EMA'),
    'hyso.models.fit': ('CNNSENormalWithFit',),
    'hyso.augmentation.augmentation': ('build_transforms', 'get_loaders'),
    'hyso.callbacks.callbacks': ('EarlyStopping', 'ModelCheckpoint', 'LRSchedulerCallback'),
}.items():
    loaded = importlib.import_module(module)
    assert pathlib.Path(loaded.__file__).resolve().is_relative_to(target)
    assert all(hasattr(loaded, name) for name in names)
assert pathlib.Path(hyso.__file__).resolve().is_relative_to(target)
assert importlib.metadata.version('hyso') == '0.2.0'
x = torch.randn(2, 3, 16, 16)
models = [
    EBO(conv_channels=[4, 8], residual_per_block=1, device='cpu'),
    ELO(conv_channels=[4, 8], residual_per_block=1),
    CEMO(conv_channels=[4, 8], residuals_per_stage=1, use_ema=False, device='cpu'),
]
with torch.no_grad():
    for model in models:
        output = model.eval()(x)
        assert output.shape == (2, 10) and output.device.type == 'cpu'
        assert torch.isfinite(output).all()
for name, module in list(sys.modules.items()):
    if name == 'hyso' or name.startswith('hyso.'):
        assert pathlib.Path(module.__file__).resolve().is_relative_to(target)
assert not torch.cuda.is_initialized()
print(json.dumps({'source': hyso.__file__, 'version': importlib.metadata.version('hyso'),
                  'cpu_models': 3, 'cuda_initialized': torch.cuda.is_initialized()}))
"""


@pytest.mark.parametrize("kind", ["wheel", "sdist"])
def test_installed_distribution_import_and_cpu_forward(distributions, kind):
    work, wheel, sdist = distributions
    target = work / ("installed-" + kind)
    artifact = wheel if kind == "wheel" else sdist
    command(["-m", "pip", "--isolated", "--no-cache-dir", "install", "--no-index",
             "--no-deps", "--no-build-isolation", "--no-compile",
             "--target", str(target), str(artifact)], work)
    observed = json.loads(command(["-c", SMOKE, str(target)], work).strip().splitlines()[-1])
    assert Path(observed["source"]).resolve().is_relative_to(target.resolve())
    assert observed["cpu_models"] == 3 and observed["cuda_initialized"] is False
    (work / (kind + "-smoke.json")).write_text(json.dumps(observed), encoding="utf-8")
