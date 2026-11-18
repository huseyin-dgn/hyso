"""CPU-only P13 native admission/command contracts; never starts Nsight or CUDA."""
from pathlib import Path
import subprocess
import pytest

from tools.full_acceptance import p13_capture_native as native
from tools.full_acceptance import p13_capture_finalize as finalize
from tools.full_acceptance import registry


def test_native_capture_is_opt_in_and_separate_from_layer():
    exps = {x["id"]: x for x in registry.experiments()}
    p13 = exps["06.p13_capture"]
    assert p13["handler"] == "p13_capture"
    assert p13["authorization"]["approved"] is False
    assert p13["automatic_retry"] is False
    assert p13["requires"] == ["gpu", "p13_nsys_plan"]
    assert exps["06.layer"]["handler"] == "layer"
    assert registry.FEATURE_EXPERIMENTS["P13"] == ["06.layer"]


def test_profile_command_is_fixed_shell_free_and_scoped(tmp_path):
    nsys = tmp_path / "nsys.exe"
    python = tmp_path / "python.exe"
    source = tmp_path / "p13_capture_worker.py"
    manifest = tmp_path / "manifest.json"
    argv = native.capture_command(nsys, python, source, manifest, tmp_path)
    assert argv[0] == str(nsys)
    assert argv[1] == "profile"
    assert "--trace=cuda,nvtx" in argv
    assert "--sample=none" in argv
    assert "--cpuctxsw=none" in argv
    assert argv[-4:] == ["--manifest", str(manifest), "--output", str(tmp_path)]
    assert all("--kill" not in flag for flag in argv)
    export = native.export_command(nsys, tmp_path)
    assert export[0:3] == [str(nsys), "export", "--type=sqlite"]
    assert export[-1].endswith("p13-capture.nsys-rep")


def test_timeout_is_hard_native_boundary(monkeypatch, tmp_path):
    def timed_out(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])
    monkeypatch.setattr(native.subprocess, "run", timed_out)
    with pytest.raises(Exception) as error:
        native._execute(["nsys"], tmp_path, "mock", 1)
    assert getattr(error.value, "safety_stop", None) == "P13_nested_nsys_timeout"


def test_nested_native_crash_is_hard_boundary(monkeypatch, tmp_path):
    def crash(*args, **kwargs):
        return subprocess.CompletedProcess(args[0], 0xC0000409)
    monkeypatch.setattr(native.subprocess, "run", crash)
    with pytest.raises(Exception) as error:
        native._execute(["nsys"], tmp_path, "mock", 1)
    assert getattr(error.value, "safety_stop", None) == "nested_native_crash"


def test_receipt_refuses_active_controller_without_writing(monkeypatch, tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    out = tmp_path / "native"; out.mkdir()
    (out / "run-status.json").write_text('{"controller_active":true,"preservation_status":"PASS"}')
    monkeypatch.setattr(finalize, "verify", lambda path, **kwargs: {"output": str(out)})
    target = tmp_path / "finalized"
    with pytest.raises(ValueError, match="finalize cleanly"):
        finalize.build(manifest, "fake-hash", target)
    assert not target.exists()


def test_native_readiness_refuses_missing_tool(monkeypatch, tmp_path):
    from tools.full_acceptance.reporting import prerequisites
    manifest = {"optional": {}, "hardware": {"gpu_uuid": "GPU-fake"}}
    experiment = {"requires": ["p13_nsys_plan"]}
    assert any("Nsight" in reason for reason in prerequisites(manifest, experiment))


def test_p13_capture_worker_and_finalizer_never_run_on_import():
    import tools.full_acceptance.p13_capture_worker as worker
    assert callable(worker.main)
    assert callable(finalize.build)
