"""CPU-only real pretrained weights + synthetic frames, never camera acceptance."""
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
import time

import numpy as np
import pytest
import torch

from tools import p30_mobilenet_adapter as adapter
from tools.full_acceptance.workloads import validate_frame_processor
from tools.full_acceptance import reporting
from hyso.performance import streaming


@pytest.fixture(scope="module")
def processor():
    path = Path(os.environ["LOCALAPPDATA"]) / "hyso-p30-assets/mobilenet_v3_small-047dcff4.pth"
    if not path.exists(): pytest.skip("official local weights unavailable; no download")
    before = adapter._sha256(path)
    value = adapter.build_processor(checkpoint_path=str(path), checkpoint_sha256=before)
    yield value
    value.close()
    assert adapter._sha256(path) == before and not torch.cuda.is_initialized()


def test_pretrained_contract_and_cpu_synthetic_frame(processor, monkeypatch):
    monkeypatch.setattr(torch.cuda, "_lazy_init", lambda *a: pytest.fail("CUDA initialized"))
    identity = processor.identity
    config = {key:copy.deepcopy(identity[key]) for key in
              ("model_id", "input_contract", "output_contract", "preprocessing", "label_semantics")}
    config["checkpoint"] = dict(sha256=identity["checkpoint_sha256"])
    validate_frame_processor(config, identity)
    assert identity["label_semantics"]["class_count"] == 1000
    assert len(identity["label_semantics"]["labels_sha256"]) == 64
    assert identity["output_contract"]["labeled_quality_claim"] is False
    assert [s["name"] for s in processor.process(np.zeros((240, 320, 3), dtype=np.uint8))] == ["preprocess", "inference"]


@pytest.mark.parametrize("frame", [None, object(), np.zeros((0, 1, 3), dtype=np.uint8),
    np.zeros((2, 2, 4), dtype=np.uint8), np.zeros((2, 2, 3), dtype=np.float32)])
def test_invalid_frame_never_enters_model(processor, frame, monkeypatch):
    monkeypatch.setattr(processor.model, "forward", lambda *a: pytest.fail("invalid frame reached inference"))
    with pytest.raises(ValueError): processor.process(frame)


@pytest.mark.parametrize("field", ["model_id", "input_contract", "output_contract", "preprocessing", "label_semantics"])
def test_trained_flag_cannot_mask_contract_drift(processor, field):
    config = {key:copy.deepcopy(processor.identity[key]) for key in
              ("model_id", "input_contract", "output_contract", "preprocessing", "label_semantics")}
    config["checkpoint"] = dict(sha256=processor.identity["checkpoint_sha256"])
    config[field] = "different"
    with pytest.raises(ValueError, match="contract mismatch"): validate_frame_processor(config, processor.identity)


@pytest.mark.parametrize("failure", ["constructor", "open", "metadata", "backend", "journal", "ready"])
def test_startup_failure_closes_processor_and_capture(monkeypatch, tmp_path, failure):
    import cv2
    closed = []
    class Capture:
        def isOpened(self): return failure != "open"
        def get(self, key):
            if failure == "metadata": raise RuntimeError("metadata failure")
            return 25.
        def getBackendName(self):
            if failure == "backend": raise RuntimeError("backend failure")
            return "SYNTHETIC"
        def release(self): closed.append("capture")
    def create(*a):
        if failure == "constructor": raise RuntimeError("constructor failure")
        return Capture()
    def fail(*a, **k): raise RuntimeError("publication failure")
    monkeypatch.setattr(cv2, "VideoCapture", create)
    if failure == "journal": monkeypatch.setattr(streaming, "_Journal", fail)
    if failure == "ready": monkeypatch.setattr(streaming, "_atomic", fail)
    processor = SimpleNamespace(identity={}, close=lambda:closed.append("processor"))
    with pytest.raises(RuntimeError):
        streaming._prepare_capture(tmp_path, dict(token="synthetic"), streaming.StreamProtocol(),
            streaming.StreamSource("camera", 0, "mocked source; no hardware access"), processor, {})
    assert closed == (["processor"] if failure == "constructor" else ["capture", "processor"])


def test_video_provenance_does_not_supply_physical_camera_identity():
    cfg = dict(factory="f:build", factory_source={}, checkpoint={}, model_id="model", input_contract={},
        output_contract={}, preprocessing={}, label_semantics={}, provenance="real recorded video")
    manifest = dict(optional=dict(camera_adapter=dict(configuration=cfg)))
    video = reporting.prerequisites(manifest, dict(id="08.video", requires=["camera_adapter"]))
    camera = reporting.prerequisites(manifest, dict(id="08.camera", requires=["camera_adapter"]))
    assert not video
    assert any("physical camera" in reason for reason in camera)


def test_contract_only_prepare_never_decodes_or_infers(tmp_path, monkeypatch):
    from tools import p30_mobilenet_prepare as preparation
    def forbidden(*a, **k): pytest.fail("contract-only preparation reached capture/inference")
    monkeypatch.setattr(preparation, "validate_video", forbidden)
    monkeypatch.setattr(adapter, "build_processor", forbidden)
    root = Path(os.environ["LOCALAPPDATA"]) / "hyso-p30-assets"
    if not (root / "mobilenet_v3_small-047dcff4.pth").exists(): pytest.skip("no local pretrained asset")
    config = tmp_path / "new-config.json"
    preparation.prepare(SimpleNamespace(offline=True, contract_only=True, skip_smoke=False,
        asset_dir=str(root), camera_index=0, video=None, config=str(config), camera_provenance=None))
    data = json.loads(config.read_text(encoding="utf-8"))
    assert data["camera_provenance"] is None and data["model_id"]


def test_capture_release_failure_signals_consumer_and_cannot_complete(monkeypatch, tmp_path):
    import cv2
    class Capture:
        def isOpened(self): return True
        def get(self, key): return 25.
        def getBackendName(self): return "SYNTHETIC"
        def read(self): return True, np.zeros((2, 2, 3), dtype=np.uint8)
        def release(self): raise RuntimeError("controlled release failure")
    closed = []
    def process(frame):
        stamp = time.perf_counter_ns()
        return [dict(name="inference", start_ns=stamp, end_ns=stamp)]
    processor = SimpleNamespace(identity={}, process=process, close=lambda:closed.append(True))
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a:Capture())
    monkeypatch.setattr(streaming, "_factory", lambda *a:(processor, {}))
    source = streaming.StreamSource("camera", 0, "mocked fixture; no physical camera")
    streaming._worker_stream(tmp_path, dict(source=asdict(source), protocol=asdict(streaming.StreamProtocol(max_frames=1)),
                                           recipe={}, token="synthetic"))
    terminal = json.loads((tmp_path / "terminal.json").read_text())
    assert terminal["status"] == "failed" and terminal["closed"] is False
    assert "release failure" in terminal["reason"] and closed == [True]


@pytest.mark.parametrize("live", [False, True])
@pytest.mark.parametrize("completed", [False, True])
def test_acceptance_binds_source_and_keeps_failed_throughput_null(processor, monkeypatch, tmp_path, live, completed):
    from tools.full_acceptance import handlers
    import hyso.performance
    config = {key:copy.deepcopy(processor.identity[key]) for key in
              ("model_id", "input_contract", "output_contract", "preprocessing", "label_semantics")}
    video = tmp_path / "fixture.avi"
    video.write_bytes(b"synthetic source identity only; never decoded")
    config.update(checkpoint=dict(sha256=processor.identity["checkpoint_sha256"]),
        provenance="synthetic video test", video=dict(path=str(video), sha256=adapter._sha256(video)),
        camera_index=0, camera_provenance="mocked physical-source attestation; no device opened")
    capture = dict(kind="camera" if live else "video_file", location=0 if live else str(video),
        provenance=config["camera_provenance"] if live else config["provenance"],
        file_sha256=None if live else config["video"]["sha256"])
    data = dict(evidence=dict(header=dict(processor=processor.identity, capture=capture)),
        analysis=dict(status="completed" if completed else "aborted", received=1,
            processed=1 if completed else 0, dropped=0 if completed else 1, pending=0,
            completed_processing_fps=1.0 if completed else None))
    monkeypatch.setattr(hyso.performance, "measure_streaming", lambda *a, **k:SimpleNamespace(to_dict=lambda:copy.deepcopy(data)))
    monkeypatch.setattr(handlers, "save", lambda *a, **k:{})
    manifest = dict(output=str(tmp_path), source_snapshot=dict(root=str(tmp_path)),
        optional=dict(camera_adapter=dict(configuration=config)), protocols=dict(camera={}, video={}))
    result = handlers.camera(manifest, tmp_path / "stage", live=live)
    assert result["status"] == ("PASS" if completed else "FAIL")
    assert result["criteria"]["real_camera_trained"] is (live and completed)
    assert result["criteria"]["real_video_trained"] is (not live and completed)
    metric = result["metrics"][0]
    if not completed: assert metric["value"] is None and metric["kind"] == "missing" and metric["reason"]
    capture["location"] = "wrong source"
    with pytest.raises(ValueError, match="source differs"):
        handlers.camera(manifest, tmp_path / "tampered", live=live)
